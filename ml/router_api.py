"""Single-port streaming reverse proxy for independently loaded models."""

import argparse
import asyncio
import json
import os

from ml.config import get_router_config


class RoutingError(ValueError):
    """A request could not be mapped to a configured backend."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def _model_routes(config: dict) -> dict[str, str]:
    routes: dict[str, str] = {}
    for backend_name, backend in (config.get("backends") or {}).items():
        for model in backend.get("models") or []:
            if model in routes and routes[model] != backend_name:
                raise ValueError(f"Model route {model!r} is configured twice")
            routes[str(model)] = str(backend_name)
    return routes


def validate_config(config: dict) -> None:
    backends = config.get("backends") or {}
    if not backends:
        raise ValueError("Router has no backends")
    if int(config.get("max_concurrency", 1)) < 1:
        raise ValueError("Router max_concurrency must be at least 1")
    for name, backend in backends.items():
        if not backend.get("url"):
            raise ValueError(f"Router backend {name!r} has no URL")
        if not backend.get("models"):
            raise ValueError(f"Router backend {name!r} has no model IDs")
    _model_routes(config)
    for path, backend_name in (config.get("path_routes") or {}).items():
        if not str(path).startswith("/"):
            raise ValueError(f"Router path {path!r} must start with /")
        if backend_name not in backends:
            raise ValueError(
                f"Router path {path!r} references unknown backend {backend_name!r}"
            )


def route_request(
    config: dict,
    *,
    path: str,
    content_type: str,
    body: bytes,
    model_header: str | None,
) -> tuple[str, bytes]:
    """Return ``(backend_name, forwarded_body)`` for one request."""
    model_from_body = None
    json_body = None
    if content_type.lower().startswith("application/json") and body:
        try:
            json_body = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RoutingError("Request body is not valid JSON") from exc
        if isinstance(json_body, dict):
            model_from_body = json_body.get("model")

    if model_header and model_from_body and model_header != model_from_body:
        raise RoutingError("X-Model and JSON model fields disagree")
    requested_model = model_header or model_from_body
    routes = _model_routes(config)

    if requested_model:
        backend_name = routes.get(str(requested_model))
        if not backend_name:
            raise RoutingError(f"Unknown model: {requested_model}")
    else:
        backend_name = (config.get("path_routes") or {}).get(path)
        if not backend_name:
            raise RoutingError(
                "Request must include a configured JSON model field or X-Model header"
            )

    backend = config["backends"][backend_name]
    served_model = backend.get("served_model")
    if json_body is not None and isinstance(json_body, dict) and served_model:
        json_body["model"] = served_model
        body = json.dumps(json_body, separators=(",", ":")).encode()
    return str(backend_name), body


def advertised_models(config: dict) -> list[dict]:
    return [
        {"id": model, "object": "model", "owned_by": "ml-compute-router"}
        for model in _model_routes(config)
    ]


def main() -> None:
    import httpx
    import uvicorn
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask

    args = _parse_args()
    config = get_router_config()
    validate_config(config)
    api_key = os.getenv("API_KEY")
    timeout = float(config.get("request_timeout_seconds", 600))
    max_request_bytes = int(config.get("max_request_bytes", 50_000_000))
    max_concurrency = int(config.get("max_concurrency", 1))
    inference_slots = asyncio.Semaphore(max_concurrency)
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    app = FastAPI(title="ml-compute model router")

    def authorize(authorization: str | None) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def backend_health(name: str, backend: dict) -> tuple[str, dict]:
        url = str(backend["url"]).rstrip("/") + str(
            backend.get("health_path", "/v1/models")
        )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            response = await client.get(url, headers=headers, timeout=5)
            return name, {"ready": response.is_success, "status": response.status_code}
        except httpx.HTTPError as exc:
            return name, {"ready": False, "error": type(exc).__name__}

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await client.aclose()

    @app.get("/health")
    async def health() -> JSONResponse:
        results = await asyncio.gather(
            *(
                backend_health(name, backend)
                for name, backend in config["backends"].items()
            )
        )
        backends = dict(results)
        ready = all(result["ready"] for result in backends.values())
        return JSONResponse(
            {
                "status": "ok" if ready else "degraded",
                "ready": ready,
                "backends": backends,
            }
        )

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)) -> dict:
        authorize(authorization)
        return {"object": "list", "data": advertised_models(config)}

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy(
        path: str,
        request: Request,
        authorization: str | None = Header(default=None),
        x_model: str | None = Header(default=None, alias="X-Model"),
    ):
        authorize(authorization)
        body = await request.body()
        if len(body) > max_request_bytes:
            raise HTTPException(status_code=413, detail="Request body is too large")
        request_path = "/" + path
        try:
            backend_name, forwarded_body = route_request(
                config,
                path=request_path,
                content_type=request.headers.get("content-type", ""),
                body=body,
                model_header=x_model,
            )
        except RoutingError as exc:
            status_code = 404 if str(exc).startswith("Unknown model:") else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

        backend = config["backends"][backend_name]
        target = str(backend["url"]).rstrip("/") + request_path
        excluded_request_headers = {"host", "content-length", "x-model"}
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in excluded_request_headers
        }
        if request.client:
            headers["x-forwarded-for"] = request.client.host

        await inference_slots.acquire()
        try:
            upstream_request = client.build_request(
                request.method,
                target,
                params=request.query_params,
                headers=headers,
                content=forwarded_body,
            )
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            inference_slots.release()
            raise HTTPException(
                status_code=502,
                detail=f"Backend {backend_name} is unavailable: {type(exc).__name__}",
            ) from exc
        except Exception:
            inference_slots.release()
            raise

        hop_by_hop = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
            "content-length",
        }
        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in hop_by_hop
        }

        async def close_upstream() -> None:
            try:
                await upstream.aclose()
            finally:
                inference_slots.release()

        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(close_upstream),
        )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
