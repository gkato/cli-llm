"""Authenticated, deny-by-default streaming proxy for the DSpark vLLM API."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import secrets
from collections.abc import Iterable

from ml.config import get_dspark_proxy_config


# Keep this list deliberately small. In particular, do not expose vLLM's
# unguarded /invocations, /generative_scoring, /tokenize, or /detokenize routes.
_EXACT_ROUTES = {
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/responses"),
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"),
}
_RESPONSE_ID = re.compile(r"^/v1/responses/[A-Za-z0-9_-]+$")
_RESPONSE_CANCEL = re.compile(r"^/v1/responses/[A-Za-z0-9_-]+/cancel$")

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}
_REPLACED_REQUEST_HEADERS = _HOP_BY_HOP | {
    "authorization",
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def validate_config(config: dict) -> None:
    upstream = str(config.get("upstream_url", ""))
    if not upstream.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("DSpark proxy upstream_url must be loopback HTTP")
    if int(config.get("max_concurrency", 4)) < 1:
        raise ValueError("DSpark proxy max_concurrency must be at least 1")
    if int(config.get("max_request_bytes", 50_000_000)) < 1:
        raise ValueError("DSpark proxy max_request_bytes must be positive")


def is_allowed_route(method: str, path: str) -> bool:
    """Return whether one method/path pair may reach vLLM."""
    normalized_method = method.upper()
    if (normalized_method, path) in _EXACT_ROUTES:
        return True
    if normalized_method in {"GET", "DELETE"} and _RESPONSE_ID.fullmatch(path):
        return True
    return normalized_method == "POST" and bool(_RESPONSE_CANCEL.fullmatch(path))


def is_authorized(authorization: str | None, api_key: str) -> bool:
    if not authorization or not api_key:
        return False
    scheme, separator, supplied = authorization.partition(" ")
    return (
        bool(separator)
        and scheme.lower() == "bearer"
        and secrets.compare_digest(supplied, api_key)
    )


def build_upstream_headers(
    incoming: Iterable[tuple[str, str]],
    api_key: str,
    client_host: str | None,
    scheme: str,
) -> dict[str, str]:
    """Replace caller-controlled auth and forwarding metadata exactly once."""
    headers = {
        key: value
        for key, value in incoming
        if key.lower() not in _REPLACED_REQUEST_HEADERS
    }
    headers["Authorization"] = f"Bearer {api_key}"
    if client_host:
        headers["X-Forwarded-For"] = client_host
    headers["X-Forwarded-Proto"] = scheme
    return headers


def main() -> None:
    import httpx
    import uvicorn
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask

    args = _parse_args()
    config = get_dspark_proxy_config()
    validate_config(config)
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        raise RuntimeError("API_KEY is required; the DSpark proxy fails closed")

    timeout = float(config.get("request_timeout_seconds", 3600))
    max_request_bytes = int(config.get("max_request_bytes", 50_000_000))
    max_concurrency = int(config.get("max_concurrency", 4))
    upstream_url = str(config["upstream_url"]).rstrip("/")
    health_path = str(config.get("upstream_health_path", "/v1/models"))
    inference_slots = asyncio.Semaphore(max_concurrency)
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    app = FastAPI(
        title="ml-compute DSpark safety proxy",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    def authorize(authorization: str | None) -> None:
        if not is_authorized(authorization, api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await client.aclose()

    @app.get("/health")
    async def health() -> JSONResponse:
        try:
            response = await client.get(
                upstream_url + health_path,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5,
            )
            ready = response.is_success
        except httpx.HTTPError:
            ready = False
        return JSONResponse(
            {"status": "ok" if ready else "degraded", "ready": ready},
            status_code=200 if ready else 503,
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    async def proxy(
        path: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        request_path = "/" + path
        if not is_allowed_route(request.method, request_path):
            raise HTTPException(status_code=404, detail="Route is not exposed")

        body = await request.body()
        if len(body) > max_request_bytes:
            raise HTTPException(status_code=413, detail="Request body is too large")

        headers = build_upstream_headers(
            request.headers.items(),
            api_key,
            request.client.host if request.client else None,
            request.url.scheme,
        )

        await inference_slots.acquire()
        try:
            upstream_request = client.build_request(
                request.method,
                upstream_url + request_path,
                params=request.query_params,
                headers=headers,
                content=body,
            )
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            inference_slots.release()
            raise HTTPException(
                status_code=502,
                detail=f"DSpark backend is unavailable: {type(exc).__name__}",
            ) from exc
        except Exception:
            inference_slots.release()
            raise

        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _HOP_BY_HOP
        }
        response_headers["X-Content-Type-Options"] = "nosniff"

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

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
