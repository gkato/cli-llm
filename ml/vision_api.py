"""HTTP API for registry-backed Transformers text detection models."""

import argparse
import asyncio
import os
import time
from io import BytesIO


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--box-threshold", type=float, default=0.45)
    parser.add_argument("--max-candidates", type=int, default=3000)
    parser.add_argument("--unclip-ratio", type=float, default=1.4)
    parser.add_argument("--max-image-pixels", type=int, default=40_000_000)
    parser.add_argument("--max-image-bytes", type=int, default=25_000_000)
    return parser.parse_args()


class TextDetectionService:
    def __init__(self, model_id: str, device: str, dtype: str):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype_by_name = {
            "auto": None,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_by_name:
            raise ValueError(f"Unsupported vision dtype: {dtype}")
        model_dtype = dtype_by_name[dtype]
        if device == "cpu" and model_dtype in (torch.float16, torch.bfloat16):
            model_dtype = torch.float32

        load_kwargs = {"dtype": model_dtype} if model_dtype is not None else {}
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_id, **load_kwargs
        ).to(device)
        self.model.eval()
        self.device = device
        self.dtype = model_dtype or next(self.model.parameters()).dtype

    def detect(
        self,
        image_bytes: bytes,
        *,
        threshold: float,
        box_threshold: float,
        max_candidates: int,
        unclip_ratio: float,
        max_image_pixels: int,
    ) -> tuple[dict, float]:
        import torch
        from PIL import Image, UnidentifiedImageError

        started = time.perf_counter()
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Request body is not a supported image") from exc
        width, height = image.size
        if width * height > max_image_pixels:
            raise ValueError(
                f"Image has {width * height:,} pixels; limit is "
                f"{max_image_pixels:,}"
            )

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if pixel_values.is_floating_point():
            pixel_values = pixel_values.to(self.dtype)

        # Paddle's image processor records the original image dimensions as a
        # tensor during preprocessing.  Its post-processor calls tensor
        # methods on this value, so rebuilding it as a list of Python tuples
        # raises an AttributeError at inference time.  Keep the processor's
        # value when available, with a tensor fallback for compatible custom
        # processors that do not return target_sizes.
        target_sizes = inputs.get("target_sizes")
        if target_sizes is None:
            target_sizes = torch.tensor(
                [[height, width]], dtype=torch.float32
            )

        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)

        # PP-OCR post-processing converts the probability map to NumPy and
        # passes it through OpenCV.  Cast FP16/BF16 model output to FP32 first;
        # those lower-precision NumPy dtypes are not portable across OpenCV
        # builds.
        outputs.last_hidden_state = outputs.last_hidden_state.float()
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            box_threshold=box_threshold,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
        )[0]

        detections = []
        boxes = results["boxes"].detach().cpu().tolist()
        scores = results["scores"].detach().cpu().tolist()
        for box, score in zip(boxes, scores):
            if box and isinstance(box[0], list):
                polygon = [
                    {"x": float(point[0]), "y": float(point[1])}
                    for point in box
                ]
                xs = [point["x"] for point in polygon]
                ys = [point["y"] for point in polygon]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                xmin, ymin, xmax, ymax = (float(value) for value in box)
                bbox = [xmin, ymin, xmax, ymax]
                polygon = [
                    {"x": xmin, "y": ymin},
                    {"x": xmin, "y": ymax},
                    {"x": xmax, "y": ymax},
                    {"x": xmax, "y": ymin},
                ]
            detections.append(
                {"score": float(score), "bbox": bbox, "polygon": polygon}
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "image": {"width": width, "height": height},
            "detections": detections,
        }, elapsed_ms


def main() -> None:
    import uvicorn
    from fastapi import FastAPI, Header, HTTPException, Query, Request

    args = _parse_args()
    if args.max_concurrency < 1:
        raise ValueError("max-concurrency must be at least 1")
    service = TextDetectionService(args.model_id, args.device, args.dtype)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    api_key = os.getenv("API_KEY")
    app = FastAPI(title="ml-compute text detection")

    def authorize(authorization: str | None) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": args.served_model_name}

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)) -> dict:
        authorize(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "owned_by": "ml-compute",
                }
            ],
        }

    @app.post("/v1/text/detections")
    async def detect(
        request: Request,
        authorization: str | None = Header(default=None),
        threshold: float = Query(default=args.threshold, ge=0.0, le=1.0),
        box_threshold: float = Query(
            default=args.box_threshold, ge=0.0, le=1.0
        ),
        max_candidates: int = Query(
            default=args.max_candidates, ge=1, le=10_000
        ),
        unclip_ratio: float = Query(default=args.unclip_ratio, gt=0.0, le=10.0),
    ) -> dict:
        authorize(authorization)
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=415,
                detail=(
                    "Send raw image bytes with Content-Type: "
                    "image/png or image/jpeg"
                ),
            )
        image_bytes = await request.body()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image body")
        if len(image_bytes) > args.max_image_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {args.max_image_bytes} byte limit",
            )

        try:
            async with semaphore:
                result, elapsed_ms = await asyncio.to_thread(
                    service.detect,
                    image_bytes,
                    threshold=threshold,
                    box_threshold=box_threshold,
                    max_candidates=max_candidates,
                    unclip_ratio=unclip_ratio,
                    max_image_pixels=args.max_image_pixels,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "object": "text_detection",
            "model": args.served_model_name,
            **result,
            "processing_ms": round(elapsed_ms, 2),
        }

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
