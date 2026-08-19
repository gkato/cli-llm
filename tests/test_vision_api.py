import unittest
from io import BytesIO
from types import SimpleNamespace

import torch
from PIL import Image

from ml.vision_api import TextDetectionService


class _FakeProcessor:
    def __init__(self):
        self.target_sizes = torch.tensor([[24.0, 32.0]])
        self.postprocess_target_sizes = None
        self.postprocess_dtype = None

    def __call__(self, *, images, return_tensors):
        return {
            "pixel_values": torch.zeros((1, 3, 24, 32)),
            "target_sizes": self.target_sizes,
        }

    def post_process_object_detection(self, outputs, **kwargs):
        self.postprocess_target_sizes = kwargs["target_sizes"]
        self.postprocess_dtype = outputs.last_hidden_state.dtype
        return [
            {
                "boxes": torch.tensor([[1.0, 2.0, 11.0, 12.0]]),
                "scores": torch.tensor([0.9]),
            }
        ]


class _FakeModel:
    def __call__(self, *, pixel_values):
        return SimpleNamespace(
            last_hidden_state=torch.zeros(
                (1, 1, 24, 32), dtype=torch.float16
            )
        )


class VisionApiTests(unittest.TestCase):
    def test_detect_preserves_target_sizes_and_casts_for_postprocessing(self):
        processor = _FakeProcessor()
        service = TextDetectionService.__new__(TextDetectionService)
        service.processor = processor
        service.model = _FakeModel()
        service.device = "cpu"
        service.dtype = torch.float32

        image_buffer = BytesIO()
        Image.new("RGB", (32, 24)).save(image_buffer, format="PNG")

        result, _elapsed_ms = service.detect(
            image_buffer.getvalue(),
            threshold=0.2,
            box_threshold=0.45,
            max_candidates=3000,
            unclip_ratio=1.4,
            max_image_pixels=1_000_000,
        )

        self.assertIs(
            processor.postprocess_target_sizes, processor.target_sizes
        )
        self.assertEqual(processor.postprocess_dtype, torch.float32)
        self.assertEqual(result["image"], {"width": 32, "height": 24})
        self.assertEqual(len(result["detections"]), 1)


if __name__ == "__main__":
    unittest.main()
