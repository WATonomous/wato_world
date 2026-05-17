"""2D object detector wrapper (GroundingDINO).

Lazy-imports the model so the module can be imported even when groundingdino
is not installed.  Returns empty results in that case with a one-time warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_warned_missing = False


@dataclass
class Detection:
    """Single 2D detection from the detector."""

    bbox_xyxy: np.ndarray  # (4,) float32: [x1, y1, x2, y2] in pixels
    class_name: str  # raw detector label (synonym)
    score: float


class GroundingDINODetector:
    """GroundingDINO text-prompted 2D object detector.

    Usage:
        det = GroundingDINODetector(model_id="IDEA-Research/grounding-dino-tiny")
        detections = det.detect(image_rgb, ["car", "pedestrian", ...], threshold=0.25)

    Falls back to empty detections if groundingdino is not installed.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._device = device or self._default_device()
        self._model = None  # lazy-loaded on first detect() call
        self._processor = None

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> bool:
        """Lazy-load model.  Returns False if unavailable."""
        global _warned_missing
        if self._model is not None:
            return True
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self._model_id
            ).to(self._device)
            log.info("GroundingDINO loaded (%s) on %s", self._model_id, self._device)
            return True
        except Exception as exc:  # noqa: BLE001
            if not _warned_missing:
                log.warning(
                    "GroundingDINO unavailable (%s) — returning empty detections. "
                    "Install: pip install transformers groundingdino-py",
                    exc,
                )
                _warned_missing = True
            return False

    def detect(
        self,
        image_rgb: np.ndarray,
        text_prompts: list[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        """Detect objects in image_rgb (H×W×3 uint8).

        text_prompts is a flat list of synonym strings; they are joined with
        ' . ' as required by GroundingDINO's text encoder.
        Returns a list of Detection objects (may be empty).
        """
        if not self._load():
            return []

        import torch
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image_rgb)
        query = " . ".join(text_prompts)
        inputs = self._processor(images=pil_img, text=query, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_img.size[::-1]],
        )[0]

        detections: list[Detection] = []
        for box, score, label in zip(
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            results["labels"],
        ):
            detections.append(
                Detection(
                    bbox_xyxy=box.astype(np.float32),
                    class_name=str(label),
                    score=float(score),
                )
            )
        return detections
