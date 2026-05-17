"""2D object detector wrappers + ensemble.

Defines:
- `Detection` — common box dataclass returned by every detector.
- `DetectorBase` — Protocol describing the detect() interface every adapter
  must satisfy (GroundingDINODetector, YOLOWorldDetector, mocks in tests).
- `GroundingDINODetector` — text-prompted detector via HuggingFace
  transformers.  Pre-existing.
- `DetectorEnsemble` — runs multiple detectors per frame and merges by IoU
  within class.  Used to combine GroundingDINO + YOLO-World per
  docs/research/detector_ensemble_guidance.md.

Each adapter lazy-imports its underlying model package so this module can
be imported (and tested) without any GPU model installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

_warned_missing = False


@dataclass
class Detection:
    """Single 2D detection from a detector.

    `detector_name` is populated by adapters that emit through DetectorEnsemble
    so the ensemble can track which detector contributed which box.  Optional
    so existing producers don't need to change.
    """

    bbox_xyxy: np.ndarray  # (4,) float32: [x1, y1, x2, y2] in pixels
    class_name: str        # raw detector label (synonym)
    score: float
    detector_name: Optional[str] = None


@runtime_checkable
class DetectorBase(Protocol):
    """Structural type for all 2D detectors that feed perception_2d's pipeline.

    Every adapter (GroundingDINODetector, YOLOWorldDetector, test mocks, ...)
    must expose `detect(image_rgb, text_prompts, box_threshold, text_threshold)`
    returning `list[Detection]`.  Using `Protocol` rather than ABC keeps the
    adapters duck-typed and avoids forcing inheritance.
    """

    def detect(
        self,
        image_rgb: np.ndarray,
        text_prompts: list[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> list[Detection]: ...


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
        self._model = None   # lazy-loaded on first detect() call
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
            detections.append(Detection(
                bbox_xyxy=box.astype(np.float32),
                class_name=str(label),
                score=float(score),
                detector_name="grounding_dino",
            ))
        return detections


# ---------------------------------------------------------------------------
# Detector ensemble.
# ---------------------------------------------------------------------------


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Single-pair IoU over xyxy boxes.  Returns 0 if either box is degenerate."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1); inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class DetectorEnsemble:
    """Run multiple detectors on the same image and merge by IoU within class.

    Per docs/research/detector_ensemble_guidance.md.  When two detectors fire
    on the same object (IoU > `iou_threshold` and matching class), keep the
    highest-score detection and tag it with the contributing detector names.
    Singletons (no overlap) are kept as-is.

    Notes:
      * Matching is per *raw* class name (the synonym output by each detector).
        Class harmonization to canonical names happens later in the pipeline
        via `ComponentConfig.class_from_synonym`.
      * "Highest score wins" is the simplest fusion rule — Weighted Box Fusion
        / KBF (MS3D++) is a future iteration documented in the research note.
    """

    def __init__(
        self,
        detectors: list[DetectorBase],
        iou_threshold: float = 0.6,
    ) -> None:
        if not detectors:
            raise ValueError("DetectorEnsemble requires at least one detector")
        self._detectors = list(detectors)
        self._iou_threshold = float(iou_threshold)

    def detect(
        self,
        image_rgb: np.ndarray,
        text_prompts: list[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        all_dets: list[Detection] = []
        for det in self._detectors:
            all_dets.extend(
                det.detect(
                    image_rgb,
                    text_prompts,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            )
        return self._merge_by_iou(all_dets)

    def _merge_by_iou(self, detections: list[Detection]) -> list[Detection]:
        """Group detections of the same class with IoU > threshold; keep best score per group."""
        if not detections:
            return []
        # Sort by score descending — greedy claim of overlapping boxes.
        sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
        kept: list[Detection] = []
        for cand in sorted_dets:
            merged = False
            for k in kept:
                if k.class_name == cand.class_name and _box_iou(cand.bbox_xyxy, k.bbox_xyxy) > self._iou_threshold:
                    # Record overlap in detector_name for downstream provenance.
                    if k.detector_name and cand.detector_name and cand.detector_name not in k.detector_name:
                        k.detector_name = f"{k.detector_name}+{cand.detector_name}"
                    merged = True
                    break
            if not merged:
                kept.append(cand)
        return kept
