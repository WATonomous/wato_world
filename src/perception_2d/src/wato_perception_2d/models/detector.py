"""Open-vocabulary 2D detector — the box source for SAM2.

Replaces SAM 3.1's concept-everything segmentation with a detector that emits a
*bounded, deduplicated, class-labeled* set of boxes per frame. Those boxes prompt
SAM2 (sam2_tracker.py), so downstream gets one masklet per detected object rather
than every region matching a text concept.

Default backend is GroundingDINO via HuggingFace Transformers
(``AutoModelForZeroShotObjectDetection``, e.g. ``IDEA-Research/grounding-dino-base``),
which needs no CUDA custom-op compile. The model is text-prompted with the
taxonomy class names (one prompt phrase per class), so it stays open-vocab
capable but constrained to our classes.

Lazy-imports transformers so this module can be *imported* without it installed;
calling detect() without it raises loudly (no degraded fallback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detected object in a single frame."""

    box_xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    cls: str  # canonical taxonomy class
    score: float
    phrase: str = ""  # raw detector label before canonicalisation


# A (prompt_text, canonical_class) concept — same shape the tracker/config use.
Concept = tuple[str, str]


def _box_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """IoU of two (x1, y1, x2, y2) boxes; 0 when they don't overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def nms_per_class(dets: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy per-class non-max suppression.

    GroundingDINO can return several near-duplicate boxes for the same object
    (one per matched synonym token). Suppress lower-scoring boxes that overlap a
    kept box of the *same* canonical class above ``iou_threshold``. Boxes of
    different classes never suppress each other (a pedestrian on a bike is two
    objects). Pure function — unit-tested without a model.
    """
    keep: list[Detection] = []
    for det in sorted(dets, key=lambda d: d.score, reverse=True):
        if any(
            k.cls == det.cls and _box_iou(k.box_xyxy, det.box_xyxy) >= iou_threshold
            for k in keep
        ):
            continue
        keep.append(det)
    return keep


def match_label_to_class(label: str, concepts: list[Concept]) -> Optional[str]:
    """Map a detector's raw label phrase back to a canonical taxonomy class.

    GroundingDINO returns the matched prompt substring (e.g. "car", or a merged
    "car truck"). We pick the concept whose prompt text overlaps the label;
    longest prompt-text match wins so "pickup truck" beats "truck". Returns None
    when nothing matches (caller drops the detection — keeps output on-taxonomy).
    Pure function — unit-tested without a model.
    """
    lbl = label.strip().lower()
    if not lbl:
        return None
    best: Optional[str] = None
    best_len = 0
    for prompt_text, canonical in concepts:
        pt = prompt_text.strip().lower()
        if not pt:
            continue
        if (pt in lbl or lbl in pt) and len(pt) > best_len:
            best = canonical
            best_len = len(pt)
    return best


def build_text_prompt(concepts: list[Concept]) -> str:
    """GroundingDINO text prompt: lowercased prompt phrases joined by ' . '.

    The trailing ' .' separator is the format the GroundingDINO processor expects
    for multi-class grounding.
    """
    phrases = [pt.strip().lower() for pt, _ in concepts if pt and pt.strip()]
    # de-dup while preserving order (two classes may share a prompt synonym)
    seen: set[str] = set()
    uniq = [p for p in phrases if not (p in seen or seen.add(p))]
    return " . ".join(uniq) + " ." if uniq else ""


class GroundingDinoDetector:
    """GroundingDINO open-vocabulary detector (HuggingFace Transformers backend).

    Raises RuntimeError if transformers / the checkpoint is unavailable — no
    degraded fallback (perception_2d fails loud).
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        *,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_iou: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._nms_iou = nms_iou
        self._device = device or self._default_device()
        self._model = None
        self._processor = None

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )

            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(self._model_id)
                .to(self._device)
                .eval()
            )
            log.info("GroundingDINO loaded (%s) on %s", self._model_id, self._device)
        except Exception as exc:
            self._model = None
            raise RuntimeError(
                f"GroundingDINO model {self._model_id!r} could not be loaded: {exc}. "
                "Install 'transformers' and fetch the checkpoint "
                "(scripts/fetch_models.py). perception_2d does not fall back."
            ) from exc

    def detect(self, image_rgb: np.ndarray, concepts: list[Concept]) -> list[Detection]:
        """Detect taxonomy objects in one RGB image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            concepts: (prompt_text, canonical_class) per taxonomy class. The
                prompt texts form the GroundingDINO text query; predicted labels
                are mapped back to canonical classes via match_label_to_class.

        Returns:
            Per-class-NMS'd list of Detection. Empty if nothing scores above the
            thresholds.

        Raises:
            RuntimeError: if transformers / the checkpoint is unavailable.
        """
        text = build_text_prompt(concepts)
        if not text:
            return []
        self._load()

        import torch
        from PIL import Image as PILImage

        H, W = image_rgb.shape[:2]
        pil = PILImage.fromarray(image_rgb)
        inputs = self._processor(images=pil, text=text, return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(H, W)],
        )[0]

        dets: list[Detection] = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]  # list[str]
        for box, score, label in zip(boxes, scores, labels):
            canonical = match_label_to_class(str(label), concepts)
            if canonical is None:
                continue  # off-taxonomy phrase — drop to keep output clean
            x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            dets.append(
                Detection(
                    box_xyxy=(x1, y1, x2, y2),
                    cls=canonical,
                    score=float(score),
                    phrase=str(label),
                )
            )
        return nms_per_class(dets, self._nms_iou)


def detector_importable() -> bool:
    """Cheap check that the detector backend (transformers) imports.

    Mirrors sam2_importable(): the pipeline calls this before the depth pass so a
    missing detector install fails fast rather than after the whole depth pass.
    """
    import importlib

    try:
        importlib.import_module("transformers")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Detector backend (`transformers`) not importable: %s", exc)
        return False
