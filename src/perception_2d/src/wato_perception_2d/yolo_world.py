"""YOLO-World open-vocabulary 2D object detector (ultralytics wrapper).

Parallel detector branch to GroundingDINODetector, per
docs/research/detector_ensemble_guidance.md.  Same `detect()` signature so
the ensemble wrapper in `detector.py` can call them interchangeably.

YOLO-World accepts a list of free-form class strings at inference time
(open-vocabulary via CLIP text embeddings) — same prompt taxonomy as
GroundingDINO.  Useful for rare-class recall (construction workers, debris)
that GroundingDINO sometimes misses.

Lazy-imports ultralytics so the module imports even without the package
(in which case `detect` returns an empty list with a one-time warning).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from wato_common.artifact_store import detector_checkpoint_path
from wato_perception_2d.detector import Detection

log = logging.getLogger(__name__)

_warned_missing = False


class YOLOWorldDetector:
    """YOLO-World detector with lazy ultralytics model load.

    Default checkpoint: $MODELS_ROOT/yolo_world/yolov8l-worldv2.pt.
    Download with `watod fetch-models`.
    """

    DEFAULT_FILENAME = "yolov8l-worldv2.pt"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path or detector_checkpoint_path(
            "yolo_world", self.DEFAULT_FILENAME
        )
        self._device = device or self._default_device()
        self._model = None
        self._current_classes: Optional[tuple[str, ...]] = None

    @staticmethod
    def _default_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> bool:
        global _warned_missing
        if self._model is not None:
            return True
        try:
            from ultralytics import YOLOWorld
            if not os.path.exists(self._checkpoint_path):
                raise FileNotFoundError(
                    f"YOLO-World checkpoint not found at {self._checkpoint_path}; "
                    "run `watod fetch-models`."
                )
            self._model = YOLOWorld(self._checkpoint_path)
            # ultralytics dispatches to .cuda() / .cpu() internally on call;
            # avoiding explicit .to() here keeps the api consistent with
            # ultralytics' own examples.
            log.info("YOLO-World loaded (%s) on %s", self._checkpoint_path, self._device)
            return True
        except Exception as exc:  # noqa: BLE001
            if not _warned_missing:
                log.warning(
                    "YOLO-World unavailable (%s) — returning empty detections. "
                    "Install: pip install ultralytics + run `watod fetch-models`.",
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
        """Detect objects in image_rgb using text_prompts as the open-vocab class set.

        `text_threshold` is accepted for API parity with GroundingDINODetector
        but unused — YOLO-World fuses text and box scores into a single score
        thresholded by `box_threshold`.
        """
        del text_threshold  # unused; kept for interface parity
        if not self._load():
            return []

        # Only re-set classes when the prompt list actually changes — avoids
        # rebuilding the CLIP text embeddings on every frame.
        prompts_tuple = tuple(text_prompts)
        if prompts_tuple != self._current_classes:
            self._model.set_classes(list(text_prompts))
            self._current_classes = prompts_tuple

        # ultralytics expects BGR or PIL.Image; numpy RGB works via the
        # `source=` parameter so long as we pass via Image first.  Easiest:
        # pass the array directly — ultralytics handles channel layout.
        results = self._model.predict(
            image_rgb,
            conf=box_threshold,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy()
        cls_idx = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names  # dict[int, str]
        for box, score, idx in zip(boxes_xyxy, scores, cls_idx):
            detections.append(
                Detection(
                    bbox_xyxy=box,
                    class_name=str(names.get(int(idx), text_prompts[idx % len(text_prompts)])),
                    score=float(score),
                )
            )
        return detections
