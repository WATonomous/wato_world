"""Closed-set 2D detector (Ultralytics YOLO26), the default box source for SAM2.

Selected with ``detection.backend: yolo``. Same contract as
``GroundingDinoDetector.detect``: class-labeled boxes for one frame, which
prompt SAM2 (sam2_tracker.py). The weights are COCO-trained, so the class
vocabulary is fixed: ``detection.yolo.class_map`` maps COCO names to taxonomy
classes, and a detection is kept only when its class maps to a class in the
taxonomy (the concepts). Taxonomy classes COCO has no counterpart for
(traffic_cone, barrier) are never detected by this backend — the pipeline logs
which at startup. The prompt text of a concept is ignored; only its canonical
class matters.

Inference traps handled here, each of which gives plausible-looking wrong output
rather than an error:

* Colour order. Ultralytics treats a numpy image as BGR (OpenCV convention);
  pipeline frames are RGB. Passing RGB straight through measurably changes the
  detections (on a WATO frame it lost a fire hydrant and a stop sign and shifted
  every score), so detect() converts explicitly.
* Weights by path, never by name. ``YOLO("yolo26x.pt")`` with a bare name
  downloads whatever upstream currently ships; the pinned file
  (model_registry.RAW_CHECKPOINTS["yolo26"], fetched by scripts/fetch_models.py)
  is loaded from its path and a missing file is an error.
* Offline. ``YOLO_OFFLINE=1`` disables Ultralytics' online checks and
  telemetry, and ``YOLO_CONFIG_DIR`` points its settings file at a writable
  temp dir; both must be set before ultralytics is first imported.

Lazy-imports ultralytics; detect() without it raises.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

import numpy as np

from wato_perception_2d.models.detector import Concept, Detection

log = logging.getLogger(__name__)


def check_class_map(
    class_map: dict[str, str], concepts: list[Concept]
) -> tuple[set[str], set[str], set[str]]:
    """Compare a weights→taxonomy class map with the taxonomy.

    Returns ``(misnamed, unused, uncovered)``:

    * misnamed — targets that aren't taxonomy classes but nearly match one
      ("pedestrain", "traffic cone" vs "traffic_cone"): every such detection
      would be silently dropped, so the pipeline refuses to start;
    * unused — other targets outside the taxonomy (a smaller taxonomy just
      doesn't use them);
    * uncovered — taxonomy classes nothing maps to: never detected by this
      backend.

    Pure function.
    """
    import difflib

    taxonomy = {cls for _, cls in concepts}

    def norm(c: str) -> str:
        return c.lower().replace("_", " ").strip()

    by_norm = {norm(c): c for c in taxonomy}
    outside = set(class_map.values()) - taxonomy
    misnamed = {
        t
        for t in outside
        if norm(t) in by_norm or difflib.get_close_matches(norm(t), by_norm, cutoff=0.8)
    }
    return misnamed, outside - misnamed, taxonomy - set(class_map.values())


class YoloDetector:
    """Ultralytics YOLO detector over a fixed (COCO) vocabulary.

    Raises RuntimeError if ultralytics / the checkpoint is unavailable, and
    ValueError if ``class_map`` names a class the weights don't have.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        class_map: dict[str, str],
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 1280,
        device: Optional[str] = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._class_map = dict(class_map)
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._device = device or self._default_device()
        self._model = None
        self._ids: dict[int, str] = {}  # weights' class id → taxonomy class

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
        if not os.path.isfile(self._checkpoint):
            raise RuntimeError(
                f"YOLO checkpoint {self._checkpoint!r} not found. Fetch the pinned "
                "weights with scripts/fetch_models.py (model_registry: yolo26). "
                "perception_2d does not download weights at runtime."
            )
        # Read by ultralytics at import time — set before the first import.
        os.environ.setdefault("YOLO_OFFLINE", "1")
        os.environ.setdefault(
            "YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "ultralytics")
        )
        try:
            from ultralytics import YOLO

            model = YOLO(self._checkpoint, task="detect")
        except Exception as exc:
            raise RuntimeError(
                f"YOLO checkpoint {self._checkpoint!r} could not be loaded: {exc}. "
                "Install 'ultralytics' (docker/requirements/perception_2d.txt)."
            ) from exc
        by_name = {name: int(i) for i, name in model.names.items()}
        unknown = set(self._class_map) - set(by_name)
        if unknown:
            raise ValueError(
                f"detection.yolo.class_map names {sorted(unknown)}, which "
                f"{os.path.basename(self._checkpoint)} doesn't have; its classes "
                f"are {sorted(by_name)}"
            )
        self._ids = {by_name[name]: cls for name, cls in self._class_map.items()}
        self._model = model
        log.info("YOLO loaded (%s) on %s", self._checkpoint, self._device)

    def detect(self, image_rgb: np.ndarray, concepts: list[Concept]) -> list[Detection]:
        """Detect taxonomy objects in one RGB image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            concepts: (prompt_text, canonical_class) per taxonomy class; only
                the canonical classes are used, to select which mapped classes
                to keep.

        Returns:
            Detections whose class maps to a taxonomy class, ``phrase`` holding
            the weights' own class name.

        Raises:
            RuntimeError: if ultralytics / the checkpoint is unavailable.
        """
        wanted = {cls for _, cls in concepts}
        self._load()
        ids = sorted(i for i, cls in self._ids.items() if cls in wanted)
        if not ids:
            return []
        result = self._model.predict(
            np.ascontiguousarray(image_rgb[..., ::-1]),  # RGB → BGR (see module doc)
            imgsz=self._imgsz,
            conf=self._conf,
            iou=self._iou,
            classes=ids,
            device=self._device,
            verbose=False,
        )[0]
        boxes = result.boxes
        names = self._model.names
        dets: list[Detection] = []
        for xyxy, cls_id, score in zip(
            _numpy(boxes.xyxy), _numpy(boxes.cls), _numpy(boxes.conf)
        ):
            cls = self._ids.get(int(cls_id))
            if cls is None or cls not in wanted:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            dets.append(
                Detection(
                    box_xyxy=(x1, y1, x2, y2),
                    cls=cls,
                    score=float(score),
                    phrase=str(names[int(cls_id)]),
                )
            )
        return dets


def _numpy(x) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def yolo_importable() -> bool:
    """Cheap check that ultralytics imports (mirrors detector_importable)."""
    import importlib

    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "ultralytics")
    )
    try:
        importlib.import_module("ultralytics")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Detector backend (`ultralytics`) not importable: %s", exc)
        return False
