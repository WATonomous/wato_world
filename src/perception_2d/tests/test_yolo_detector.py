"""Tests for the YOLO detector backend — colour order, class mapping and
fail-loud loading — exercised with a fake Ultralytics model (no weights, no
ultralytics install needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from wato_perception_2d.models import yolo_detector
from wato_perception_2d.models.yolo_detector import YoloDetector, check_class_map

COCO = {0: "person", 1: "bicycle", 2: "car", 7: "truck", 10: "fire hydrant"}
CLASS_MAP = {
    "person": "pedestrian",
    "car": "car",
    "truck": "truck",
    "bicycle": "bicycle",
}
CONCEPTS = [("pedestrian", "pedestrian"), ("car", "car"), ("bicycle", "bicycle")]


class _Boxes:
    def __init__(self, rows):
        self.xyxy = np.array([r[0] for r in rows], dtype=np.float32).reshape(-1, 4)
        self.cls = np.array([r[1] for r in rows], dtype=np.float32)
        self.conf = np.array([r[2] for r in rows], dtype=np.float32)


class _FakeYolo:
    """Stands in for ultralytics.YOLO: records predict() inputs."""

    def __init__(self, rows):
        self.names = dict(COCO)
        self._rows = rows
        self.calls: list[tuple[np.ndarray, dict]] = []

    def predict(self, source, **kw):
        self.calls.append((source, kw))
        keep = kw.get("classes")
        rows = [r for r in self._rows if keep is None or r[1] in keep]
        return [type("R", (), {"boxes": _Boxes(rows)})()]


def _detector(rows, class_map=CLASS_MAP):
    det = YoloDetector(
        "/unused.pt", class_map=class_map, imgsz=1280, conf=0.3, device="cpu"
    )
    fake = _FakeYolo(rows)
    det._model = fake
    det._ids = {i: class_map[n] for i, n in COCO.items() if n in class_map}
    return det, fake


def test_frames_are_passed_to_ultralytics_as_bgr():
    """Ultralytics reads numpy images as BGR; pipeline frames are RGB. Passing
    RGB through measurably changes detections on real frames."""
    det, fake = _detector([])
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # red channel
    det.detect(rgb, CONCEPTS)
    sent, kw = fake.calls[0]
    np.testing.assert_array_equal(sent, rgb[..., ::-1])
    assert sent[0, 0, 2] == 200 and sent[0, 0, 0] == 0  # red is last in BGR
    assert kw["imgsz"] == 1280 and kw["conf"] == 0.3 and kw["verbose"] is False


def test_only_mapped_taxonomy_classes_are_requested_and_kept():
    det, fake = _detector(
        [
            ((0, 0, 10, 20), 0, 0.9),  # person → pedestrian
            ((5, 5, 50, 30), 2, 0.8),  # car
            ((60, 0, 90, 40), 7, 0.7),  # truck: mapped, but not in this taxonomy
            ((1, 1, 3, 9), 10, 0.6),  # fire hydrant: unmapped
        ]
    )
    out = det.detect(np.zeros((100, 100, 3), dtype=np.uint8), CONCEPTS)
    assert sorted(fake.calls[0][1]["classes"]) == [0, 1, 2]  # person, bicycle, car
    assert [(d.cls, d.phrase, d.box_xyxy) for d in out] == [
        ("pedestrian", "person", (0.0, 0.0, 10.0, 20.0)),
        ("car", "car", (5.0, 5.0, 50.0, 30.0)),
    ]


def test_no_taxonomy_class_reachable_means_no_forward():
    det, fake = _detector([((0, 0, 1, 1), 2, 0.9)])
    assert det.detect(np.zeros((8, 8, 3), np.uint8), [("barrier", "barrier")]) == []
    assert fake.calls == []


def test_check_class_map_separates_typos_from_a_smaller_taxonomy():
    """A typo ("pedestrain") or spelling variant ("traffic cone") would silently
    drop every such detection — fatal. A target the taxonomy simply doesn't
    have ("truck" here) is fine; it's just never used."""
    concepts = CONCEPTS + [("traffic cone", "traffic_cone")]
    misnamed, unused, uncovered = check_class_map(
        {**CLASS_MAP, "person": "pedestrain", "fire hydrant": "traffic cone"}, concepts
    )
    assert misnamed == {"pedestrain", "traffic cone"}
    assert unused == {"truck"}
    assert uncovered == {"pedestrian", "traffic_cone"}  # the typo covers nothing


def test_missing_checkpoint_fails_loud(tmp_path):
    det = YoloDetector(str(tmp_path / "nope.pt"), class_map=CLASS_MAP)
    with pytest.raises(RuntimeError, match="fetch_models.py"):
        det._load()


def test_class_map_naming_a_class_the_weights_lack_fails_loud(tmp_path, monkeypatch):
    import sys
    import types

    ckpt = tmp_path / "w.pt"
    ckpt.write_bytes(b"")
    fake_mod = types.ModuleType("ultralytics")
    fake_mod.YOLO = lambda path, task=None: _FakeYolo([])
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)
    det = YoloDetector(str(ckpt), class_map={**CLASS_MAP, "motorbike": "motorcycle"})
    with pytest.raises(ValueError, match="motorbike"):
        det._load()
    assert yolo_detector.os.environ.get("YOLO_OFFLINE") == "1"


# ---------------------------------------------------------------------------
# Backend selection (pipeline.build_detector)
# ---------------------------------------------------------------------------


def _cfg(**detection):
    from wato_perception_2d.config import ComponentConfig

    return ComponentConfig.model_validate({"detection": detection})


def test_backend_switch_builds_the_configured_detector():
    from wato_perception_2d.models.detector import GroundingDinoDetector
    from wato_perception_2d.pipeline import build_detector

    taxonomy = [("car", "car"), ("pedestrian", "pedestrian")]
    assert isinstance(build_detector(_cfg(backend="yolo"), taxonomy), YoloDetector)
    assert isinstance(
        build_detector(_cfg(backend="grounding_dino"), taxonomy), GroundingDinoDetector
    )


def test_yolo_cannot_be_prompted_by_open_vocabulary_discovery():
    from wato_perception_2d.config import ComponentConfig
    from wato_perception_2d.pipeline import build_detector

    cfg = ComponentConfig.model_validate(
        {"detection": {"backend": "yolo"}, "discovery": {"backend": "florence2"}}
    )
    with pytest.raises(ValueError, match="fixed"):
        build_detector(cfg, [])


def test_misspelled_class_map_target_stops_the_pipeline():
    from wato_perception_2d.pipeline import build_detector

    cfg = _cfg(backend="yolo", yolo={"class_map": {"person": "pedestrain"}})
    with pytest.raises(ValueError, match="pedestrain"):
        build_detector(cfg, [("pedestrian", "pedestrian")])
