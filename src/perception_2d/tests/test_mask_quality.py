"""mask_quality: the per-camera diagnostics report the pipeline writes."""

import os

import pytest

from wato_perception_2d import mask_quality


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def test_report_round_trips_atomically(tmp_path):
    events = [{"kind": "mask", "camera_seq": 3, "stability": 0.9}]
    out = mask_quality.write(
        "bag",
        "0000",
        "CAM",
        backend="dam4sam",
        refinement={"enabled": False},
        events=events,
    )
    assert out == mask_quality.path("bag", "0000", "CAM")
    report = mask_quality.read("bag", "0000", "CAM")
    assert report["backend"] == "dam4sam" and report["events"] == events
    assert not list(tmp_path.rglob("*.tmp"))


def test_report_is_readable_like_other_artifacts():
    """A 0600 report (NamedTemporaryFile's mode) can't be read by a container
    running as another user; the report follows the umask like every artifact."""
    old = os.umask(0o022)
    try:
        out = mask_quality.write(
            "bag", "0000", "CAM", backend="sam2", refinement={}, events=[]
        )
    finally:
        os.umask(old)
    assert os.stat(out).st_mode & 0o777 == 0o644


def test_unserializable_events_leave_no_partial_report():
    with pytest.raises(ValueError):
        mask_quality.write(
            "bag",
            "0000",
            "CAM",
            backend="sam2",
            refinement={},
            events=[{"stability": float("nan")}],
        )
    assert not os.path.exists(mask_quality.path("bag", "0000", "CAM"))
    assert not os.path.exists(mask_quality.path("bag", "0000", "CAM") + ".tmp")
