"""wato_common.artifacts.detections_3d: what detection_3d writes,
proposal_generation reads back."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from wato_common.artifact_store import detections_3d_frames_path, detections_3d_path
from wato_common.artifacts import ArtifactFormatError
from wato_common.artifacts import detections_3d as d3
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import DETECTION_3D_SCHEMA

BAG, CHUNK = "b", "0000"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _pose() -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (10.0, -2.5, 1.84)
    m[:2, :2] = [[0.0, 1.0], [-1.0, 0.0]]
    return m


def _frames() -> list[d3.DetectorFrame]:
    return [
        d3.DetectorFrame(
            sweep_id=12,
            frame_id=3,
            reference_timestamp_ns=1_000,
            status=d3.OK,
            input_sweep_ids=(8, 10, 12),
            n_points=90_000,
            camera_timestamps_ns=(990, 1_005, 1_010, 995, 1_001, 999),
            world_T_model=_pose(),
        ),
        d3.DetectorFrame(
            sweep_id=15,
            frame_id=None,
            reference_timestamp_ns=1_100,
            status=d3.MISSING_CAMERA,
            detail="PANO_SS",
        ),
    ]


def _det(**overrides) -> d3.Detection3D:
    fields = dict(
        detection_id="det-000012-0",
        sweep_id=12,
        frame_id=3,
        reference_timestamp_ns=1_000,
        cx=12.0,
        cy=3.0,
        cz=0.8,
        w=1.9,
        l=4.5,
        h=1.6,
        heading=2.9,
        vx=0.4,
        vy=-0.1,
        class_id=0,
        cls="car",
        score=0.71,
        model_x=1.0,
        model_y=2.0,
        model_z=-1.0,
        model_w=1.9,
        model_l=4.5,
        model_h=1.6,
        model_rot=0.3,
        model_vx=0.1,
        model_vy=0.4,
    )
    fields.update(overrides)
    return d3.Detection3D(**fields)


def _dets() -> list[d3.Detection3D]:
    return [
        _det(),
        _det(detection_id="det-000012-1", class_id=8, cls="pedestrian", score=0.02),
    ]


def test_round_trip():
    d3.write(BAG, CHUNK, _frames(), _dets())
    assert d3.exists(BAG, CHUNK)
    frames, dets = d3.read(BAG, CHUNK)
    assert dets == _dets()
    assert [dataclasses.replace(f, world_T_model=None) for f in frames] == [
        dataclasses.replace(f, world_T_model=None) for f in _frames()
    ]
    np.testing.assert_array_equal(frames[0].world_T_model, _pose())
    assert frames[1].world_T_model is None
    assert frames[0].ok and not frames[1].ok


def test_frames_written_first_so_detections_mark_completion(tmp_path):
    with pytest.raises(ValueError):
        d3.write(BAG, CHUNK, _frames(), [_det(sweep_id=15)])
    assert not d3.exists(BAG, CHUNK)


@pytest.mark.parametrize(
    "dets, frames, message",
    [
        (lambda: _dets() + [_det()], _frames, "appears twice"),
        (
            lambda: [_det(sweep_id=15, frame_id=None, reference_timestamp_ns=1_100)],
            _frames,
            "not an ok frame",
        ),
        (lambda: [_det(frame_id=4)], _frames, "disagree"),
        (lambda: [_det(w=0.0)], _frames, "non-positive"),
        (lambda: [_det(class_id=1)], _frames, "not in CLASSES"),
        (lambda: [], lambda: _frames() + [_frames()[1]], "appears twice"),
        (
            lambda: [],
            lambda: [dataclasses.replace(_frames()[0], world_T_model=None)],
            "4x4",
        ),
        (
            lambda: [],
            lambda: [dataclasses.replace(_frames()[0], input_sweep_ids=(8,))],
            "reference sweep",
        ),
        (
            lambda: [],
            lambda: [dataclasses.replace(_frames()[1], status="late")],
            "status",
        ),
    ],
)
def test_write_refuses_broken_invariants(dets, frames, message):
    with pytest.raises(ValueError, match=message):
        d3.write(BAG, CHUNK, frames(), dets())


def test_read_refuses_detection_of_a_skipped_frame():
    d3.write(BAG, CHUNK, _frames(), [])
    row = dataclasses.asdict(
        _det(sweep_id=15, frame_id=None, reference_timestamp_ns=1_100)
    )
    write_table(
        [dict(row, bag_id=BAG, chunk_id=CHUNK)],
        DETECTION_3D_SCHEMA,
        detections_3d_path(BAG, CHUNK),
    )
    with pytest.raises(ArtifactFormatError, match="not an ok frame"):
        d3.read(BAG, CHUNK)


def test_read_refuses_missing_column():
    d3.write(BAG, CHUNK, _frames(), _dets())
    rows = read_rows(detections_3d_path(BAG, CHUNK))
    for r in rows:
        del r["score"]
    import pyarrow as pa

    write_table(
        rows,
        pa.schema([f for f in DETECTION_3D_SCHEMA if f.name != "score"]),
        detections_3d_path(BAG, CHUNK),
    )
    with pytest.raises(ArtifactFormatError, match="score"):
        d3.read(BAG, CHUNK)


def test_frames_path_is_beside_detections():
    assert (
        detections_3d_frames_path(BAG, CHUNK).rsplit("/", 1)[0]
        == (detections_3d_path(BAG, CHUNK).rsplit("/", 1)[0])
    )
