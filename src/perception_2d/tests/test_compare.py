"""Tests for compare: matching two runs' masks, the review list and its
verdicts, and the per-class loss."""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from wato_common.artifact_store import poses_path
from wato_common.artifacts import frame_index, masklets
from wato_common.io.parquet_io import write_table
from wato_common.schemas import POSES_SCHEMA, FrameIndexRow
from wato_perception_2d import compare
from wato_perception_2d.config import ComponentConfig

BAG, CHUNK, CAM = "bag", "0000", "CAM"
W, H = 64, 48
FRAMES = tuple(range(10))


def _box(y0, y1, x0, x1) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


CAR = _box(10, 20, 5, 25)
PED = _box(28, 44, 40, 46)


def _write_run(root, monkeypatch, masks: dict[str, tuple[str, np.ndarray]]):
    """One chunk with one camera; ``masks`` = masklet_id → (class, mask on
    every frame)."""
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{root}")
    records = []
    for mid, (cls, mask) in masks.items():
        for s in FRAMES:
            masklets.write_mask(BAG, CHUNK, mid, s, mask)
        records.append(masklets.MaskletRecord(mid, CAM, cls, 0.9, FRAMES))
    masklets.write(BAG, CHUNK, records)


def _write_images(root, monkeypatch):
    from PIL import Image

    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{root}")
    paths = []
    for s in FRAMES:
        p = os.path.join(str(root), f"img_{s}.png")
        Image.fromarray(np.full((H, W, 3), 90, np.uint8)).save(p)
        paths.append(p)
    write_table(
        [
            {
                "bag_id": BAG,
                "chunk_id": CHUNK,
                "timestamp_ns": 0,
                **dict.fromkeys(("x", "y", "z", "qx", "qy", "qz"), 0.0),
                "qw": 1.0,
                "world_T_ego_flat": np.eye(4).ravel().tolist(),
                "source": "/odom",
                "valid": True,
                "interval_drop_reason": None,
            }
        ],
        POSES_SCHEMA,
        poses_path(BAG, CHUNK),
    )
    frame_index.write(
        BAG,
        CHUNK,
        [
            FrameIndexRow(
                frame_id=str(s),
                bag_id=BAG,
                chunk_id=CHUNK,
                sweep_id=s,
                lidar_id="lidar_top",
                lidar_path="",
                reference_timestamp_ns=s * 50_000_000,
                cam_id=CAM,
                image_path=p,
                camera_seq=s,
                camera_timestamp_ns=s * 50_000_000,
                valid_camera=True,
            )
            for s, p in zip(FRAMES, paths)
        ],
    )


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """A baseline with a car and a pedestrian; the current run kept the car."""
    base, cur = tmp_path / "baseline", tmp_path / "current"
    _write_run(base, monkeypatch, {"car_a": ("car", CAR), "ped_a": ("pedestrian", PED)})
    _write_run(cur, monkeypatch, {"car_b": ("car", CAR)})
    _write_images(cur, monkeypatch)
    return f"file://{base}", tmp_path / "out"


def _review(out) -> list[dict]:
    with open(out / "review.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_lost_masklet_is_listed_with_a_sheet(runs):
    baseline, out = runs
    summary = compare.run(
        None, bag_id=BAG, baseline_uri=baseline, out_dir=str(out), chunk_ids=[CHUNK]
    )
    rows = _review(out)
    assert [(r["masklet_id"], r["lost_masks"], r["verdict"]) for r in rows] == [
        ("ped_a", "10", "")
    ]
    assert (out / "sheets" / "000.png").exists()
    assert (summary["car"]["masks"], summary["car"]["lost"]) == (10, 0)
    assert summary["pedestrian"]["unreviewed"] == 10
    # The run's root is restored after reading the baseline's.
    assert os.environ["ARTIFACT_ROOT_URI"].endswith("current")


def test_verdicts_survive_a_rerun_and_count_in_the_summary(runs):
    baseline, out = runs
    compare.run(
        None, bag_id=BAG, baseline_uri=baseline, out_dir=str(out), chunk_ids=[CHUNK]
    )
    rows = _review(out)
    rows[0]["verdict"] = "junk"
    with open(out / "review.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=compare.REVIEW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = compare.run(
        None, bag_id=BAG, baseline_uri=baseline, out_dir=str(out), chunk_ids=[CHUNK]
    )
    assert _review(out)[0]["verdict"] == "junk"
    assert summary["pedestrian"]["junk"] == 10
    assert summary["pedestrian"]["unreviewed"] == 0


def test_masklet_inside_the_ego_mask_is_premarked_junk(runs):
    baseline, out = runs
    cfg = ComponentConfig(
        ego_mask={
            "image_size": [W, H],
            "max_fraction": 0.5,
            "polygons": {CAM: [[[38, 26], [48, 26], [48, 47], [38, 47]]]},
        }
    )
    summary = compare.run(
        cfg, bag_id=BAG, baseline_uri=baseline, out_dir=str(out), chunk_ids=[CHUNK]
    )
    assert (_review(out)[0]["verdict"], _review(out)[0]["note"]) == (
        "junk",
        "inside the ego mask",
    )
    assert summary["pedestrian"]["junk"] == 10
