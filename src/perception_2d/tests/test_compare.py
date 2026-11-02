"""Tests for compare: matching two runs' masks, the review list and its
verdicts, and the per-class loss."""

from __future__ import annotations

import csv
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wato_common.artifact_store import frame_index_path, local_path
from wato_common.artifacts import masklets
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
    path = local_path(frame_index_path(BAG, CHUNK))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "frame_id": [str(s) for s in FRAMES],
                "sweep_id": list(FRAMES),
                "cam_id": [CAM] * len(FRAMES),
                "image_path": paths,
                "camera_seq": list(FRAMES),
                "camera_timestamp_ns": [s * 50_000_000 for s in FRAMES],
                "valid_camera": [True] * len(FRAMES),
            }
        ),
        path,
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
