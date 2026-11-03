"""wato_common.artifacts.gt_boxes: what the GT importers write, evaluation
reads back."""

from __future__ import annotations

import dataclasses

import pytest

from wato_common.artifact_store import gt_boxes_path
from wato_common.artifacts import ArtifactFormatError, gt_boxes as gt
from wato_common.io.parquet_io import write_table
from wato_common.schemas import GT_BOX_SCHEMA, GtBoxRow

BAG = "b"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _box(**overrides) -> gt.GtBox:
    fields = dict(
        source="nuscenes_markers",
        timestamp_ns=1_000,
        instance_id="28114",
        cx=373.3,
        cy=1130.4,
        cz=0.8,
        w=0.62,
        l=0.67,
        h=1.64,
        heading=-0.37,
        cls="pedestrian",
        source_class="human.pedestrian.adult",
        motion_state="moving",
    )
    fields.update(overrides)
    return gt.GtBox(**fields)


def _bag() -> list[gt.GtBox]:
    return [
        _box(),
        _box(timestamp_ns=2_000),
        # Outside the taxonomy, motion unknown: kept so it can be ignored.
        _box(
            instance_id="9",
            cls=None,
            source_class="movable_object.trafficcone",
            motion_state=None,
        ),
        _box(
            instance_id="10",
            cls="car",
            source_class="vehicle.car",
            motion_state="parked",
        ),
    ]


def test_round_trip():
    boxes = _bag()
    assert not gt.exists(BAG)
    gt.write(BAG, boxes)
    assert gt.exists(BAG)
    assert gt.read(BAG) == boxes
    gt.write(BAG, [])
    assert gt.read(BAG) == []


def test_record_tracks_the_row_schema():
    """A column added to GtBoxRow must be added to GtBox and GT_BOX_SCHEMA too."""
    record = {f.name for f in dataclasses.fields(gt.GtBox)}
    assert record == set(GtBoxRow.model_fields) - {"bag_id"}
    assert set(GT_BOX_SCHEMA.names) == set(GtBoxRow.model_fields)


@pytest.mark.parametrize(
    "broken, match",
    [
        (lambda bs: bs + [bs[0]], "appears twice"),
        (lambda bs: [dataclasses.replace(bs[0], source="cvat")] + bs[1:], "source"),
        (
            lambda bs: [dataclasses.replace(bs[0], motion_state="stopped")] + bs[1:],
            "motion_state",
        ),
        (lambda bs: [dataclasses.replace(bs[0], h=0.0)] + bs[1:], "non-positive"),
    ],
)
def test_invariants_are_enforced_on_write_and_read(broken, match):
    boxes = broken(_bag())
    with pytest.raises(ValueError, match=match):
        gt.write(BAG, boxes)

    # A file that bypassed the writer is rejected on read.
    rows = [GtBoxRow(bag_id=BAG, **dataclasses.asdict(b)).model_dump() for b in boxes]
    write_table(rows, GT_BOX_SCHEMA, gt_boxes_path(BAG))
    with pytest.raises(ArtifactFormatError, match=match):
        gt.read(BAG)


def test_same_instance_and_time_from_two_sources_is_allowed():
    boxes = [_box(), _box(source="sustechpoints")]
    gt.write(BAG, boxes)
    assert gt.read(BAG) == boxes
