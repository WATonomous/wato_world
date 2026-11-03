"""wato_common.artifacts.refined_labels: what label_refinement writes,
student_training reads back."""

from __future__ import annotations

import dataclasses

import pytest

from wato_common.artifact_store import refined_labels_path
from wato_common.artifacts import ArtifactFormatError, refined_labels as rl
from wato_common.io.parquet_io import write_table
from wato_common.schemas import REFINED_TRACK_SCHEMA, RefinedTrackRow

BAG = "b"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _box(sweep_id=4, **overrides) -> rl.RefinedBox:
    fields = dict(
        track_id="0000-0",
        chunk_id="0000",
        sweep_id=sweep_id,
        reference_timestamp_ns=1_000 * sweep_id,
        cx=10.0 + sweep_id,
        cy=0.0,
        cz=0.8,
        w=1.8,
        l=4.4,
        h=1.6,
        heading=0.1,
        heading_directed=True,
        cls="car",
        motion_state="moving",
        size_source="percentile",
        confidence=0.6,
        owned=True,
        residual_smoothness=0.12,
    )
    fields.update(overrides)
    return rl.RefinedBox(**fields)


def _bag():
    return [
        _box(4),
        _box(5, owned=False),
        _box(
            4,
            track_id="0000-1",
            cls=None,
            motion_state="parked",
            size_source="aggregate",
            heading_directed=False,
            w=1.9,
            residual_smoothness=None,
        ),
    ]


def test_round_trip():
    boxes = _bag()
    assert not rl.exists(BAG)
    rl.write(BAG, boxes)
    assert rl.exists(BAG)
    assert rl.read(BAG) == boxes
    rl.write(BAG, [])
    assert rl.read(BAG) == []


def test_record_tracks_the_row_schema():
    assert {f.name for f in dataclasses.fields(rl.RefinedBox)} == set(
        RefinedTrackRow.model_fields
    ) - {"bag_id"}
    assert set(REFINED_TRACK_SCHEMA.names) == set(RefinedTrackRow.model_fields)


@pytest.mark.parametrize(
    "broken, match",
    [
        (lambda bs: bs + [bs[0]], "appears twice"),
        (
            lambda bs: [dataclasses.replace(bs[0], motion_state="stopped")] + bs[1:],
            "motion_state",
        ),
        (
            lambda bs: [dataclasses.replace(bs[0], size_source="slf")] + bs[1:],
            "size_source",
        ),
        (lambda bs: [dataclasses.replace(bs[0], h=0.0)] + bs[1:], "non-positive"),
        (
            lambda bs: [bs[0], dataclasses.replace(bs[1], l=4.0)] + bs[2:],
            "size differs",
        ),
    ],
)
def test_invariants_are_enforced_on_write_and_read(broken, match):
    boxes = broken(_bag())
    with pytest.raises(ValueError, match=match):
        rl.write(BAG, boxes)
    rows = [
        RefinedTrackRow(bag_id=BAG, **dataclasses.asdict(b)).model_dump() for b in boxes
    ]
    write_table(rows, REFINED_TRACK_SCHEMA, refined_labels_path(BAG))
    with pytest.raises(ArtifactFormatError, match=match):
        rl.read(BAG)
