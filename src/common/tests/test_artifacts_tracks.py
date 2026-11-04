"""wato_common.artifacts.tracks: what tracking writes, label_refinement reads
back."""

from __future__ import annotations

import dataclasses

import pytest

from wato_common.artifact_store import track_summaries_path, tracks_path
from wato_common.artifacts import ArtifactFormatError, tracks as tr
from wato_common.io.parquet_io import write_table
from wato_common.schemas import (
    TRACK_SCHEMA,
    TRACK_SUMMARY_SCHEMA,
    TrackRow,
    TrackSummaryRow,
    encode_str_list,
)

BAG = "b"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _frame(sweep_id=4, **overrides) -> tr.TrackFrame:
    fields = dict(
        track_id="0000-0",
        chunk_id="0000",
        sweep_id=sweep_id,
        reference_timestamp_ns=1_000 * sweep_id,
        proposal_id=f"fused-{sweep_id:06d}-0",
        cx=10.0,
        cy=0.0,
        cz=0.8,
        w=1.8,
        l=4.2,
        h=1.6,
        heading=0.1,
        cls="car",
        score=0.6,
        owned=True,
        supporting_cam_ids=("PANO_NN",),
        supporting_masklet_ids=("m3",),
    )
    fields.update(overrides)
    return tr.TrackFrame(**fields)


def _summary(**overrides) -> tr.TrackSummary:
    fields = dict(
        track_id="0000-0",
        n_frames=2,
        first_timestamp_ns=4_000,
        last_timestamp_ns=6_000,
        motion_state="moving",
        cls="car",
        n_strong=1,
        frac_two_source=0.5,
        frac_lidar_mos=0.5,
        net_displacement_m=3.0,
        centre_var_m2=1.2,
        median_points=40.0,
    )
    fields.update(overrides)
    return tr.TrackSummary(**fields)


def _bag():
    # Sweep 6 of chunk 0000 is in the span chunk 0001 owns; 0001 saw the same
    # instant as its sweep 1. Two rows, one instant.
    frames = [
        _frame(4),
        _frame(
            6, owned=False, cls=None, supporting_cam_ids=(), supporting_masklet_ids=()
        ),
        _frame(1, chunk_id="0001", reference_timestamp_ns=6_000),
    ]
    return frames, [_summary()]


def test_round_trip():
    frames, summaries = _bag()
    assert not tr.exists(BAG)
    tr.write(BAG, frames, summaries)
    assert tr.exists(BAG)
    assert tr.read(BAG) == (frames, summaries)
    tr.write(BAG, [], [])
    assert tr.read(BAG) == ([], [])


def test_per_instant_keeps_the_owned_frame_of_a_shared_instant():
    frames, _ = _bag()
    one = tr.per_instant(frames)
    assert [(f.chunk_id, f.sweep_id) for f in one] == [("0000", 4), ("0001", 1)]
    # An instant only the non-owner saw keeps that chunk's frame.
    assert tr.per_instant(frames[:2]) == frames[:2]


def test_records_track_the_row_schemas():
    assert {f.name for f in dataclasses.fields(tr.TrackFrame)} == set(
        TrackRow.model_fields
    ) - {"bag_id"}
    assert set(TRACK_SCHEMA.names) == set(TrackRow.model_fields)
    assert {f.name for f in dataclasses.fields(tr.TrackSummary)} == set(
        TrackSummaryRow.model_fields
    ) - {"bag_id"}
    assert set(TRACK_SUMMARY_SCHEMA.names) == set(TrackSummaryRow.model_fields)


@pytest.mark.parametrize(
    "broken, match",
    [
        (lambda fs, ss: (fs, ss + ss), "two summaries"),
        (
            lambda fs, ss: (fs, [dataclasses.replace(ss[0], motion_state="stopped")]),
            "motion_state",
        ),
        (lambda fs, ss: (fs + [_frame(8, track_id="0000-9")], ss), "no summary"),
        (lambda fs, ss: (fs, ss + [_summary(track_id="0000-1")]), "no frames"),
        (lambda fs, ss: (fs + [_frame(4, proposal_id="x")], ss), "two frames at one"),
        (
            lambda fs, ss: (
                fs[:1] + [dataclasses.replace(fs[1], owned=True)] + fs[2:],
                ss,
            ),
            "two owned frames at one instant",
        ),
        (lambda fs, ss: (fs, [dataclasses.replace(ss[0], n_frames=3)]), "n_frames"),
        (
            lambda fs, ss: (fs, [dataclasses.replace(ss[0], last_timestamp_ns=9)]),
            "timestamps",
        ),
    ],
)
def test_invariants_are_enforced_on_write_and_read(broken, match):
    frames, summaries = broken(*_bag())
    with pytest.raises(ValueError, match=match):
        tr.write(BAG, frames, summaries)

    # Files that bypassed the writer are rejected on read.
    rows = []
    for f in frames:
        row = dataclasses.asdict(f)
        for name in ("supporting_cam_ids", "supporting_masklet_ids"):
            row[name] = encode_str_list(list(row[name]))
        rows.append(TrackRow(bag_id=BAG, **row).model_dump())
    write_table(rows, TRACK_SCHEMA, tracks_path(BAG))
    write_table(
        [
            TrackSummaryRow(bag_id=BAG, **dataclasses.asdict(s)).model_dump()
            for s in summaries
        ],
        TRACK_SUMMARY_SCHEMA,
        track_summaries_path(BAG),
    )
    with pytest.raises(ArtifactFormatError, match=match):
        tr.read(BAG)
