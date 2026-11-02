"""wato_common.artifacts.proposals: what proposal_generation writes, tracking
reads back."""

from __future__ import annotations

import dataclasses

import pytest

from wato_common.artifact_store import proposals_path
from wato_common.artifacts import ArtifactFormatError, proposals as pr
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import (
    PROPOSAL_SCHEMA,
    ProposalRow,
    encode_int_list,
    encode_str_list,
)

BAG, CHUNK = "b", "0000"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _proposal(**overrides) -> pr.Proposal:
    fields = dict(
        proposal_id="mos-4",
        sweep_id=12,
        frame_id=3,
        sweep_ids=(12, 13, 14),
        reference_timestamp_ns=1_000,
        cx=1.0,
        cy=2.0,
        cz=0.5,
        w=1.8,
        l=4.2,
        h=1.5,
        heading=0.1,
        cls="car",
        score=0.7,
        provenance="lidar_mos",
        lidar_point_count=120,
        supporting_cam_ids=("PANO_NN",),
        supporting_masklet_ids=("m3",),
        motion_score=1.9,
        n_sources=2,
        frac_persistent=0.05,
        track_life=9,
    )
    fields.update(overrides)
    return pr.Proposal(**fields)


def _chunk() -> list[pr.Proposal]:
    return [
        _proposal(),
        # Class-less, from a sweep frame_sync matched to no canonical sweep.
        _proposal(
            proposal_id="mos-5",
            sweep_id=15,
            frame_id=None,
            sweep_ids=(15,),
            cls=None,
            lidar_point_count=None,
            supporting_cam_ids=(),
            supporting_masklet_ids=(),
        ),
        _proposal(
            proposal_id="lifted-0",
            provenance="lifted",
            motion_score=None,
            n_sources=None,
            frac_persistent=None,
            track_life=None,
            lift_confidence=0.8,
        ),
        _proposal(
            proposal_id="fused-0",
            provenance="fused",
            member_proposal_ids=("mos-4", "lifted-0"),
        ),
    ]


def test_round_trip():
    proposals = _chunk()
    assert not pr.exists(BAG, CHUNK)
    pr.write(BAG, CHUNK, proposals)
    assert pr.exists(BAG, CHUNK)
    assert pr.read(BAG, CHUNK) == proposals
    assert [p.proposal_id for p in pr.fused(proposals)] == ["fused-0"]
    pr.write(BAG, CHUNK, [])
    assert pr.read(BAG, CHUNK) == []


def test_record_tracks_the_row_schema():
    """A column added to ProposalRow must be added to Proposal and to
    PROPOSAL_SCHEMA too."""
    record = {f.name for f in dataclasses.fields(pr.Proposal)}
    assert record == set(ProposalRow.model_fields) - {"bag_id", "chunk_id"}
    assert set(PROPOSAL_SCHEMA.names) == set(ProposalRow.model_fields)


@pytest.mark.parametrize(
    "broken, match",
    [
        (lambda ps: ps + [ps[0]], "appears twice"),
        (
            lambda ps: [dataclasses.replace(ps[0], provenance="detector")] + ps[1:],
            "provenance",
        ),
        (
            lambda ps: [dataclasses.replace(ps[0], sweep_ids=(13, 14))] + ps[1:],
            "not in sweep_ids",
        ),
        (
            lambda ps: ps[:3] + [dataclasses.replace(ps[3], member_proposal_ids=())],
            "lists no member",
        ),
        (
            lambda ps: ps[:3]
            + [dataclasses.replace(ps[3], member_proposal_ids=("mos-9",))],
            "not in the file",
        ),
        (
            lambda ps: ps[:3]
            + [dataclasses.replace(ps[3], member_proposal_ids=("mos-5",))],
            "from sweep 15",
        ),
        (
            lambda ps: ps
            + [
                _proposal(
                    proposal_id="fused-1",
                    provenance="fused",
                    member_proposal_ids=("fused-0",),
                )
            ],
            "itself fused",
        ),
        (
            lambda ps: [dataclasses.replace(ps[0], member_proposal_ids=("lifted-0",))]
            + ps[1:],
            "only fused rows",
        ),
    ],
)
def test_invariants_are_enforced_on_write_and_read(broken, match):
    proposals = broken(_chunk())
    with pytest.raises(ValueError, match=match):
        pr.write(BAG, CHUNK, proposals)

    # A file that bypassed the writer is rejected on read.
    pr.write(BAG, CHUNK, _chunk())
    uri = proposals_path(BAG, CHUNK)
    rows = read_rows(uri)
    by_id = {r["proposal_id"]: r for r in rows}
    bad_rows = []
    for p in proposals:
        row = dict(by_id.get(p.proposal_id, rows[0]))
        row.update(
            proposal_id=p.proposal_id,
            provenance=p.provenance,
            sweep_id=p.sweep_id,
            sweep_ids=encode_int_list(list(p.sweep_ids)),
            member_proposal_ids=encode_str_list(list(p.member_proposal_ids)),
        )
        bad_rows.append(row)
    write_table(bad_rows, PROPOSAL_SCHEMA, uri)
    with pytest.raises(ArtifactFormatError, match=match):
        pr.read(BAG, CHUNK)
