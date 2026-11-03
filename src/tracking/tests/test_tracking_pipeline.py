"""tracking end to end on a staged chunk, and its per-track decisions.

Scene (world frame), 12 frames at 20 Hz, all fused proposals:
  P  a parked car at (0, 10): lifted only, instance "iP", confident
  M  a car driving +x at 10 m/s from (-10, 0): lidar_mos + lifted "iM" pairs
  X  a one-frame class-less lidar_mos blip at (30, 30): short and weak, dropped
"""

from __future__ import annotations

import pytest

from wato_common.artifact_store import chunks_index_path
from wato_common.artifacts import proposals, tracks
from wato_common.artifacts.proposals import Proposal
from wato_common.io.parquet_io import write_table
from wato_common.schemas import CHUNK_SCHEMA
from wato_tracking import detections, summary
from wato_tracking.config import ComponentConfig, MotionConfig, StrongConfig
from wato_tracking.pipeline import ownership, run

BAG = "b"
MS = 1_000_000
N = 12


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _p(pid, sweep, cx, cy, provenance, members=(), **kw) -> Proposal:
    base = dict(
        proposal_id=pid,
        sweep_id=sweep,
        frame_id=sweep,
        sweep_ids=(sweep,),
        reference_timestamp_ns=1_000 * MS + sweep * 50 * MS,
        cx=cx,
        cy=cy,
        cz=0.8,
        w=1.8,
        l=4.4,
        h=1.6,
        heading=0.0,
        cls=kw.pop("cls", "car"),
        score=0.6,
        provenance=provenance,
        lidar_point_count=kw.pop("points", 40),
        member_proposal_ids=tuple(members),
    )
    base.update(kw)
    return Proposal(**base)


def _stage(chunk_id="0000"):
    rows = []
    for k in range(N):
        mx = -10.0 + 0.5 * k
        rows += [
            _p(f"lifted-iP-{k:06d}", k, 0.0, 10.0, "lifted", lift_confidence=0.9),
            _p(f"fused-{k:06d}-0", k, 0.0, 10.0, "fused", [f"lifted-iP-{k:06d}"]),
            _p(f"mos-{k}", k, mx, 0.0, "lidar_mos", cls=None, motion_score=2.0),
            _p(f"lifted-iM-{k:06d}", k, mx, 0.0, "lifted", lift_confidence=0.4),
            _p(
                f"fused-{k:06d}-1",
                k,
                mx,
                0.0,
                "fused",
                [f"mos-{k}", f"lifted-iM-{k:06d}"],
            ),
        ]
    rows += [
        _p("mos-99", 3, 30.0, 30.0, "lidar_mos", cls=None, points=6),
        _p("fused-000003-2", 3, 30.0, 30.0, "fused", ["mos-99"], cls=None, points=6),
    ]
    proposals.write(BAG, chunk_id, rows)
    t0 = 1_000 * MS
    write_table(
        [
            dict(
                bag_id=BAG,
                chunk_id=chunk_id,
                t_start_ns=t0,
                t_end_ns=t0 + 600 * MS,
                t_overlap_start_ns=t0,
                t_overlap_end_ns=t0 + 600 * MS,
            )
        ],
        CHUNK_SCHEMA,
        chunks_index_path(BAG),
    )


def test_parked_and_moving_cars_are_tracked_and_the_blip_dropped():
    _stage()
    run(ComponentConfig(), bag_id=BAG)
    frames, summaries = tracks.read(BAG)
    assert len(summaries) == 2
    by_state = {s.motion_state: s for s in summaries}
    parked, moving = by_state["parked"], by_state["moving"]
    assert parked.n_frames == moving.n_frames == N
    assert parked.cls == moving.cls == "car"
    assert moving.frac_two_source == 1.0 and moving.frac_lidar_mos == 1.0
    assert parked.n_strong == N and moving.n_strong == N
    assert moving.net_displacement_m == pytest.approx(0.5 * (N - 1 - 2), abs=0.6)
    assert all(f.owned for f in frames)
    assert {f.proposal_id for f in frames if f.track_id == moving.track_id} == {
        f"fused-{k:06d}-1" for k in range(N)
    }
    # Skips without --force; re-tracking one chunk keeps the others.
    run(ComponentConfig(), bag_id=BAG)
    run(ComponentConfig(), bag_id=BAG, chunk_id="0000", force=True)
    assert len(tracks.read(BAG)[1]) == 2


def test_missing_proposals_name_the_step_to_run():
    write_table(
        [
            dict(
                bag_id=BAG,
                chunk_id="0000",
                t_start_ns=0,
                t_end_ns=1,
                t_overlap_start_ns=0,
                t_overlap_end_ns=1,
            )
        ],
        CHUNK_SCHEMA,
        chunks_index_path(BAG),
    )
    with pytest.raises(FileNotFoundError, match="run proposal_generation"):
        run(ComponentConfig(), bag_id=BAG)


def test_ownership_splits_each_shared_span_at_its_midpoint():
    chunks = [
        dict(chunk_id="0001", t_start_ns=28, t_end_ns=58),
        dict(chunk_id="0000", t_start_ns=0, t_end_ns=30),
    ]
    w = ownership(chunks)
    assert w["0000"] == (float("-inf"), 29.0) and w["0001"] == (29.0, float("inf"))


def test_lifted_instance_is_parsed_from_the_row_id():
    assert detections.instance_of("lifted-b9eb2372-6bd-000001") == "b9eb2372-6bd"


def test_strong_needs_two_sources_or_a_confident_dense_lift():
    def det(members, **lift):
        by_id = {
            "lifted-i-000001": _p("lifted-i-000001", 1, 0, 0, "lifted", **lift),
            "mos-1": _p("mos-1", 1, 0, 0, "lidar_mos", cls=None),
        }
        fused = _p("fused-000001-0", 1, 0, 0, "fused", members)
        return detections.detection(fused, by_id, StrongConfig())

    assert det(["mos-1", "lifted-i-000001"], lift_confidence=0.1).strong
    assert det(["lifted-i-000001"], lift_confidence=0.9, points=40).strong
    assert not det(["lifted-i-000001"], lift_confidence=0.9, points=5).strong
    assert not det(["mos-1"]).strong
    assert det(["lifted-i-000001"], lift_confidence=0.9).key == "i"


def test_motion_state_rules():
    cfg = MotionConfig()
    assert summary.motion_state(3, 5.0, 1.0, cfg) == "unknown"
    # A walker's 0.4 s fragment: 0.6 m at 1.5 m/s is moving.
    assert summary.motion_state(8, 0.6, 0.4, cfg) == "moving"
    # A parked car whose one-sided box drifted 0.8 m over 5 s is parked.
    assert summary.motion_state(100, 0.8, 5.0, cfg) == "parked"
    # A car creeping 1.3 m in 19 s at a light moved: 1 m net is enough.
    assert summary.motion_state(300, 1.3, 19.0, cfg) == "moving"
    # Jitter on a short track: fast but not far.
    assert summary.motion_state(8, 0.3, 0.35, cfg) == "parked"
