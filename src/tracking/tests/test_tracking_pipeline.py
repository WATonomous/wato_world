"""tracking end to end on staged chunks, and its per-track decisions.

Scene (world frame), 12 frames at 20 Hz, all fused proposals:
  P  a parked car at (0, 10): lifted only, instance "iP", confident
  M  a car driving +x at 10 m/s from (-10, 0): lidar_mos + lifted "iM" pairs
  X  a one-frame class-less lidar_mos blip at (30, 30): short and weak, dropped
Staged as one chunk, or as two sharing frames 4-7 (each chunk's own sweep
and instance ids, as upstream writes them).
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
from wato_tracking.pipeline import run

BAG = "b"
MS = 1_000_000
N = 12


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _t(k):
    return 1_000 * MS + k * 50 * MS


def _p(pid, sweep, cx, cy, provenance, members=(), k=None, **kw) -> Proposal:
    base = dict(
        proposal_id=pid,
        sweep_id=sweep,
        frame_id=sweep,
        sweep_ids=(sweep,),
        reference_timestamp_ns=_t(sweep if k is None else k),
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


def _chunk_rows(chunk_id, ks, instance):
    """Proposals of frames ks, the chunk's sweeps numbered from 0."""
    rows = []
    for s, k in enumerate(ks):
        mx = -10.0 + 0.5 * k
        lp, lm = f"lifted-{instance}P-{s:06d}", f"lifted-{instance}M-{s:06d}"
        rows += [
            _p(lp, s, 0.0, 10.0, "lifted", k=k, lift_confidence=0.9),
            _p(f"fused-{s:06d}-0", s, 0.0, 10.0, "fused", [lp], k=k),
            _p(f"mos-{s}", s, mx, 0.0, "lidar_mos", k=k, cls=None, motion_score=2.0),
            _p(lm, s, mx, 0.0, "lifted", k=k, lift_confidence=0.4),
            _p(f"fused-{s:06d}-1", s, mx, 0.0, "fused", [f"mos-{s}", lm], k=k),
        ]
        if k == 3:
            rows += [
                _p("mos-99", s, 30.0, 30.0, "lidar_mos", k=k, cls=None, points=6),
                _p(
                    f"fused-{s:06d}-2",
                    s,
                    30.0,
                    30.0,
                    "fused",
                    ["mos-99"],
                    k=k,
                    cls=None,
                    points=6,
                ),
            ]
    proposals.write(BAG, chunk_id, rows)


def _index(*windows):
    """Chunks "0000", "0001", ... from (t_start, t_end, t_overlap_start,
    t_overlap_end) in ms."""
    write_table(
        [
            dict(
                bag_id=BAG,
                chunk_id=f"{i:04d}",
                t_start_ns=a * MS,
                t_end_ns=b * MS,
                t_overlap_start_ns=c * MS,
                t_overlap_end_ns=d * MS,
            )
            for i, (a, b, c, d) in enumerate(windows)
        ],
        CHUNK_SCHEMA,
        chunks_index_path(BAG),
    )


def _stage():
    _chunk_rows("0000", range(N), "i")
    _index((1_000, 1_600, 1_000, 1_600))


def _stage_two_chunks():
    # Frames 0-7 in chunk 0000, 4-7 again (as sweeps 0-3) and 8-11 in 0001:
    # shared span 1200-1350 ms, split at (1300 + 1250) / 2 = 1275 ms, so 0000
    # owns frames 0-5 and 0001 frames 6-11.
    _chunk_rows("0000", range(0, 8), "a")
    _chunk_rows("0001", range(4, N), "b")
    _index((1_000, 1_300, 1_000, 1_350), (1_250, 1_550, 1_200, 1_550))


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
    # Skips without --force.
    run(ComponentConfig(), bag_id=BAG)
    run(ComponentConfig(), bag_id=BAG, force=True)
    assert len(tracks.read(BAG)[1]) == 2


def test_each_car_is_one_track_across_the_chunk_seam():
    _stage_two_chunks()
    run(ComponentConfig(), bag_id=BAG)
    frames, summaries = tracks.read(BAG)
    assert len(summaries) == 2
    for s in summaries:
        fs = [f for f in frames if f.track_id == s.track_id]
        assert s.track_id.startswith("0000-")
        assert {f.chunk_id for f in fs} == {"0000", "0001"}
        # 16 rows (4 instants seen by both chunks), one owned per instant.
        assert len(fs) == 16 and s.n_frames == N
        owned = sorted(f.reference_timestamp_ns for f in fs if f.owned)
        assert owned == [_t(k) for k in range(N)]
        assert {
            f.chunk_id for f in fs if f.owned and f.reference_timestamp_ns >= _t(6)
        } == {"0001"}
    moving = next(s for s in summaries if s.motion_state == "moving")
    assert moving.net_displacement_m == pytest.approx(0.5 * (N - 1 - 2), abs=0.6)


def test_a_track_only_the_non_owner_saw_is_dropped():
    # Chunk 0000 sees the cars at frames 6-7, which chunk 0001 owns; 0001
    # sees nothing.
    _chunk_rows("0000", range(6, 8), "a")
    proposals.write(BAG, "0001", [])
    _index((1_000, 1_300, 1_000, 1_350), (1_250, 1_550, 1_200, 1_550))
    run(ComponentConfig(), bag_id=BAG)
    assert tracks.read(BAG) == ([], [])


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
    assert not det(
        ["lifted-i-000001"], lift_confidence=0.9, points=30, lift_seed_point_count=19
    ).strong
    assert det(
        ["lifted-i-000001"], lift_confidence=0.9, points=30, lift_seed_point_count=20
    ).strong
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
