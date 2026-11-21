"""Re-linking a chunk's tracks after the forward pass (Offline-Poly's STWO).

A car drives +x at 10 m/s, frames 50 ms apart; its tracks are cut by hand.
"""

from __future__ import annotations

from wato_common.artifacts.proposals import Proposal
from wato_common.schemas import Box3D
from wato_tracking.config import RelinkConfig
from wato_tracking.detections import DetFrame, Detection
from wato_tracking.relink import relink
from wato_tracking.tracker import Observation

MS = 1_000_000
N = 80
FRAMES = [DetFrame(k, k * 50 * MS, ()) for k in range(N)]
V = 10.0


def _obs(k, cx, cy=0.0, cls="car", key=None, l=4.4, w=1.8, v=V) -> Observation:  # noqa: E741
    p = Proposal(
        proposal_id=f"fused-{k}-{cx}-{cy}",
        sweep_id=k,
        frame_id=k,
        sweep_ids=(k,),
        reference_timestamp_ns=FRAMES[k].t_ns,
        cx=cx,
        cy=cy,
        cz=0.8,
        w=w,
        l=l,
        h=1.6,
        heading=0.0,
        cls=cls,
        score=1.0,
        provenance="fused",
        member_proposal_ids=(),
    )
    det = Detection(p, strong=True, key=key, two_source=False, has_lidar_mos=False)
    box = Box3D(cx=cx, cy=cy, cz=0.8, w=w, l=l, h=1.6, heading=0.0)
    return Observation(FRAMES[k], det, box, (v, 0.0, 0.0))


def _car(ks, **kw) -> list[Observation]:
    return [_obs(k, V * 0.05 * k, **kw) for k in ks]


def _relink(tracks, **kw):
    return relink(tracks, FRAMES, RelinkConfig(**kw), max_centre_dist_m=2.0)


def _sizes(tracks):
    return sorted(len(t) for t in tracks)


def test_two_pieces_of_one_car_across_a_gap_are_fused():
    out = _relink([_car(range(0, 10)), _car(range(16, 26))])
    assert _sizes(out) == [20]
    assert [o.frame.sweep_id for o in out[0]] == list(range(0, 10)) + list(
        range(16, 26)
    )


def test_chains_form_over_sweeps():
    pieces = [_car(range(0, 10)), _car(range(15, 25)), _car(range(30, 40))]
    assert _sizes(_relink(pieces)) == [30]


def test_a_gap_beyond_two_horizons_is_never_fused():
    # 2.5 s apart: no frame where both predictions are within 1 s.
    assert _sizes(_relink([_car(range(0, 10)), _car(range(60, 70))])) == [10, 10]


def test_tracks_alive_at_once_are_never_fused():
    a = _car(range(0, 20))
    b = [_obs(k, V * 0.05 * k + 0.2) for k in range(10, 30)]
    assert _sizes(_relink([a, b])) == [20, 20]


def test_known_different_classes_are_kept_apart_unless_told_otherwise():
    tracks = [_car(range(0, 10)), _car(range(16, 26), cls="truck")]
    assert _sizes(_relink(tracks)) == [10, 10]
    assert _sizes(_relink(tracks, class_consistent=False)) == [20]
    # An unclassed piece is consistent with any class.
    assert _sizes(_relink([_car(range(0, 10)), _car(range(16, 26), cls=None)])) == [20]


def test_a_neighbouring_car_does_not_fuse():
    # The second car drives 3 m to the side: the boxes never overlap.
    tracks = [
        _car(range(0, 10)),
        [_obs(k, V * 0.05 * k, cy=3.0) for k in range(16, 26)],
    ]
    assert _sizes(_relink(tracks)) == [10, 10]


def test_a_shared_instance_fuses_a_one_sided_box_with_no_overlap():
    # After the gap only a 0.2 m-wide side face, 1 m off the predicted centre,
    # is seen: IoU 0, but SAM2 followed the car through both pieces.
    a = _car(range(0, 10), key="i1")
    b = [_obs(k, V * 0.05 * k, cy=1.0, l=4.4, w=0.2, key="i1") for k in range(16, 26)]
    assert _sizes(_relink([a, b], shared_instance=False)) == [10, 10]
    assert _sizes(_relink([a, b])) == [20]
    other = [
        _obs(o.frame.sweep_id, o.det.proposal.cx, cy=1.0, w=0.2, key="i2") for o in b
    ]
    assert _sizes(_relink([a, other])) == [10, 10]


def test_one_to_one_the_better_overlap_wins():
    a = _car(range(0, 10))
    good = _car(range(14, 24))
    off = [_obs(k, V * 0.05 * k, cy=1.2) for k in range(14, 24)]
    out = _relink([a, good, off])
    assert _sizes(out) == [10, 20]
    fused = max(out, key=len)
    assert all(o.det.proposal.cy == 0.0 for o in fused)
