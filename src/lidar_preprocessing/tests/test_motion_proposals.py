"""Tests for Step F — recall-oriented motion proposals (motion_proposals/).

The end-to-end scene (12 sweeps at 20 Hz, sensor at (0, 0, 1.8)):
  * mover   — a 4×2×1.5 m car driving +1 m/frame along x at y=+5, flagged by
              the seg method (SEG_DYNAMIC), with "wheel" returns below the
              0.25 m height floor that only BOX_FILL may recover;
  * parked  — the same car standing still at y=−5, a seg false positive,
              also covered by IWU-evicted map points (IWU_EVICTED);
  * slider  — a 6 m wall fragment whose *visible* window slides 0.3 m/frame
              along the wall (x=30), an AW false positive (AW_DYNAMIC). Its
              centroid moves, the object doesn't — main's measured failure
              mode for velocity gates;
  * ground  — a Patchwork++-flagged road plane, and near-ego self-returns.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    global_iwu_path,
    lidar_proc_summary_path,
    local_path,
    motion_clusters_path,
    motion_proposals_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows
from wato_common.schemas import Box3D
from wato_lidar_preprocessing.config import (
    ComponentConfig,
    MotionFilterParams,
    UnionParams,
)
from wato_lidar_preprocessing.motion_proposals import (
    AW_AMBIGUOUS,
    AW_DYNAMIC,
    BOX_FILL,
    IWU_EVICTED,
    MF_MOS,
    SEG_DYNAMIC,
    UNMAPPED,
    cluster_points,
    compute_sweep_bits,
    fill_box,
    process_chunk,
    proposals_up_to_date,
    should_box_fill,
    track_motion,
)
from wato_lidar_preprocessing.motion_proposals._core import _Sources

from ._staging import (
    grid_points,
    meta_row,
    write_chunks,
    write_dynamic_mask,
    write_flat_ground,
    write_index,
    write_static_map,
    write_summary,
    write_world,
)

ORIGIN = np.array([0.0, 0.0, 1.8])
VOX_ORIGIN = np.array([-50.0, -50.0, -10.0])
VOXEL = 0.5
N_SWEEPS = 12


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _cfg(**mp_kw) -> ComponentConfig:
    return ComponentConfig(
        sensor_model={"profile": "hdl32e"},
        voxel_size_m=VOXEL,
        global_map_voxel_size_m=0.3,
        dynamic_min_range_m=2.5,
        union=UnionParams(motion_filter=MotionFilterParams(persistence_max_sweeps=10)),
        motion_proposals=mp_kw or {},
    )


def _car(cx: float, cy: float) -> np.ndarray:
    return grid_points((cx - 2.0, cx + 2.0), (cy - 1.0, cy + 1.0), (0.3, 1.8), 0.2)


def _slider(s: int) -> np.ndarray:
    y0 = -3.0 + 0.3 * s
    return grid_points(30.0, (y0, y0 + 6.0), (0.3, 3.0), 0.2)


def _stage_scene(bag: str, chunk: str = "c0", *, with_iwu: bool = True) -> dict:
    write_chunks(bag, [chunk])
    write_summary(bag, chunk)
    write_flat_ground(bag, chunk, z=0.0)
    slider_all = np.vstack([_slider(s) for s in range(N_SWEEPS)])
    write_static_map(bag, chunk, origin=VOX_ORIGIN, voxel=VOXEL, dynamic_xyz=slider_all)
    if with_iwu:
        wall = grid_points(30.0, (-10.0, 10.0), (0.3, 3.0), 0.3)
        parked_ghost = _car(10.0, -5.0)
        xyz = np.vstack([wall, parked_ghost])
        evicted = np.r_[np.zeros(len(wall), bool), np.ones(len(parked_ghost), bool)]
        path = local_path(global_iwu_path(bag))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, xyz=xyz, evicted=evicted)

    rows, layout = [], {}
    road = grid_points((-5.0, 40.0), (-8.0, 8.0), 0.0, 0.5)
    ego = grid_points((0.5, 1.5), (-0.5, 0.5), 1.0, 0.25)
    for s in range(N_SWEEPS):
        mover = _car(10.0 + 1.0 * s, 5.0)
        wheels = grid_points((9.0 + s, 11.0 + s), (4.2, 5.8), 0.1, 0.4)
        parked = _car(10.0, -5.0)
        slider = _slider(s)
        parts = [mover, wheels, parked, slider, road, ego]
        xyz = np.vstack(parts)
        sizes = np.cumsum([0] + [len(p) for p in parts])
        sl = {
            n: slice(sizes[i], sizes[i + 1])
            for i, n in enumerate(
                ["mover", "wheels", "parked", "slider", "road", "ego"]
            )
        }
        ground = np.zeros(len(xyz), bool)
        ground[sl["road"]] = True
        seg = np.zeros(len(xyz), bool)
        seg[sl["mover"]] = True
        seg[sl["parked"]] = True
        seg[sl["ego"]] = True
        uri = write_world(bag, chunk, s, xyz, ORIGIN, ground_mask=ground)
        write_dynamic_mask(bag, chunk, s, seg)
        rows.append(meta_row(bag, chunk, s, uri, len(xyz)))
        layout[s] = sl
    write_index(bag, chunk, rows)
    return layout


def _bits(bag, chunk, s):
    d = np.load(local_path(motion_proposals_path(bag, chunk, s)))
    return d["source_bits"], d["cluster_id"]


def _cluster_near(rows, x, y, frame_id):
    cands = [
        r
        for r in rows
        if r["frame_id"] == frame_id
        and abs(r["cx"] - x) < 1.5
        and abs(r["cy"] - y) < 1.5
    ]
    assert len(cands) == 1, cands
    return cands[0]


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_end_to_end_scene(tmp_env):
    bag, chunk = "bag_mp", "c0"
    layout = _stage_scene(bag, chunk)
    res = process_chunk(_cfg(), bag, chunk)
    rows = read_rows(motion_clusters_path(bag, chunk))
    assert res.n_clusters == len(rows) > 0

    mid = N_SWEEPS // 2
    mover = _cluster_near(rows, 10.0 + mid, 5.0, mid)
    parked = _cluster_near(rows, 10.0, -5.0, mid)
    slider = _cluster_near(rows, 30.0, 0.3 * mid, mid)

    # Motion: Chen's criterion separates the three, and the sliding window
    # (centroid drift, no motion) stays below it.
    assert mover["motion_score"] > 1.0 and mover["track_life"] == N_SWEEPS
    assert parked["motion_score"] < 0.2
    assert slider["motion_score"] < 1.0
    assert mover["box_filled"] and not parked["box_filled"] and not slider["box_filled"]

    # Soft features.
    assert mover["source_bits"] & int(SEG_DYNAMIC)
    assert parked["source_bits"] & int(IWU_EVICTED)
    assert slider["source_bits"] & int(AW_DYNAMIC)
    assert mover["frac_seg_dynamic"] == pytest.approx(1.0)
    assert parked["frac_persistent"] > 0.9  # same voxels all 12 sweeps (>= 10)
    assert mover["frac_persistent"] < 0.1
    assert mover["track_hint_id"] != parked["track_hint_id"]
    assert json.loads(mover["sweep_ids"]) == [mid]

    for s in range(N_SWEEPS):
        bits, cid = _bits(bag, chunk, s)
        sl = layout[s]
        n_world = sl["ego"].stop
        assert bits.shape == cid.shape == (n_world,)
        # Exclusions: ground and near-ego never flagged (ego was seg-dynamic).
        assert not bits[sl["road"]].any()
        assert not bits[sl["ego"]].any()
        # Wheels are below the height floor → no source fires, but BOX_FILL
        # recovers them because the mover's box is grown down to the ground.
        wheel_bits = bits[sl["wheels"]]
        assert (wheel_bits == BOX_FILL).all()
        assert (cid[sl["wheels"]] >= 0).all()
        # MF-MOS never ran → bit 3 never set.
        assert not (bits & MF_MOS).any()
        # UNMAPPED: the cars are far from the IWU-kept wall; the slider isn't.
        assert (bits[sl["mover"]] & UNMAPPED).all()
        assert not (bits[sl["slider"]] & UNMAPPED).any()
        # Parked car: never box-filled. Mover: every point, including the
        # seeds on the box hull (regression: boundary points failed <= by ε).
        assert not (bits[sl["parked"]] & BOX_FILL).any()
        assert (bits[sl["mover"]] & BOX_FILL).all()

    summary = read_rows(lidar_proc_summary_path(bag, chunk))[0]
    assert summary["n_clusters"] == res.n_clusters
    assert summary["n_clusters_moving"] == res.n_clusters_moving >= 1
    assert summary["n_points_proposal"] == res.n_points_proposal > 0


def test_missing_iwu_leaves_only_its_bit_clear(tmp_env):
    bag, chunk = "bag_mp_noiwu", "c0"
    layout = _stage_scene(bag, chunk, with_iwu=False)
    res = process_chunk(_cfg(), bag, chunk)
    assert any("IWU_EVICTED" in m for m in res.missing_sources)
    for s in range(N_SWEEPS):
        bits, _ = _bits(bag, chunk, s)
        assert not (bits & IWU_EVICTED).any()
        # UNMAPPED falls back to the chunk static map (empty here → absent).
        assert (bits[layout[s]["mover"]] & SEG_DYNAMIC).all()


def test_up_to_date_tracks_inputs(tmp_env):
    bag, chunk = "bag_mp_fresh", "c0"
    _stage_scene(bag, chunk)
    assert not proposals_up_to_date(bag, chunk)
    process_chunk(_cfg(), bag, chunk)
    assert proposals_up_to_date(bag, chunk)
    # A newer static map (e.g. a re-run of Step B) makes proposals stale.
    sm = local_path(static_map_path(bag, chunk))
    t = os.path.getmtime(local_path(motion_clusters_path(bag, chunk)))
    os.utime(sm, (t + 10, t + 10))
    assert not proposals_up_to_date(bag, chunk)


def test_box_fill_can_be_disabled(tmp_env):
    bag, chunk = "bag_mp_nofill", "c0"
    _stage_scene(bag, chunk)
    process_chunk(_cfg(box_fill=False), bag, chunk)
    rows = read_rows(motion_clusters_path(bag, chunk))
    assert not any(r["box_filled"] for r in rows)
    for s in range(N_SWEEPS):
        bits, _ = _bits(bag, chunk, s)
        assert not (bits & BOX_FILL).any()


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


def test_compute_sweep_bits_voxel_sources_and_height_floor():
    xyz = np.array(
        [
            [10.0, 0.0, 1.0],  # dynamic voxel
            [12.0, 0.0, 1.0],  # ambiguous voxel
            [14.0, 0.0, 0.1],  # dynamic voxel but under the height floor
            [16.0, 0.0, 1.0],  # nothing
        ]
    )
    from wato_lidar_preprocessing.voxel import voxel_indices

    keys = voxel_indices(xyz, VOX_ORIGIN, VOXEL)
    src = _Sources(
        voxel_origin=VOX_ORIGIN,
        voxel_size=VOXEL,
        dynamic_keys=np.sort(keys[[0, 2]]),
        ambiguous_keys=np.sort(keys[[1]]),
        ground_grid=(np.zeros((400, 400), np.float32), np.array([-100.0, -100.0]), 0.5),
    )
    bits = compute_sweep_bits(xyz, None, ORIGIN, src, _cfg())
    assert bits.tolist() == [int(AW_DYNAMIC), int(AW_AMBIGUOUS), 0, 0]


def test_seed_and_attach_bits_are_disjoint_roles():
    """A point with a seed bit AND an attach bit is a seed, never an attach
    candidate (regression: bitwise ~ on uint8 made it both)."""
    from wato_lidar_preprocessing.motion_proposals import ATTACH_BITS, SEED_BITS

    bits = np.array([AW_DYNAMIC | AW_AMBIGUOUS, AW_AMBIGUOUS, UNMAPPED, 0], np.uint8)
    is_seed = (bits & SEED_BITS) != 0
    is_attach = ((bits & ATTACH_BITS) != 0) & ~is_seed
    assert is_seed.tolist() == [True, False, False, False]
    assert is_attach.tolist() == [False, True, True, False]


def test_cluster_points_separates_blobs_and_marks_noise():
    a = grid_points((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), 0.2)
    b = grid_points((20.0, 21.0), (0.0, 1.0), (0.0, 1.0), 0.2)
    stray = np.array([[50.0, 50.0, 0.0]])
    labels = cluster_points(np.vstack([a, b, stray]), min_cluster_pts=5)
    la, lb = labels[: len(a)], labels[len(a) : len(a) + len(b)]
    assert len(set(la)) == 1 and len(set(lb)) == 1 and la[0] != lb[0]
    assert labels[-1] == -1
    assert (cluster_points(stray, 5) == -1).all()


def test_track_motion_net_not_path_length():
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 0.3, size=(40, 2))  # static object, noisy centroid
    net, score = track_motion(jitter, max_side_m=4.0)
    assert score < 0.3
    line = np.column_stack([np.arange(10.0), np.zeros(10)])
    net, score = track_motion(line, max_side_m=4.0)
    assert net == pytest.approx(7.0)  # median of x=0,1,2 → 1; of x=7,8,9 → 8
    assert score == pytest.approx(7.0 / 4.0)
    assert track_motion(line[:1], 4.0) == (0.0, 0.0)


def test_fill_box_reaches_ground():
    b = Box3D(cx=0, cy=0, cz=1.05, w=2, l=4, h=1.5, heading=0.0)  # z ∈ [0.3, 1.8]
    f = fill_box(b, 0.25)
    assert f.cz - f.h / 2 == pytest.approx(0.05)
    assert f.cz + f.h / 2 == pytest.approx(1.8)


def test_box_fill_needs_motion_life_and_low_persistence():
    assert should_box_fill(2.0, 10, 0.0)
    assert not should_box_fill(0.9, 10, 0.0)  # Chen's criterion
    assert not should_box_fill(2.0, 2, 0.0)  # too short a track
    # Sliding-window wall fragment: displacement beats its size, but its
    # points dwell in the same voxels.
    assert not should_box_fill(1.46, 49, 1.0)
