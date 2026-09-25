"""Tests for Step E — hybrid UniLiPs IWU (iwu/_core.py).

Geometry used throughout: sensor at (0, 0, 1), a "car" block at x≈10 and a
wall at x=20 behind it. Scan clouds are dense enough (5 cm) that every
map-point pixel of the hdl32e full-sphere image contains a return.
"""

from __future__ import annotations

import numpy as np
import pytest

from wato_lidar_preprocessing.config import ComponentConfig, IWUParams
from wato_lidar_preprocessing.iwu import (
    ALPHA,
    P_INIT,
    IWUState,
    load_global_iwu,
    run_iwu,
    update_with_sweep,
)
from wato_lidar_preprocessing.iwu._core import (
    _MAX_HALF_WINDOW_ROWS,
    _sampled_sweeps,
    _seen_through,
    _TileIndex,
)
from wato_lidar_preprocessing.iwu import image_geometry
from wato_lidar_preprocessing.sensor_model import get_sensor_model

from ._staging import (
    grid_points,
    meta_row,
    write_chunks,
    write_global_static_map,
    write_index,
    write_world,
)

SENSOR = get_sensor_model("hdl32e")
ORIGIN = np.array([0.0, 0.0, 1.0])
MATCH = 0.30

CAR_MAP = grid_points(10.0, (-1.0, 1.0), (0.3, 1.5), 0.3)
WALL_MAP = grid_points(20.0, (-3.0, 3.0), (0.0, 2.4), 0.3)
CAR_SCAN = grid_points(10.0, (-1.0, 1.0), (0.3, 1.5), 0.05)
WALL_SCAN = grid_points(20.0, (-4.0, 4.0), (-1.5, 3.5), 0.05)


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _setup(map_xyz):
    return IWUState.fresh(map_xyz.shape[0]), _TileIndex(map_xyz[:, :2])


def _update(state, map_xyz, tiles, scan, travel=0.0):
    return update_with_sweep(
        state,
        map_xyz,
        tiles,
        scan,
        None,
        ORIGIN,
        SENSOR,
        MATCH,
        MATCH,
        sensor_travel_m=travel,
    )


def test_ema_matches_hand_computation():
    """One match then one seen-through at close range (r* = 1, C = 0)."""
    m = np.array([[10.0, 0.0, 1.0]])
    state, tiles = _setup(m)
    _update(state, m, tiles, m.copy())  # return exactly on the map point
    assert state.p_static[0] == pytest.approx(ALPHA * P_INIT + (1 - ALPHA) * 1.0)
    # Nothing near the map point, but a return straight behind it at 20 m.
    _update(state, m, tiles, np.array([[20.0, 0.0, 1.0]]))
    assert state.p_static[0] == pytest.approx(ALPHA * (ALPHA * P_INIT + (1 - ALPHA)))
    assert state.n_match[0] == 1 and state.n_seen_through[0] == 1


def test_repeatedly_observed_wall_stays_static():
    state, tiles = _setup(WALL_MAP)
    for _ in range(5):
        _update(state, WALL_MAP, tiles, WALL_SCAN)
    assert (state.p_static > 0.9).all()
    assert not state.evicted(3).any()


def test_parked_then_gone_object_is_evicted_wall_survives():
    map_xyz = np.vstack([CAR_MAP, WALL_MAP])
    is_car = np.arange(map_xyz.shape[0]) < CAR_MAP.shape[0]
    state, tiles = _setup(map_xyz)
    for _ in range(3):  # car parked
        _update(state, map_xyz, tiles, np.vstack([CAR_SCAN, WALL_SCAN]))
    for _ in range(6):  # car gone: its map points are seen through
        _update(state, map_xyz, tiles, WALL_SCAN)
    ev = state.evicted(3)
    assert ev[is_car].all()
    assert not ev[~is_car].any()


def test_occluded_map_points_are_never_decayed():
    """A pedestrian in front of the wall hides part of it. The paper's
    literal rule would decay the wall's nearest map points from the
    pedestrian's returns; the hybrid rule must leave them untouched."""
    ped = grid_points(10.0, (-0.5, 0.5), (0.0, 2.0), 0.05)
    # Wall returns everywhere except the pedestrian's shadow on x=20.
    shadow = (np.abs(WALL_SCAN[:, 1]) < 1.6) & (WALL_SCAN[:, 2] < 3.4)
    scan = np.vstack([ped, WALL_SCAN[~shadow]])
    state, tiles = _setup(WALL_MAP)
    for _ in range(5):
        _update(state, WALL_MAP, tiles, scan)
    hidden = (np.abs(WALL_MAP[:, 1]) < 1.0) & (WALL_MAP[:, 2] < 2.9)
    assert hidden.any()
    assert (state.n_seen_through[hidden] == 0).all()
    assert (state.n_match[hidden] == 0).all()
    assert np.allclose(state.p_static[hidden], P_INIT)


def test_out_of_range_points_not_updated():
    far = np.array([[SENSOR.max_range_m + 50.0, 0.0, 1.0]])
    state, tiles = _setup(far)
    _update(state, far, tiles, WALL_SCAN)
    assert state.n_match[0] == 0 and state.n_seen_through[0] == 0


# --------------------------------------------------------------------------- #
# Seen-through window (the fix for real-data over-eviction)
# --------------------------------------------------------------------------- #

H, W = image_geometry(SENSOR)  # hdl32e: 1.33° rows, 0.33° columns


def _img_with(*returns):
    img = np.full((H, W), np.inf)
    for (row, col), rng in returns:
        img[row, col] = rng
    return img


def _one(img, r, lateral, row=60, col=500):
    return bool(
        _seen_through(
            img, np.array([row]), np.array([col]), np.array([r]), lateral, MATCH
        )[0]
    )


def test_seen_through_vetoed_by_return_on_next_ring():
    """Map point between rings: its own pixel saw the far background, but the
    next ring hit its surface at its range. Single-pixel logic called that
    seen-through; the window must not."""
    img = _img_with(((60, 500), 40.0), ((61, 500), 20.05))
    assert not _one(img, 20.0, MATCH)
    # With only the far return in reach, it IS seen through.
    assert _one(_img_with(((60, 500), 40.0)), 20.0, MATCH)


def test_sensor_travel_widens_the_window():
    """A near return 5 columns (1.7°) away is outside a 0.3 m window at 20 m
    (±4 columns incl. the aliasing column) but inside once half-sweep sensor
    travel widens it to 0.6 m (±7 columns)."""
    img = _img_with(((60, 500), 40.0), ((60, 505), 20.05))
    assert _one(img, 20.0, MATCH)
    assert not _one(img, 20.0, MATCH + 0.3)


def test_too_close_to_resolve_is_never_seen_through():
    """At 2 m a 0.3 m window spans more than the max rows → no decay."""
    theta_rows = np.degrees(np.arctan2(MATCH, 2.0)) / (180.0 / H)
    assert np.ceil(theta_rows) + 1 > _MAX_HALF_WINDOW_ROWS
    assert not _one(_img_with(((60, 500), 40.0)), 2.0, MATCH)


def test_empty_window_is_no_evidence():
    assert not _one(_img_with(), 20.0, MATCH)


def test_reinforces_every_supported_map_point_not_just_nearest():
    """Two map points 0.2 m apart, one return between them: both are within
    the match radius, both are reinforced (the paper would pick one)."""
    m = np.array([[10.0, 0.0, 1.0], [10.0, 0.2, 1.0]])
    state, tiles = _setup(m)
    _update(state, m, tiles, np.array([[10.0, 0.1, 1.0]]))
    assert state.n_match.tolist() == [1, 1]


# --------------------------------------------------------------------------- #
# Bag-level orchestration
# --------------------------------------------------------------------------- #


def test_sampled_sweeps_dedupes_overlap_and_strides_to_rate(tmp_env):
    bag = "bag_iwu_sample"
    write_chunks(bag, ["c0", "c1"])
    xyz = np.array([[5.0, 0.0, 1.0]])
    for chunk, sweep_ids in (("c0", range(0, 20)), ("c1", range(15, 35))):
        rows = []
        for s in sweep_ids:
            uri = write_world(bag, chunk, s, xyz, ORIGIN)
            rows.append(meta_row(bag, chunk, s, uri, 1))
        write_index(bag, chunk, rows)
    cfg = ComponentConfig(
        sensor_model={"profile": "hdl32e"}, iwu=IWUParams(update_rate_hz=4.0)
    )
    picked = _sampled_sweeps(cfg, bag)
    # 35 unique sweeps at 20 Hz → stride 5 → sweeps 0, 5, …, 30.
    assert [int(r["sweep_id"]) for r in picked] == list(range(0, 35, 5))


def test_run_iwu_end_to_end_writes_global_iwu(tmp_env):
    bag = "bag_iwu_e2e"
    write_chunks(bag, ["c0"])
    write_global_static_map(bag, np.vstack([CAR_MAP, WALL_MAP]))
    rows = []
    for s in range(9):
        scan = np.vstack([CAR_SCAN, WALL_SCAN]) if s < 3 else WALL_SCAN
        uri = write_world(bag, "c0", s, scan, ORIGIN)
        rows.append(meta_row(bag, "c0", s, uri, scan.shape[0]))
    write_index(bag, "c0", rows)
    # Sample every sweep so the 3-then-6 sequence is seen in full.
    cfg = ComponentConfig(
        sensor_model={"profile": "hdl32e"},
        global_map_voxel_size_m=MATCH,
        iwu=IWUParams(update_rate_hz=20.0),
    )
    res = run_iwu(cfg, bag)
    assert res.n_sweeps_used == 9
    assert res.n_evicted == CAR_MAP.shape[0]
    iwu_map = load_global_iwu(bag)
    assert iwu_map is not None
    assert iwu_map.evicted_xyz.shape[0] == CAR_MAP.shape[0]
    assert iwu_map.kept_xyz.shape[0] == WALL_MAP.shape[0]


def test_run_iwu_requires_global_map(tmp_env):
    write_chunks("bag_no_map", ["c0"])
    with pytest.raises(FileNotFoundError, match="reduce"):
        run_iwu(ComponentConfig(), "bag_no_map")


def test_load_global_iwu_missing_returns_none(tmp_env):
    assert load_global_iwu("never_ran") is None
