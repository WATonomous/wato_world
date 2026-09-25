"""Step E — UniLiPs Iterative Weighted Update (IWU), bag level.

UniLiPs (arXiv 2601.05105, §3, Eqs. 2–4) refines an accumulated static map
by giving every map point m a static probability P(m), updated as each scan
is compared against the map:

    reinforce (Eq. 3):  P ← α·P + (1−α)·r*·(1+C(m))
    decay     (Eq. 4):  P ← α·P + (1−α)·(1−r*)·(1−C(m))

Points ending below τ_s are "floaters" — structure that was there when the
map was built and is gone in other scans: parked-then-moved cars, pedestrians
that stood still for a chunk. Those are exactly the objects a single chunk's
log-odds grid calls static, which is why this runs at bag level.

Our variant ("hybrid", chosen over the paper-literal rule):

  * Reinforce as the paper's Eq. 3, on every map point with a non-ground
    return within the match radius (30 cm in the paper; here
    global_map_voxel_size_m, the map's own resolution — 0.30 m by default).
    The paper reinforces only each return's single nearest map point, but
    this map is voxel-snapped at the same pitch as the radius, so "nearest"
    is an artefact of the snap: on real 32-beam data it starved the map
    points a scan lands between and evicted them. One update per map point
    per sweep.
  * Decay only on explicit free-space evidence. The paper decays a scan
    point's nearest map point whenever it is > 30 cm away, which lets the
    returns of a new or unmapped object (a pedestrian in front of a wall)
    erode true static structure behind it, and never distinguishes
    "occluded" from "gone". Here a map point decays only when EVERY return
    whose ray passes within ρ of it laterally ended beyond it (seen
    through): a range-adaptive min filter over the sweep's range image,
    ρ = match radius + the sensor's travel during half a sweep (world NPZs
    keep one mid-sweep origin, so a ray's true start is that uncertain).
    Occluded points, points with no nearby return, points too close to the
    sensor for the window to resolve, and points out of view are not updated.
    Measured on nuScenes scene-0061 (one 19 s chunk, AW static map, 191k
    map points): a single-pixel test with nearest-only reinforcement evicted
    54% of the map — ring gaps and origin error read as "seen through". The
    windowed test with every-supported reinforcement evicts 7.0%, and
    eviction is 2.0× more likely within 0.5 m of the union dynamic cloud than
    elsewhere (the only precision proxy available without labels).

Other deviations, each deliberate:
  * r* = SensorModel.range_weight(r, global_map_voxel_size_m) — the same
    beam-footprint credibility classify uses (crossover d* = voxel /
    divergence, ≈100 m on these scanners) instead of the paper's fixed
    r_max = 200 m, so one range model governs the whole component.
  * C(m) = 0. The label-consensus term needs semantic labels this stage
    doesn't have; `consensus` is the hook for semantic_lifting's accumulated
    (label, count) map later.
  * The range image is world-axis-aligned and spans the full sphere, centred
    on the sweep origin. World NPZs store the sensor position but not its
    orientation, and a full-sphere image makes sensor tilt irrelevant: a map
    point whose direction the scanner never sampled simply lands on an empty
    pixel and is skipped. Pixel size is the scanner's native angular step.
  * P starts at P_INIT = 0.5 (the paper doesn't state it) and an eviction
    needs at least cfg.min_observations updates, so a single stray
    seen-through can't evict a point nobody else observed.

The EMA is order-dependent: P reads as "consistent with the most recent
looks". For a recall-oriented proposal source that is the useful semantics —
an object that was parked and then left ends evicted.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from tqdm import tqdm

from wato_common.artifact_store import (
    chunks_index_path,
    global_iwu_path,
    global_static_map_path,
    lidar_proc_index_path,
    local_path,
)
from wato_common.io.parquet_io import read_rows
from wato_lidar_preprocessing.classify.io_helpers import load_world_full
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.range_image import min_range_image, spherical_pixels
from wato_lidar_preprocessing.sensor_model import SensorModel

log = logging.getLogger(__name__)

# UniLiPs §3 constants (paper values; not tuned here).
ALPHA = 0.7  # EMA memory
TAU_STATIC = 0.5  # static iff P >= τ_s
# Initial static probability. Unstated in the paper; 0.5 = no opinion, so the
# first observation decides the direction and eviction needs real evidence.
P_INIT = 0.5

# Widest seen-through window, in image rows (≈ beam rings) either side. A map
# point whose window needs more is too close to the sensor for one mid-sweep
# origin to resolve (≈ 9 m on these scanners), so it gets no decay at all.
_MAX_HALF_WINDOW_ROWS = 4

# Side of the 2D tiles used to crop the map to a sweep's range [m]. Pure
# performance: ~(2·range/tile)² tile lookups per sweep instead of a distance
# test against every map point.
_TILE_M = 25.0


@dataclass
class IWUState:
    """Per-map-point IWU state. Arrays are aligned to the map's xyz."""

    p_static: np.ndarray  # float64
    n_match: np.ndarray  # int32
    n_seen_through: np.ndarray  # int32

    @classmethod
    def fresh(cls, n: int) -> "IWUState":
        return cls(
            p_static=np.full(n, P_INIT, dtype=np.float64),
            n_match=np.zeros(n, dtype=np.int32),
            n_seen_through=np.zeros(n, dtype=np.int32),
        )

    def evicted(self, min_updates: int) -> np.ndarray:
        n_upd = self.n_match + self.n_seen_through
        return (self.p_static < TAU_STATIC) & (n_upd >= min_updates)


@dataclass
class IWUResult:
    out_uri: str
    n_map_points: int
    n_sweeps_used: int
    n_evicted: int


class _TileIndex:
    """2D tile buckets over the map for fast "points within R of origin"."""

    def __init__(self, xy: np.ndarray, tile_m: float = _TILE_M) -> None:
        self.tile = float(tile_m)
        ij = np.floor(xy / self.tile).astype(np.int64)
        self._lo = ij.min(axis=0) if ij.shape[0] else np.zeros(2, dtype=np.int64)
        ij = ij - self._lo
        self._ny = int(ij[:, 1].max()) + 1 if ij.shape[0] else 1
        keys = ij[:, 0] * self._ny + ij[:, 1]
        self._order = np.argsort(keys, kind="stable")
        sk = keys[self._order]
        ukeys, starts = np.unique(sk, return_index=True)
        ends = np.append(starts[1:], sk.shape[0])
        self._spans = {int(k): (int(a), int(b)) for k, a, b in zip(ukeys, starts, ends)}

    def candidates(self, centre_xy: np.ndarray, radius: float) -> np.ndarray:
        lo = np.floor((centre_xy - radius) / self.tile).astype(np.int64) - self._lo
        hi = np.floor((centre_xy + radius) / self.tile).astype(np.int64) - self._lo
        parts = []
        for ix in range(max(lo[0], 0), hi[0] + 1):
            for iy in range(max(lo[1], 0), min(hi[1], self._ny - 1) + 1):
                span = self._spans.get(ix * self._ny + iy)
                if span is not None:
                    parts.append(self._order[span[0] : span[1]])
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def image_geometry(sensor: SensorModel) -> tuple[int, int]:
    """(rows, cols) of the full-sphere image at the scanner's native step."""
    v_step = (sensor.fov_up_deg - sensor.fov_down_deg) / max(sensor.beams - 1, 1)
    return int(math.ceil(180.0 / v_step)), sensor.azimuth_columns


def _ema(state: IWUState, idx: np.ndarray, target: np.ndarray) -> None:
    p = ALPHA * state.p_static[idx] + (1.0 - ALPHA) * target
    state.p_static[idx] = np.clip(p, 0.0, 1.0)


def _seen_through(
    img: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    r: np.ndarray,
    lateral_m: float,
    range_tol_m: float,
) -> np.ndarray:
    """True where EVERY return within `lateral_m` of the point's direction
    ended more than `range_tol_m` beyond it.

    The per-point angular window is atan(lateral_m / r), in image rows and
    columns, plus one row and one column: the image grid is world-aligned
    while the scanner's rings are not, so the ring nearest a point can alias
    one row further than the angle alone says. Applied as a min filter
    (UniLiPs Eq. 1's min-over-neighbourhood, used here for free space rather
    than occlusion). A return anywhere in the window that stops short of the
    point — the point's own surface on the next ring, an occluder — vetoes
    the decay. Points whose window would span more than
    _MAX_HALF_WINDOW_ROWS rows are too close to judge and are never marked
    seen-through.
    """
    h, w = img.shape
    theta = np.arctan2(lateral_m, np.maximum(r, 1e-6))
    hw_r = np.ceil(theta / (np.pi / h)).astype(np.int64) + 1
    hw_c = np.ceil(theta / (2.0 * np.pi / w)).astype(np.int64) + 1
    ok = hw_r <= _MAX_HALF_WINDOW_ROWS
    measured = np.full(r.shape[0], np.inf)
    if ok.any():
        key = hw_r * (w + 1) + hw_c
        for k in np.unique(key[ok]):
            a, b = divmod(int(k), w + 1)
            filt = ndimage.minimum_filter(
                img, size=(2 * a + 1, 2 * b + 1), mode=("nearest", "wrap")
            )
            sel = ok & (key == k)
            measured[sel] = filt[row[sel], col[sel]]
    return ok & np.isfinite(measured) & (measured > r + range_tol_m)


def update_with_sweep(
    state: IWUState,
    map_xyz: np.ndarray,
    tiles: _TileIndex,
    scan_xyz: np.ndarray,
    scan_ground: np.ndarray | None,
    origin: np.ndarray,
    sensor: SensorModel,
    match_radius_m: float,
    credibility_voxel_m: float,
    *,
    sensor_travel_m: float = 0.0,
    consensus: np.ndarray | None = None,
) -> tuple[int, int]:
    """Apply one sweep's hybrid IWU update in place.

    sensor_travel_m: how far the sensor moved during half this sweep. World
        NPZs keep one mid-sweep origin, so a return's true ray started up to
        this far from `origin`; it widens the seen-through window.

    Returns (n_reinforced, n_decayed) map points.
    """
    if scan_xyz.shape[0] == 0 or map_xyz.shape[0] == 0:
        return 0, 0

    cand = tiles.candidates(origin[:2], sensor.max_range_m)
    if cand.size == 0:
        return 0, 0
    rel = map_xyz[cand] - origin
    r = np.sqrt(np.einsum("ij,ij->i", rel, rel))
    keep = r <= sensor.max_range_m
    cand, rel, r = cand[keep], rel[keep], r[keep]
    if cand.size == 0:
        return 0, 0

    # --- Reinforce (Eq. 3): map points with a non-ground return within the
    # match radius. The paper reinforces each return's single nearest map
    # point; the map here is voxel-snapped at the same pitch as the radius,
    # so "nearest" is an artefact of the snap and starves the map points a
    # 32-beam scan lands between. Every map point a return supports counts.
    ng = scan_xyz if scan_ground is None else scan_xyz[~scan_ground.astype(bool)]
    supported = np.zeros(cand.size, dtype=bool)
    if ng.shape[0] > 0:
        dist, _ = cKDTree(ng).query(
            map_xyz[cand], k=1, distance_upper_bound=match_radius_m, workers=-1
        )
        supported = np.isfinite(dist)
    reinforced = cand[supported]
    if reinforced.size:
        r_star = sensor.range_weight(r[supported], credibility_voxel_m)
        c = consensus[reinforced] if consensus is not None else 0.0
        _ema(state, reinforced, r_star * (1.0 + c))
        state.n_match[reinforced] += 1

    # --- Decay (Eq. 4): only map points the sweep saw through. ------------
    cand, rel, r = cand[~supported], rel[~supported], r[~supported]
    if cand.size == 0:
        return int(reinforced.size), 0
    h, w = image_geometry(sensor)
    img = min_range_image(scan_xyz - origin, h, w, 90.0, -90.0)
    row, col, _, in_fov = spherical_pixels(rel, h, w, 90.0, -90.0)
    seen = in_fov & _seen_through(
        img, row, col, r, match_radius_m + sensor_travel_m, match_radius_m
    )
    decayed = cand[seen]
    if decayed.size:
        r_star = sensor.range_weight(r[seen], credibility_voxel_m)
        c = consensus[decayed] if consensus is not None else 0.0
        _ema(state, decayed, (1.0 - r_star) * (1.0 - c))
        state.n_seen_through[decayed] += 1
    return int(reinforced.size), int(decayed.size)


def _sampled_sweeps(cfg: ComponentConfig, bag_id: str) -> list[dict]:
    """Valid sweeps of the bag, deduplicated across chunk overlaps, strided
    per lidar to cfg.iwu.update_rate_hz, in timestamp order."""
    seen: set[tuple[str, int]] = set()
    rows: list[dict] = []
    for chunk in read_rows(chunks_index_path(bag_id)):
        idx_path = local_path(lidar_proc_index_path(bag_id, chunk["chunk_id"]))
        if not os.path.exists(idx_path):
            continue
        for r in read_rows(lidar_proc_index_path(bag_id, chunk["chunk_id"])):
            if r.get("valid") is False or not r.get("world_path"):
                continue
            key = (str(r["lidar_id"]), int(r["reference_timestamp_ns"]))
            if key in seen:  # same sweep in two overlapping chunks
                continue
            seen.add(key)
            rows.append(r)
    rows.sort(key=lambda r: int(r["reference_timestamp_ns"]))

    by_lidar: dict[str, list[dict]] = {}
    for r in rows:
        by_lidar.setdefault(str(r["lidar_id"]), []).append(r)
    kept: list[dict] = []
    for lidar_id, lrows in by_lidar.items():
        sensor = cfg.build_sensor_model(lidar_id)
        stride = max(1, round(sensor.sweep_rate_hz / cfg.iwu.update_rate_hz))
        kept.extend(lrows[::stride])
    kept.sort(key=lambda r: int(r["reference_timestamp_ns"]))
    return kept


def _half_sweep_travel(
    prev: tuple[np.ndarray, int] | None,
    origin: np.ndarray,
    t_ns: int,
    sensor: SensorModel,
) -> float:
    """Sensor travel during half a sweep [m], from the previous sampled
    sweep of the same lidar (0 for a lidar's first sweep)."""
    if prev is None or t_ns <= prev[1]:
        return 0.0
    speed = float(np.linalg.norm(origin - prev[0])) / ((t_ns - prev[1]) * 1e-9)
    return speed * sensor.sweep_duration_ms * 1e-3 / 2.0


def _write(
    bag_id: str, map_xyz: np.ndarray, state: IWUState, evicted: np.ndarray, **meta
) -> str:
    out_uri = global_iwu_path(bag_id)
    np.savez_compressed(
        local_path(out_uri),
        xyz=map_xyz,
        p_static=state.p_static.astype(np.float32),
        n_match=state.n_match,
        n_seen_through=state.n_seen_through,
        evicted=evicted,
        alpha=np.float32(ALPHA),
        tau_static=np.float32(TAU_STATIC),
        **{k: np.asarray(v) for k, v in meta.items()},
    )
    return out_uri


def run_iwu(cfg: ComponentConfig, bag_id: str) -> IWUResult:
    """Run IWU over the whole bag and write global_iwu.npz."""
    map_path = local_path(global_static_map_path(bag_id))
    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"global_static_map.npz not found for bag {bag_id!r} — run `reduce` first"
        )
    map_xyz = np.asarray(np.load(map_path)["xyz"], dtype=np.float64)
    state = IWUState.fresh(map_xyz.shape[0])
    if map_xyz.shape[0] == 0:
        log.warning("bag %s: global static map is empty — IWU is a no-op", bag_id)
        out = _write(bag_id, map_xyz, state, np.zeros(0, dtype=bool), n_sweeps_used=0)
        return IWUResult(out, 0, 0, 0)

    sweeps = _sampled_sweeps(cfg, bag_id)
    tiles = _TileIndex(map_xyz[:, :2])
    match_radius = cfg.global_map_voxel_size_m
    log.info(
        "bag %s: IWU over %d map points, %d sampled sweeps "
        "(α=%.2f τ=%.2f match=%.2fm rate=%.1fHz)",
        bag_id,
        map_xyz.shape[0],
        len(sweeps),
        ALPHA,
        TAU_STATIC,
        match_radius,
        cfg.iwu.update_rate_hz,
    )

    n_used = 0
    last: dict[str, tuple[np.ndarray, int]] = {}  # lidar → (origin, t_ns)
    for row in tqdm(sweeps, desc=f"iwu {bag_id}", unit="sweep"):
        xyz, _, origin, ground = load_world_full(row["world_path"])
        if origin is None or xyz.shape[0] == 0:
            continue
        origin = np.asarray(origin, dtype=np.float64)
        lidar_id = str(row["lidar_id"])
        sensor = cfg.build_sensor_model(lidar_id)
        t_ns = int(row["reference_timestamp_ns"])
        update_with_sweep(
            state,
            map_xyz,
            tiles,
            xyz,
            ground,
            origin,
            sensor,
            match_radius,
            credibility_voxel_m=cfg.global_map_voxel_size_m,
            sensor_travel_m=_half_sweep_travel(
                last.get(lidar_id), origin, t_ns, sensor
            ),
        )
        last[lidar_id] = (origin, t_ns)
        n_used += 1

    evicted = state.evicted(cfg.min_observations)
    out = _write(
        bag_id,
        map_xyz,
        state,
        evicted,
        n_sweeps_used=n_used,
        update_rate_hz=np.float32(cfg.iwu.update_rate_hz),
    )
    log.info(
        "bag %s: IWU evicted %d / %d map points (%d never observed) → %s",
        bag_id,
        int(evicted.sum()),
        map_xyz.shape[0],
        int(((state.n_match + state.n_seen_through) == 0).sum()),
        out,
    )
    return IWUResult(out, map_xyz.shape[0], n_used, int(evicted.sum()))


@dataclass
class IWUMap:
    """global_iwu.npz split into the refined static map and the floaters."""

    kept_xyz: np.ndarray
    evicted_xyz: np.ndarray


def load_global_iwu(bag_id: str) -> IWUMap | None:
    """None when IWU hasn't run for the bag."""
    path = local_path(global_iwu_path(bag_id))
    if not os.path.exists(path):
        return None
    d = np.load(path)
    xyz = np.asarray(d["xyz"], dtype=np.float64)
    ev = np.asarray(d["evicted"], dtype=bool)
    return IWUMap(kept_xyz=xyz[~ev], evicted_xyz=xyz[ev])
