"""Step F — recall-oriented moving-object proposals.

A seg-agnostic consumer layer (like `union`, it may read every method's
artifacts; `aw` and `mos` still never import each other). Built on Chen et
al., "Automatic Labeling to Generate Training Data for Online LiDAR-based
Moving Object Segmentation" (arXiv 2201.04501): coarse dynamic candidates →
class-agnostic clustering (HDBSCAN) → multi-object tracking → "moved further
than its own size". Two changes to that recipe, both because this artifact is
a *proposal* source whose false positives downstream association removes:

  * Nothing is dropped on motion evidence. Chen relabels non-moving clusters
    static; here every cluster keeps its row with soft features
    (motion_score, track_life, n_sources, frac_seg_dynamic, frac_persistent)
    for downstream to threshold.
  * The coarse stage is a union of every heuristic we have, not one
    map-cleaning method. Each point records which ones fired (source_bits).

Per point, source_bits (uint8):

  bit  name            seeds?  source
  0    AW_DYNAMIC      seed    voxel in static_map.npz dynamic_voxel_keys (aw, union)
  1    AW_AMBIGUOUS    attach  voxel in static_map.npz ambiguous_voxel_keys (aw, union)
  2    IWU_EVICTED     seed    within the match radius of an IWU-evicted map point
  3    MF_MOS          seed    raw MF-MOS moving mask (mos, union)
  4    SEG_DYNAMIC     seed    the run's final dynamic_mask.npy (any method)
  5    BOX_FILL        —       inside the box of a moving cluster (Chen)
  6    UNMAPPED        attach  no IWU-refined (or chunk static) map point nearby —
                               UniLiPs' "no correspondence in the refined map"

Seed bits start clusters. Attach bits never do: those points only join a
cluster whose box they fall inside. AMBIGUOUS covers vegetation and fences
and UNMAPPED covers every unmapped surface, so letting them seed would swamp
the clusterer and the tracker with structure.

Every source is optional — a bit whose artifact is missing is simply never
set (logged once per chunk). Excluded from every bit: Patchwork++ ground,
near-ego points (cfg.dynamic_min_range_m), and points lower than
motion_proposals.min_height_above_ground_m over Step C's ground grid. BOX_FILL
ignores the height floor, so the wheels and feet it removed come back on
moving clusters.

Motion scoring uses wato_common.tracking (Chen's cost, Hungarian assignment,
constant-velocity Kalman filter). `main` measured that a raw centroid-velocity
gate fakes motion on static structure (per-sweep visibility drifts a
cluster's centroid), so the score is deliberately conservative:
motion_score = BEV distance between the median Kalman-filtered centre of the
first and of the last few frames of the track (net, not path length — jitter
does not accumulate), divided by the largest box side the track ever showed.
Chen's criterion is motion_score > 1. BOX_FILL additionally requires the
cluster's points not to dwell (frac_persistent < 0.5), which catches the
sliding-window wall fragments size-normalised displacement alone lets through.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from wato_common.artifact_store import (
    dynamic_mask_path,
    global_iwu_path,
    lidar_proc_index_path,
    lidar_proc_summary_path,
    local_path,
    motion_clusters_path,
    motion_proposals_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import (
    CHUNK_SUMMARY_SCHEMA,
    MOTION_CLUSTER_SCHEMA,
    Box3D,
    ChunkSummaryRow,
    MotionClusterRow,
    encode_int_list,
)
from wato_common.tracking import MultiObjectTracker, fit_bev_box, points_in_box
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.iwu import load_global_iwu
from wato_lidar_preprocessing.mf_mos.segment import (
    _load_world,
    load_mf_mos_world_mask,
    near_ego_mask,
)
from wato_lidar_preprocessing.union.motion_filter import persistence_keep
from wato_lidar_preprocessing.union.segment import (
    _height_above_ground,
    _load_ground_grid,
)
from wato_lidar_preprocessing.voxel import keys_in_sorted, voxel_indices

log = logging.getLogger(__name__)

AW_DYNAMIC = np.uint8(1 << 0)
AW_AMBIGUOUS = np.uint8(1 << 1)
IWU_EVICTED = np.uint8(1 << 2)
MF_MOS = np.uint8(1 << 3)
SEG_DYNAMIC = np.uint8(1 << 4)
BOX_FILL = np.uint8(1 << 5)
UNMAPPED = np.uint8(1 << 6)

SEED_BITS = np.uint8(AW_DYNAMIC | IWU_EVICTED | MF_MOS | SEG_DYNAMIC)
ATTACH_BITS = np.uint8(AW_AMBIGUOUS | UNMAPPED)

BIT_NAMES: dict[int, str] = {
    int(AW_DYNAMIC): "AW_DYNAMIC",
    int(AW_AMBIGUOUS): "AW_AMBIGUOUS",
    int(IWU_EVICTED): "IWU_EVICTED",
    int(MF_MOS): "MF_MOS",
    int(SEG_DYNAMIC): "SEG_DYNAMIC",
    int(BOX_FILL): "BOX_FILL",
    int(UNMAPPED): "UNMAPPED",
}

# Chen §III-D: an instance is moving if its trajectory exceeds its max side.
MOVING_SCORE = 1.0
# Box fill needs a track of at least this many frames. A 1–2 frame "track"
# has no motion evidence worth painting whole boxes with.
BOX_FILL_MIN_TRACK_LIFE = 3
# ...and a cluster whose points mostly do NOT dwell in the same voxels
# (frac_persistent, main's measured persistence statistic). Displacement over
# size alone is fooled by a thin wall fragment whose visible window slides
# further than the fragment is long (seen on nuScenes scene-0061: a
# 1.6×0.1 m fragment drifting 3 m, frac_persistent 1.0); the clear movers
# there sat at ≤ 0.11. Slow movers that linger in their voxels (a
# pedestrian-sized track at 0.52) lose the fill — the row keeps its score
# either way; only the painting is gated.
BOX_FILL_MAX_PERSISTENT = 0.5
# Frames medianed at each end of a track for its net displacement. Enough to
# reject one bad fit at either end; small enough for short tracks.
_ENDPOINT_FRAMES = 3
# Attach-only points join a cluster within this margin of its seed box —
# about one classify voxel, so edge returns straddling the box still join.
_ATTACH_MARGIN_M = 0.25
# BOX_FILL paints points within this margin of the (grown) box: ~3σ of the
# scanners' 2–3 cm range noise. Seeds on the fitted hull sit exactly on the
# box boundary, and their neighbours' returns scatter a few cm either side.
_FILL_MARGIN_M = 0.1
# Seeds are deduplicated on this grid before HDBSCAN (runtime only; well
# under any object's point spacing).
_CLUSTER_GRID_M = 0.1
# Maps are cropped to the chunk's bbox grown by this much before KD-trees
# are built — only needs to cover the match radius, the rest is slack.
_MAP_CROP_MARGIN_M = 5.0


@dataclass
class ProposalResult:
    chunk_id: str
    n_points_proposal: int
    n_clusters: int
    n_clusters_moving: int
    missing_sources: list[str] = field(default_factory=list)


@dataclass
class _Sources:
    """Chunk-level inputs every sweep's bits are computed against."""

    voxel_origin: np.ndarray | None = None
    voxel_size: float = 0.0
    dynamic_keys: np.ndarray | None = None
    ambiguous_keys: np.ndarray | None = None
    ground_grid: tuple | None = None
    evicted_tree: cKDTree | None = None
    mapped_tree: cKDTree | None = None
    missing: list[str] = field(default_factory=list)


@dataclass
class _Cluster:
    cluster_id: int
    stream: str
    frame_id: int
    t_ns: int
    sweep_ids: list[int]
    box: Box3D
    n_seed: int
    n_points: int
    source_bits: int
    n_seg_dynamic: int
    track_key: tuple[str, int] | None = None
    track_life: int = 1
    net_displacement_m: float = 0.0
    motion_score: float = 0.0
    frac_persistent: float = 0.0
    box_filled: bool = False


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def _chunk_bbox(meta_rows: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    lo, hi = [], []
    for r in meta_rows:
        if r.get("valid") is False or r.get("world_xmin") is None:
            continue
        lo.append([r["world_xmin"], r["world_ymin"], r["world_zmin"]])
        hi.append([r["world_xmax"], r["world_ymax"], r["world_zmax"]])
    if not lo:
        return None
    return np.min(lo, axis=0), np.max(hi, axis=0)


def _crop_tree(xyz: np.ndarray, bbox) -> cKDTree | None:
    if xyz.shape[0] == 0:
        return None
    if bbox is not None:
        lo, hi = bbox[0] - _MAP_CROP_MARGIN_M, bbox[1] + _MAP_CROP_MARGIN_M
        xyz = xyz[np.all((xyz >= lo) & (xyz <= hi), axis=1)]
    return cKDTree(xyz) if xyz.shape[0] else None


def _load_sources(
    cfg: ComponentConfig, bag_id: str, chunk_id: str, meta_rows: list[dict]
) -> _Sources:
    src = _Sources()
    bbox = _chunk_bbox(meta_rows)

    sm_path = local_path(static_map_path(bag_id, chunk_id))
    chunk_static_xyz = np.empty((0, 3))
    if os.path.exists(sm_path):
        sm = np.load(sm_path)
        src.voxel_origin = np.asarray(sm["origin"], dtype=np.float64)
        src.voxel_size = float(sm["voxel_size"])
        chunk_static_xyz = np.asarray(sm["xyz"], dtype=np.float64)
        if "dynamic_voxel_keys" in sm.files:
            src.dynamic_keys = np.sort(np.asarray(sm["dynamic_voxel_keys"], np.int64))
        if "ambiguous_voxel_keys" in sm.files:
            src.ambiguous_keys = np.sort(
                np.asarray(sm["ambiguous_voxel_keys"], np.int64)
            )
    if src.dynamic_keys is None:
        src.missing.append("AW_DYNAMIC (no dynamic_voxel_keys — seg=mos?)")
    if src.ambiguous_keys is None:
        src.missing.append(
            "AW_AMBIGUOUS (no ambiguous_voxel_keys — seg=mos or stale classify)"
        )

    iwu_map = load_global_iwu(bag_id)
    if iwu_map is not None:
        src.evicted_tree = _crop_tree(iwu_map.evicted_xyz, bbox)
        src.mapped_tree = _crop_tree(iwu_map.kept_xyz, bbox)
    else:
        src.missing.append("IWU_EVICTED (no global_iwu.npz — run `iwu`)")
        # UNMAPPED falls back to the chunk's own static cloud.
        src.mapped_tree = _crop_tree(chunk_static_xyz, None)
    if src.mapped_tree is None:
        src.missing.append("UNMAPPED (no static map to compare against)")

    if cfg.motion_proposals.min_height_above_ground_m > 0.0:
        src.ground_grid = _load_ground_grid(bag_id, chunk_id)
    return src


def _load_seg_mask(bag_id: str, chunk_id: str, sweep_id: int, n: int):
    path = local_path(dynamic_mask_path(bag_id, chunk_id, sweep_id))
    if not os.path.exists(path):
        return None
    m = np.load(path).astype(bool)
    return m if m.shape[0] == n else None


# --------------------------------------------------------------------------- #
# Per-sweep bits
# --------------------------------------------------------------------------- #


def compute_sweep_bits(
    xyz: np.ndarray,
    ground_mask: np.ndarray | None,
    sweep_origin: np.ndarray | None,
    src: _Sources,
    cfg: ComponentConfig,
    *,
    mf_mask: np.ndarray | None = None,
    seg_mask: np.ndarray | None = None,
) -> np.ndarray:
    """source_bits for one sweep (BOX_FILL is applied later, per cluster)."""
    n = xyz.shape[0]
    bits = np.zeros(n, dtype=np.uint8)
    if n == 0:
        return bits

    live = _fill_eligible(xyz, ground_mask, sweep_origin, cfg)
    if src.ground_grid is not None and live.any():
        idx = np.flatnonzero(live)
        low = (
            _height_above_ground(xyz[idx], src.ground_grid)
            < cfg.motion_proposals.min_height_above_ground_m
        )
        live[idx[low]] = False
    idx = np.flatnonzero(live)
    if idx.size == 0:
        return bits
    p = xyz[idx]
    b = np.zeros(idx.size, dtype=np.uint8)

    if src.voxel_origin is not None and (
        src.dynamic_keys is not None or src.ambiguous_keys is not None
    ):
        keys = voxel_indices(p, src.voxel_origin, src.voxel_size)
        if src.dynamic_keys is not None:
            b[keys_in_sorted(keys, src.dynamic_keys)] |= AW_DYNAMIC
        if src.ambiguous_keys is not None:
            b[keys_in_sorted(keys, src.ambiguous_keys)] |= AW_AMBIGUOUS

    radius = cfg.global_map_voxel_size_m
    if src.evicted_tree is not None:
        d, _ = src.evicted_tree.query(p, k=1, distance_upper_bound=radius, workers=-1)
        b[np.isfinite(d)] |= IWU_EVICTED
    if src.mapped_tree is not None:
        d, _ = src.mapped_tree.query(p, k=1, distance_upper_bound=radius, workers=-1)
        b[~np.isfinite(d)] |= UNMAPPED
    if mf_mask is not None:
        b[mf_mask[idx]] |= MF_MOS
    if seg_mask is not None:
        b[seg_mask[idx]] |= SEG_DYNAMIC
    bits[idx] = b
    return bits


def _fill_eligible(xyz, ground_mask, sweep_origin, cfg: ComponentConfig) -> np.ndarray:
    """Not Patchwork++ ground and not near-ego — the exclusions BOX_FILL keeps."""
    ok = np.ones(xyz.shape[0], dtype=bool)
    if ground_mask is not None:
        ok &= ~ground_mask.astype(bool)
    if cfg.dynamic_min_range_m > 0.0 and sweep_origin is not None:
        ok &= ~near_ego_mask(xyz, sweep_origin, cfg.dynamic_min_range_m)
    return ok


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


def cluster_points(xyz: np.ndarray, min_cluster_pts: int) -> np.ndarray:
    """HDBSCAN labels per point (−1 = noise), on a deduplicated seed grid."""
    n = xyz.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n < min_cluster_pts:
        return labels
    q = np.floor(xyz / _CLUSTER_GRID_M).astype(np.int64)
    _, first, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)
    reps = xyz[first]
    if reps.shape[0] < min_cluster_pts:
        return labels
    from sklearn.cluster import HDBSCAN  # noqa: PLC0415 — heavy import, Step F only

    rep_labels = HDBSCAN(min_cluster_size=min_cluster_pts, copy=True).fit(reps).labels_
    return rep_labels[inverse.reshape(-1)].astype(np.int64)


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #


@dataclass
class _PointSet:
    """A subset of one frame's points, with where each came from."""

    xyz: np.ndarray  # (N, 3)
    sid: np.ndarray  # (N,) sweep_id
    pidx: np.ndarray  # (N,) index into that sweep's world NPZ
    bits: np.ndarray  # (N,) source_bits

    @classmethod
    def take(cls, sweep_id: int, idx: np.ndarray, xyz, bits) -> "_PointSet":
        return cls(xyz[idx], np.full(idx.size, sweep_id, np.int64), idx, bits[idx])

    @classmethod
    def concat(cls, parts: list["_PointSet"]) -> "_PointSet":
        if not parts:
            return cls(
                np.empty((0, 3)),
                np.empty(0, np.int64),
                np.empty(0, np.int64),
                np.empty(0, np.uint8),
            )
        return cls(
            np.concatenate([p.xyz for p in parts]),
            np.concatenate([p.sid for p in parts]),
            np.concatenate([p.pidx for p in parts]),
            np.concatenate([p.bits for p in parts]),
        )


@dataclass
class _Frame:
    stream: str
    frame_id: int
    t_ns: int
    rows: list[dict]


def _frames(meta_rows: list[dict], cfg: ComponentConfig) -> list[_Frame]:
    """Group valid sweeps into frames, in time order.

    With a canonical lidar every sweep sharing a frame_id is one frame (the
    WATO corners join lidar_cc's tick) and one tracker covers them all.
    Without one, frame_ids are per-lidar ordinals that collide across lidars,
    so each sweep is its own frame and each lidar gets its own tracker.
    """
    valid = [r for r in meta_rows if r.get("valid") is not False]
    frames: list[_Frame] = []
    canonical = cfg.frame_sync.canonical_lidar
    if canonical is not None:
        groups: dict[tuple, list[dict]] = {}
        for r in valid:
            fid = r.get("frame_id")
            key = ("f", int(fid)) if fid is not None else ("s", int(r["sweep_id"]))
            groups.setdefault(key, []).append(r)
        for key, rows in groups.items():
            canon = [r for r in rows if r["lidar_id"] == canonical]
            t = int((canon or rows)[0]["reference_timestamp_ns"])
            fid = key[1] if key[0] == "f" else -1
            frames.append(_Frame("all", fid, t, rows))
    else:
        for r in valid:
            fid = r.get("frame_id")
            frames.append(
                _Frame(
                    str(r["lidar_id"]),
                    int(fid) if fid is not None else int(r["sweep_id"]),
                    int(r["reference_timestamp_ns"]),
                    [r],
                )
            )
    frames.sort(key=lambda f: (f.t_ns, f.stream))
    return frames


# --------------------------------------------------------------------------- #
# Motion features
# --------------------------------------------------------------------------- #


def track_motion(filtered_xy: np.ndarray, max_side_m: float) -> tuple[float, float]:
    """(net displacement [m], motion_score) of one track.

    filtered_xy: (T, 2) Kalman-filtered BEV centres in frame order.
    """
    t = filtered_xy.shape[0]
    if t < 2:
        return 0.0, 0.0
    k = min(_ENDPOINT_FRAMES, max(1, t // 2))
    a = np.median(filtered_xy[:k], axis=0)
    b = np.median(filtered_xy[-k:], axis=0)
    net = float(np.linalg.norm(b - a))
    return net, net / max(max_side_m, 1e-6)


def _score_tracks(
    trackers: dict[str, MultiObjectTracker],
    step_clusters: dict[str, list[list[int]]],
    clusters: list[_Cluster],
) -> None:
    next_hint = 0
    for stream, trk in trackers.items():
        for track in trk.all_tracks():
            cids = [
                step_clusters[stream][o.frame_idx][o.det_idx] for o in track.history
            ]
            xy = np.array([[o.filtered.cx, o.filtered.cy] for o in track.history])
            max_side = max(
                max(o.measured.l, o.measured.w, o.measured.h) for o in track.history
            )
            net, score = track_motion(xy, max_side)
            for cid in cids:
                c = clusters[cid]
                c.track_key = (stream, next_hint)
                c.track_life = len(track.history)
                c.net_displacement_m = net
                c.motion_score = score
            next_hint += 1


def should_box_fill(
    motion_score: float, track_life: int, frac_persistent: float
) -> bool:
    """Chen's criterion (motion_score > 1) on a real track whose points
    don't dwell — the only clusters BOX_FILL paints."""
    return (
        motion_score > MOVING_SCORE
        and track_life >= BOX_FILL_MIN_TRACK_LIFE
        and frac_persistent < BOX_FILL_MAX_PERSISTENT
    )


def fill_box(box: Box3D, grow_down_m: float) -> Box3D:
    """The region BOX_FILL paints: the seed box extended down by the height
    floor. Seeds all sit above min_height_above_ground_m, so the seed box
    stops there — growing it by exactly that slice reaches back to the ground
    and recovers the wheels/feet the floor removed. Patchwork++ ground points
    stay excluded, so the road under a mover is never filled."""
    return Box3D(
        cx=box.cx,
        cy=box.cy,
        cz=box.cz - grow_down_m / 2.0,
        w=box.w,
        l=box.l,
        h=box.h + grow_down_m,
        heading=box.heading,
    )


# --------------------------------------------------------------------------- #
# Chunk
# --------------------------------------------------------------------------- #


def _popcount(v: int) -> int:
    return bin(int(v)).count("1")


def proposals_up_to_date(bag_id: str, chunk_id: str) -> bool:
    """motion_clusters.parquet exists and is newer than everything it read."""
    out = local_path(motion_clusters_path(bag_id, chunk_id))
    if not os.path.exists(out):
        return False
    t_out = os.path.getmtime(out)
    for uri in (
        lidar_proc_summary_path(bag_id, chunk_id),
        lidar_proc_index_path(bag_id, chunk_id),
        static_map_path(bag_id, chunk_id),
        global_iwu_path(bag_id),
    ):
        p = local_path(uri)
        if os.path.exists(p) and os.path.getmtime(p) > t_out:
            return False
    return True


def process_chunk(cfg: ComponentConfig, bag_id: str, chunk_id: str) -> ProposalResult:
    """Write per-sweep motion_proposals.npz + motion_clusters.parquet."""
    mp = cfg.motion_proposals
    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    src = _load_sources(cfg, bag_id, chunk_id, meta_rows)
    for what in src.missing:
        log.info("chunk %s: proposal source unavailable: %s", chunk_id, what)

    frames = _frames(meta_rows, cfg)
    t0 = frames[0].t_ns if frames else 0
    trackers: dict[str, MultiObjectTracker] = {}
    step_clusters: dict[str, list[list[int]]] = {}
    clusters: list[_Cluster] = []
    sweep_bits: dict[int, np.ndarray] = {}
    sweep_assign: dict[int, list[tuple[np.ndarray, int]]] = {}
    member_xyz: list[np.ndarray] = []
    member_sweep: list[np.ndarray] = []
    member_cid: list[np.ndarray] = []

    # --- Pass 1: bits, clusters, tracking ---------------------------------
    for fr in tqdm(frames, desc=f"proposals chunk {chunk_id}", unit="frame"):
        seed_parts, attach_parts = [], []
        for row in fr.rows:
            sid = int(row["sweep_id"])
            xyz, _, ground, origin = _load_world(row["world_path"])
            n = xyz.shape[0]
            mf = load_mf_mos_world_mask(
                bag_id, chunk_id, row, n, cfg.filter_nonfinite_points
            )
            seg = _load_seg_mask(bag_id, chunk_id, sid, n)
            bits = compute_sweep_bits(
                xyz, ground, origin, src, cfg, mf_mask=mf, seg_mask=seg
            )
            sweep_bits[sid] = bits
            is_seed = (bits & SEED_BITS) != 0
            is_attach = ((bits & ATTACH_BITS) != 0) & ~is_seed
            seed_parts.append(_PointSet.take(sid, np.flatnonzero(is_seed), xyz, bits))
            attach_parts.append(
                _PointSet.take(sid, np.flatnonzero(is_attach), xyz, bits)
            )

        seeds = _PointSet.concat(seed_parts)
        att = _PointSet.concat(attach_parts)
        seed_xyz, seed_sid, seed_pidx, seed_bits = (
            seeds.xyz,
            seeds.sid,
            seeds.pidx,
            seeds.bits,
        )
        att_xyz, att_sid, att_pidx, att_bits = att.xyz, att.sid, att.pidx, att.bits
        labels = cluster_points(seed_xyz, mp.min_cluster_pts)
        att_free = np.ones(att_xyz.shape[0], dtype=bool)

        frame_boxes: list[Box3D] = []
        frame_cids: list[int] = []
        for lab in np.unique(labels[labels >= 0]):
            m = labels == lab
            box = fit_bev_box(seed_xyz[m])
            if max(box.l, box.w) > mp.max_side_m:
                continue  # structure (Chen T_size) — points keep bits, no cluster
            cid = len(clusters)
            join = np.zeros(att_xyz.shape[0], dtype=bool)
            if att_free.any():
                join[att_free] = points_in_box(att_xyz[att_free], box, _ATTACH_MARGIN_M)
                att_free &= ~join
            mem_sid = np.concatenate([seed_sid[m], att_sid[join]])
            mem_pidx = np.concatenate([seed_pidx[m], att_pidx[join]])
            mem_bits = np.concatenate([seed_bits[m], att_bits[join]])
            for sid in np.unique(mem_sid):
                sel = mem_sid == sid
                sweep_assign.setdefault(int(sid), []).append((mem_pidx[sel], cid))
            member_xyz.append(np.concatenate([seed_xyz[m], att_xyz[join]]))
            member_sweep.append(mem_sid)
            member_cid.append(np.full(mem_sid.size, cid, dtype=np.int64))
            clusters.append(
                _Cluster(
                    cluster_id=cid,
                    stream=fr.stream,
                    frame_id=fr.frame_id,
                    t_ns=fr.t_ns,
                    sweep_ids=sorted(int(r["sweep_id"]) for r in fr.rows),
                    box=box,
                    n_seed=int(m.sum()),
                    n_points=int(mem_sid.size),
                    source_bits=int(np.bitwise_or.reduce(mem_bits)),
                    n_seg_dynamic=int(((mem_bits & SEG_DYNAMIC) != 0).sum()),
                )
            )
            frame_boxes.append(box)
            frame_cids.append(cid)

        trk = trackers.setdefault(fr.stream, MultiObjectTracker())
        trk.step((fr.t_ns - t0) * 1e-9, frame_boxes)
        step_clusters.setdefault(fr.stream, []).append(frame_cids)

    _score_tracks(trackers, step_clusters, clusters)

    # frac_persistent: members in voxels occupied across many sweeps — the
    # same persistence statistic union's motion filter gates on, as a feature.
    mf_params = cfg.union.motion_filter
    if member_xyz and mf_params.persistence_max_sweeps > 0:
        all_xyz = np.concatenate(member_xyz)
        all_sid = np.concatenate(member_sweep)
        all_cid = np.concatenate(member_cid)
        persistent = ~persistence_keep(
            all_xyz,
            all_sid,
            mf_params.persistence_voxel_m,
            mf_params.persistence_max_sweeps,
        )
        n_per = np.bincount(all_cid, minlength=len(clusters))
        n_pers = np.bincount(all_cid[persistent], minlength=len(clusters))
        for c in clusters:
            c.frac_persistent = float(
                n_pers[c.cluster_id] / max(n_per[c.cluster_id], 1)
            )

    # --- Pass 2: box fill + write ----------------------------------------
    fill_by_sweep: dict[int, list[_Cluster]] = {}
    if mp.box_fill:
        for c in clusters:
            if should_box_fill(c.motion_score, c.track_life, c.frac_persistent):
                c.box_filled = True
                for sid in c.sweep_ids:
                    fill_by_sweep.setdefault(sid, []).append(c)

    n_points_proposal = 0
    for row in meta_rows:
        sid = int(row["sweep_id"])
        out = local_path(motion_proposals_path(bag_id, chunk_id, sid))
        if sid not in sweep_bits:
            if os.path.exists(out):  # stale from a run where this sweep was valid
                os.remove(out)
            continue
        bits = sweep_bits[sid]
        cluster_id = np.full(bits.shape[0], -1, dtype=np.int32)
        for pidx, cid in sweep_assign.get(sid, []):
            cluster_id[pidx] = cid
        if sid in fill_by_sweep:
            xyz, _, ground, origin = _load_world(row["world_path"])
            eligible = _fill_eligible(xyz, ground, origin, cfg)
            for c in fill_by_sweep[sid]:
                box = fill_box(c.box, mp.min_height_above_ground_m)
                inside = points_in_box(xyz, box, _FILL_MARGIN_M) & eligible
                bits[inside] |= BOX_FILL
                cluster_id[inside & (cluster_id < 0)] = c.cluster_id
        n_points_proposal += int((bits != 0).sum())
        np.savez_compressed(out, source_bits=bits, cluster_id=cluster_id)

    n_moving = sum(1 for c in clusters if c.motion_score > MOVING_SCORE)
    _update_summary(bag_id, chunk_id, n_points_proposal, len(clusters), n_moving)
    _write_clusters(bag_id, chunk_id, clusters)
    log.info(
        "chunk %s: proposals %d pts, %d clusters (%d moving by Chen's criterion, "
        "%d box-filled)",
        chunk_id,
        n_points_proposal,
        len(clusters),
        n_moving,
        sum(1 for c in clusters if c.box_filled),
    )
    return ProposalResult(
        chunk_id, n_points_proposal, len(clusters), n_moving, list(src.missing)
    )


def _update_summary(bag_id, chunk_id, n_points, n_clusters, n_moving) -> None:
    uri = lidar_proc_summary_path(bag_id, chunk_id)
    if not os.path.exists(local_path(uri)):
        log.warning(
            "chunk %s: no lidar_proc_summary.parquet to record proposals in", chunk_id
        )
        return
    rows = read_rows(uri)
    if not rows:
        return
    row = dict(rows[0])
    row.update(
        n_points_proposal=n_points, n_clusters=n_clusters, n_clusters_moving=n_moving
    )
    write_table([ChunkSummaryRow(**row).model_dump()], CHUNK_SUMMARY_SCHEMA, uri)


def _write_clusters(bag_id: str, chunk_id: str, clusters: list[_Cluster]) -> None:
    rows = []
    for c in clusters:
        rows.append(
            MotionClusterRow(
                bag_id=bag_id,
                chunk_id=chunk_id,
                frame_id=c.frame_id,
                reference_timestamp_ns=c.t_ns,
                sweep_ids=encode_int_list(c.sweep_ids),
                cluster_id=c.cluster_id,
                track_hint_id=c.track_key[1] if c.track_key else -1,
                cx=c.box.cx,
                cy=c.box.cy,
                cz=c.box.cz,
                w=c.box.w,
                l=c.box.l,
                h=c.box.h,
                heading=c.box.heading,
                n_points=c.n_points,
                n_seed_points=c.n_seed,
                source_bits=c.source_bits,
                n_sources=_popcount(c.source_bits & ~int(BOX_FILL)),
                frac_seg_dynamic=c.n_seg_dynamic / max(c.n_points, 1),
                frac_persistent=c.frac_persistent,
                track_life=c.track_life,
                net_displacement_m=c.net_displacement_m,
                motion_score=c.motion_score,
                box_filled=c.box_filled,
            ).model_dump()
        )
    write_table(rows, MOTION_CLUSTER_SCHEMA, motion_clusters_path(bag_id, chunk_id))


def decode_bits(bits: np.ndarray) -> dict[str, np.ndarray]:
    """{bit name: bool mask} — convenience for scripts and viz."""
    return {name: (bits & np.uint8(v)) != 0 for v, name in BIT_NAMES.items()}


__all__ = [
    "ATTACH_BITS",
    "AW_AMBIGUOUS",
    "AW_DYNAMIC",
    "BIT_NAMES",
    "BOX_FILL",
    "BOX_FILL_MAX_PERSISTENT",
    "BOX_FILL_MIN_TRACK_LIFE",
    "IWU_EVICTED",
    "MF_MOS",
    "MOVING_SCORE",
    "ProposalResult",
    "SEED_BITS",
    "SEG_DYNAMIC",
    "UNMAPPED",
    "cluster_points",
    "compute_sweep_bits",
    "decode_bits",
    "fill_box",
    "process_chunk",
    "proposals_up_to_date",
    "should_box_fill",
    "track_motion",
]
