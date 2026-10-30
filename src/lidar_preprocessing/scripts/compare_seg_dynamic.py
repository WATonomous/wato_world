"""Quantify how much of a chunk's dynamic_map is actually static structure.

A clean dynamic cloud sits on movers, not on static surfaces. This measures,
for a chunk's dynamic_map.npz, what fraction of its points land within one
voxel of a static_map.npz surface ("on-static %"). A shift-null (translate the
dynamic cloud a few metres and re-measure) gives the chance floor from
static-map density, so the gap above the floor is the real static leakage.

    # in the lidar_preprocessing_dev container:
    python /ws/src/lidar_preprocessing/scripts/compare_seg_dynamic.py <bag> 0000

Lower on-static % (closer to the shift floor) = cleaner dynamic cloud.

`--proposals` scores Step F's proposals the same way: one row per source bit,
the union of all bits, and members of clusters with motion_score > 1. Expect
the union well above the seg dynamic_map (false positives are by design). With
global_iwu.npz the reference is the IWU-refined map, since IWU_EVICTED points
come from static_map.npz and would read ~100% "on static" against it.

On-static % moves with the static map it is scored against, so it can't
compare two Step B versions. `--save-reference FILE` freezes two point sets
from the current run instead; `--reference FILE` scores a later run of the
same chunk against them (Step B changes, the deskewed sweeps don't):

- movers: points of Step F tracks with motion_score > 2, frac_persistent < 0.3,
  track_life >= 5 and >= 20 points. Share kept = recall; higher is better.
- persistent static: dynamic points whose voxel's dynamic hits span >= 5 s (no
  mover dwells that long in one voxel). Share still dynamic = static leak;
  lower is better.

The Step B measurements in docs/lidar_preprocessing_measurements.md were
scored this way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Mirrors wato_common.artifacts.motion_proposals.BIT_NAMES; duplicated so
# this script stays runnable without the package on PYTHONPATH.
_BITS = {
    1: "AW_DYNAMIC",
    2: "AW_AMBIGUOUS",
    4: "IWU_EVICTED",
    16: "SEG_DYNAMIC",
    32: "BOX_FILL",
    64: "UNMAPPED",
}
_CAP = 2_000_000  # points scored per group (uniform subsample above this)
_PERSIST_NS = 5_000_000_000  # a car clears a 0.25 m voxel in < 1 s


def _coincide_fraction(
    query: np.ndarray, ref_tree, ref_xyz: np.ndarray, radius: float
) -> float:
    """Fraction of `query` points with a `ref` point within `radius` (metres)."""
    if query.shape[0] == 0:
        return float("nan")
    if ref_tree is not None:
        dist, _ = ref_tree.query(query, k=1, distance_upper_bound=radius)
        return float(np.isfinite(dist).mean())
    # Brute-force fallback (no scipy): chunked to bound memory.
    hits = 0
    r2 = radius * radius
    for start in range(0, query.shape[0], 2048):
        block = query[start : start + 2048]
        d2 = ((block[:, None, :] - ref_xyz[None, :, :]) ** 2).sum(axis=2)
        hits += int((d2.min(axis=1) <= r2).sum())
    return hits / query.shape[0]


def _load_proposal_groups(chunk_dir: Path) -> dict[str, np.ndarray]:
    """xyz per proposal group, read from the per-sweep proposal NPZs."""
    moving_ids: set[int] = set()
    clusters = chunk_dir / "motion_clusters.parquet"
    if clusters.exists():
        import pyarrow.parquet as pq

        t = pq.read_table(clusters, columns=["cluster_id", "motion_score"]).to_pydict()
        moving_ids = {
            int(c) for c, m in zip(t["cluster_id"], t["motion_score"]) if m > 1.0
        }

    groups: dict[str, list[np.ndarray]] = {n: [] for n in _BITS.values()}
    groups["ANY (union)"] = []
    groups["moving clusters"] = []
    for prop in sorted((chunk_dir / "lidar_proc").glob("*_motion_proposals.npz")):
        world = prop.with_name(prop.name.replace("_motion_proposals.npz", "_world.npz"))
        if not world.exists():
            continue
        w = np.load(world)
        xyz = np.stack([w["x"], w["y"], w["z"]], axis=1)
        d = np.load(prop)
        bits, cid = d["source_bits"], d["cluster_id"]
        if bits.shape[0] != xyz.shape[0]:
            continue
        for v, name in _BITS.items():
            groups[name].append(xyz[(bits & v) != 0])
        groups["ANY (union)"].append(xyz[bits != 0])
        if moving_ids:
            groups["moving clusters"].append(xyz[np.isin(cid, list(moving_ids))])
    rng = np.random.default_rng(0)
    out: dict[str, np.ndarray] = {}
    for name, parts in groups.items():
        g = np.concatenate(parts) if parts else np.empty((0, 3))
        if g.shape[0] > _CAP:
            g = g[rng.choice(g.shape[0], _CAP, replace=False)]
        out[name] = g
    return out


def _voxel_keys(xyz: np.ndarray, origin: np.ndarray, voxel: float) -> np.ndarray:
    idx = np.floor((xyz - origin) / voxel).astype(np.int64)
    return (idx[:, 0] << 40) | (idx[:, 1] << 20) | idx[:, 2]


def _valid_sweeps(chunk_dir: Path) -> list[tuple[int, int]]:
    """(sweep_id, reference_timestamp_ns) of every valid sweep, index order."""
    import pyarrow.parquet as pq

    t = pq.read_table(
        chunk_dir / "lidar_proc_index.parquet",
        columns=["sweep_id", "reference_timestamp_ns", "valid", "world_path"],
    ).to_pydict()
    return [
        (int(s), int(ts))
        for s, ts, ok, w in zip(
            t["sweep_id"], t["reference_timestamp_ns"], t["valid"], t["world_path"]
        )
        if ok is not False and w
    ]


def _save_reference(chunk_dir: Path, path: Path) -> None:
    """Freeze the mover and persistent-static sets of the current run."""
    import pyarrow.parquet as pq

    sweeps = _valid_sweeps(chunk_dir)
    t_of = dict(sweeps)
    sm = np.load(chunk_dir / "static_map.npz")
    origin, voxel = sm["origin"], float(sm["voxel_size"])
    dm = np.load(chunk_dir / "dynamic_map.npz")
    t = np.array([t_of[int(s)] for s in dm["sweep_id"]], dtype=np.int64)
    uk, inv = np.unique(_voxel_keys(dm["xyz"], origin, voxel), return_inverse=True)
    first = np.full(uk.size, np.iinfo(np.int64).max)
    last = np.full(uk.size, np.iinfo(np.int64).min)
    np.minimum.at(first, inv, t)
    np.maximum.at(last, inv, t)
    persistent = uk[(last - first) >= _PERSIST_NS]

    c = pq.read_table(chunk_dir / "motion_clusters.parquet").to_pydict()
    movers = np.array(
        [
            cid
            for cid, ms, fp, life, n in zip(
                c["cluster_id"],
                c["motion_score"],
                c["frac_persistent"],
                c["track_life"],
                c["n_points"],
            )
            if ms > 2.0 and fp < 0.3 and life >= 5 and n >= 20
        ],
        dtype=np.int64,
    )

    proc = chunk_dir / "lidar_proc"
    movers_m, persistent_m, lens = [], [], []
    for sid, _ in sweeps:
        w = np.load(proc / f"{sid:06d}_world.npz")
        xyz = np.stack([w["x"], w["y"], w["z"]], axis=1)
        dyn = np.load(proc / f"{sid:06d}_dynamic_mask.npy")
        cid = np.load(proc / f"{sid:06d}_motion_proposals.npz")["cluster_id"]
        m = np.isin(cid, movers)
        movers_m.append(m)
        persistent_m.append(
            dyn & np.isin(_voxel_keys(xyz, origin, voxel), persistent) & ~m
        )
        lens.append(dyn.shape[0])
    m_all, p_all = np.concatenate(movers_m), np.concatenate(persistent_m)
    np.savez_compressed(
        path,
        sweep_id=np.array([s for s, _ in sweeps], dtype=np.int64),
        lens=np.array(lens, dtype=np.int64),
        movers=np.packbits(m_all),
        persistent=np.packbits(p_all),
        n_points=m_all.size,
        n_dynamic=dm["xyz"].shape[0],
    )
    print(
        f"  reference -> {path}: {m_all.sum():,} mover pts ({movers.size} clusters), "
        f"{p_all.sum():,} persistent-static pts, {dm['xyz'].shape[0]:,} dynamic pts"
    )


def _score_reference(chunk_dir: Path, path: Path) -> None:
    """Share of the reference sets the current dynamic masks keep."""
    ref = np.load(path)
    n = int(ref["n_points"])
    movers = np.unpackbits(ref["movers"], count=n).astype(bool)
    persistent = np.unpackbits(ref["persistent"], count=n).astype(bool)
    offs = np.concatenate([[0], np.cumsum(ref["lens"])])
    proc = chunk_dir / "lidar_proc"
    n_dyn = kept_m = kept_p = 0
    for j, sid in enumerate(ref["sweep_id"]):
        dyn = np.load(proc / f"{int(sid):06d}_dynamic_mask.npy")
        if dyn.shape[0] != ref["lens"][j]:
            raise SystemExit(f"sweep {int(sid)}: mask length changed; re-run deskew?")
        sl = slice(offs[j], offs[j + 1])
        n_dyn += int(dyn.sum())
        kept_m += int((dyn & movers[sl]).sum())
        kept_p += int((dyn & persistent[sl]).sum())
    print(f"  vs reference {path}:")
    print(
        f"    dynamic points        {n_dyn:>10,}  "
        f"({100 * n_dyn / max(int(ref['n_dynamic']), 1):.1f}% of the reference run)"
    )
    print(
        f"    persistent static     {100 * kept_p / max(persistent.sum(), 1):5.1f}% "
        f"still dynamic (of {persistent.sum():,}; lower = less static leak)"
    )
    print(
        f"    movers                {100 * kept_m / max(movers.sum(), 1):5.1f}% "
        f"kept (of {movers.sum():,}; higher = recall kept)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag_id")
    ap.add_argument("chunk_id")
    ap.add_argument(
        "--root", default="/data/artifacts/raw", help="artifact root in the container"
    )
    ap.add_argument("--radius-m", type=float, default=0.25, help="coincidence radius")
    ap.add_argument(
        "--shift-m", type=float, default=2.0, help="null-test horizontal shift"
    )
    ap.add_argument(
        "--proposals",
        action="store_true",
        help="score Step F motion proposals per source bit instead of dynamic_map",
    )
    ap.add_argument(
        "--save-reference",
        type=Path,
        metavar="FILE",
        help="freeze this run's mover and persistent-static sets to FILE (.npz)",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        metavar="FILE",
        help="score this run's dynamic masks against a --save-reference FILE",
    )
    args = ap.parse_args()

    chunk_dir = Path(args.root) / args.bag_id / "chunks" / args.chunk_id
    if args.save_reference or args.reference:
        print(f"chunk {args.bag_id}/{args.chunk_id}")
        if args.save_reference:
            _save_reference(chunk_dir, args.save_reference)
        if args.reference:
            _score_reference(chunk_dir, args.reference)
        return
    dyn = np.load(chunk_dir / "dynamic_map.npz")["xyz"]
    stat = np.load(chunk_dir / "static_map.npz")["xyz"]

    print(f"chunk {args.bag_id}/{args.chunk_id}")
    print(f"  dynamic points: {dyn.shape[0]:,}   static points: {stat.shape[0]:,}")
    # Proposals are scored against the static cloud only; dynamic_map may be
    # empty (a quiet chunk) without making them unscoreable.
    if stat.shape[0] == 0 or (dyn.shape[0] == 0 and not args.proposals):
        print("  (need both clouds non-empty to score)")
        return

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(stat)
    except ImportError:
        tree = None
        print("  (scipy unavailable — using slow brute-force coincidence)")

    if args.proposals:
        groups = _load_proposal_groups(chunk_dir)
        iwu_path = chunk_dir.parent.parent / "global_iwu.npz"
        if iwu_path.exists():
            iwu = np.load(iwu_path)
            ref = iwu["xyz"][~iwu["evicted"].astype(bool)]
            anyp = groups["ANY (union)"]
            if anyp.shape[0]:
                lo, hi = anyp.min(axis=0) - 5.0, anyp.max(axis=0) + 5.0
                ref = ref[np.all((ref >= lo) & (ref <= hi), axis=1)]
            print(f"  reference: IWU-refined static map ({ref.shape[0]:,} pts)")
            stat = ref
            tree = cKDTree(stat) if tree is not None and stat.shape[0] else None
        shift = np.array([args.shift_m, args.shift_m, 0.0])
        print(
            f"  {'group':<18} {'points':>10} {'on-static':>10} {'floor':>7} {'leak':>7}"
        )
        for name, g in groups.items():
            if g.shape[0] == 0:
                print(f"  {name:<18} {0:>10,}          —       —       —")
                continue
            on_g = 100 * _coincide_fraction(g, tree, stat, args.radius_m)
            fl_g = 100 * _coincide_fraction(g + shift, tree, stat, args.radius_m)
            print(
                f"  {name:<18} {g.shape[0]:>10,} {on_g:>9.1f}% {fl_g:>6.1f}% "
                f"{on_g - fl_g:>+6.1f}"
            )
        return

    on = 100 * _coincide_fraction(dyn, tree, stat, args.radius_m)
    shifted = dyn + np.array([args.shift_m, args.shift_m, 0.0])
    floor = 100 * _coincide_fraction(shifted, tree, stat, args.radius_m)

    print(
        f"  z-profile: p5={np.percentile(dyn[:, 2], 5):.2f}  "
        f"median={np.median(dyn[:, 2]):.2f}  p95={np.percentile(dyn[:, 2], 95):.2f} m"
    )
    print(f"  on-static (<{args.radius_m:.2f} m of a static surface): {on:5.1f}%")
    print(f"  shift +{args.shift_m:.0f} m null (chance floor):          {floor:5.1f}%")
    print(
        f"  leakage above chance: {on - floor:+5.1f} points-pct  "
        f"(lower = cleaner; ~0 means movers-only)"
    )


if __name__ == "__main__":
    main()
