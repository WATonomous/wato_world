"""Quantify how much of a chunk's dynamic_map is actually static structure.

A clean dynamic cloud sits on movers, not on static surfaces. This measures
the failure mode behind `--seg union`: for the dynamic_map.npz a segmentation
method produced, what fraction of its points land within one voxel of a
static_map.npz surface ("on-static %"). A shift-null (translate the dynamic
cloud a few metres and re-measure) gives the chance floor from static-map
density, so the gap above the floor is the real static leakage.

Run it after each method to A/B them on the SAME chunk:

    # inside the lidar_preprocessing_dev container
    ./watod run lidar_preprocessing --bag <bag> --chunk 0000 --seg aw
    python -m wato_lidar_preprocessing.scripts.compare_seg_dynamic <bag> 0000
    ./watod run lidar_preprocessing --bag <bag> --chunk 0000 --seg union
    python -m wato_lidar_preprocessing.scripts.compare_seg_dynamic <bag> 0000

Lower on-static % (closer to the shift floor) = cleaner dynamic cloud.
Needs scipy (cKDTree); falls back to a slower brute-force check without it.

`--proposals` scores Step F's recall-oriented motion proposals the same way,
one row per source bit, for the union of all bits, and for the points of
clusters passing Chen's motion criterion (motion_score > 1). Proposals are
false-positive tolerant by design, so expect the union's on-static % to sit
well above the seg dynamic_map's; the per-bit rows show which heuristic the
leakage comes from. When the bag has global_iwu.npz the reference is the
IWU-refined static map (evicted floaters removed): IWU_EVICTED points come
from static_map.npz by construction, so against it they would read ~100%
"on static" whatever their quality.

    python -m wato_lidar_preprocessing.scripts.compare_seg_dynamic <bag> 0000 --proposals
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Mirrors wato_lidar_preprocessing.motion_proposals.BIT_NAMES; duplicated so
# this script stays runnable without the package on PYTHONPATH.
_BITS = {
    1: "AW_DYNAMIC",
    2: "AW_AMBIGUOUS",
    4: "IWU_EVICTED",
    8: "MF_MOS",
    16: "SEG_DYNAMIC",
    32: "BOX_FILL",
    64: "UNMAPPED",
}
_CAP = 2_000_000  # points scored per group (uniform subsample above this)


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
    args = ap.parse_args()

    chunk_dir = Path(args.root) / args.bag_id / "chunks" / args.chunk_id
    dyn = np.load(chunk_dir / "dynamic_map.npz")["xyz"]
    stat = np.load(chunk_dir / "static_map.npz")["xyz"]

    print(f"chunk {args.bag_id}/{args.chunk_id}")
    print(f"  dynamic points: {dyn.shape[0]:,}   static points: {stat.shape[0]:,}")
    # Proposals are scored against the static cloud only; dynamic_map may be
    # empty (e.g. seg=aw on a quiet chunk) without making them unscoreable.
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
