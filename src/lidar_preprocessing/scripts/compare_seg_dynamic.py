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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
    args = ap.parse_args()

    chunk_dir = Path(args.root) / args.bag_id / "chunks" / args.chunk_id
    dyn = np.load(chunk_dir / "dynamic_map.npz")["xyz"]
    stat = np.load(chunk_dir / "static_map.npz")["xyz"]

    print(f"chunk {args.bag_id}/{args.chunk_id}")
    print(f"  dynamic points: {dyn.shape[0]:,}   static points: {stat.shape[0]:,}")
    if dyn.shape[0] == 0 or stat.shape[0] == 0:
        print("  (need both clouds non-empty to score)")
        return

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(stat)
    except ImportError:
        tree = None
        print("  (scipy unavailable — using slow brute-force coincidence)")

    on = 100 * _coincide_fraction(dyn, tree, stat, args.radius_m)
    shifted = dyn + np.array([args.shift_m, args.shift_m, 0.0])
    floor = 100 * _coincide_fraction(shifted, tree, stat, args.radius_m)

    print(f"  z-profile: p5={np.percentile(dyn[:, 2], 5):.2f}  "
          f"median={np.median(dyn[:, 2]):.2f}  p95={np.percentile(dyn[:, 2], 95):.2f} m")
    print(f"  on-static (<{args.radius_m:.2f} m of a static surface): {on:5.1f}%")
    print(f"  shift +{args.shift_m:.0f} m null (chance floor):          {floor:5.1f}%")
    print(f"  leakage above chance: {on - floor:+5.1f} points-pct  "
          f"(lower = cleaner; ~0 means movers-only)")


if __name__ == "__main__":
    main()
