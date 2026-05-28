"""Bucket near-ego static→dynamic leakage points by voxel classification.

Usage (inside lidar_preprocessing_dev container):
    python -m wato_lidar_preprocessing.scripts.debug_near_ego_leakage \
        <bag_id> <chunk_id> [--radius-m 15] [--sample-sweeps 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wato_lidar_preprocessing.voxel import voxel_indices

CLASS_NAMES = {
    0: "STATIC (flipped by mfmos?)",
    1: "AMBIGUOUS",
    2: "UNDER_EVIDENCED",
    3: "FREE_ONLY",
    4: "DYNAMIC (carved)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_id")
    ap.add_argument("chunk_id")
    ap.add_argument("--radius-m", type=float, default=15.0)
    ap.add_argument("--sample-sweeps", type=int, default=20)
    ap.add_argument(
        "--root",
        default="/data/artifacts/raw",
        help="artifact root inside the container",
    )
    args = ap.parse_args()

    chunk_dir = Path(args.root) / args.bag_id / "chunks" / args.chunk_id
    proc_dir = chunk_dir / "lidar_proc"
    diag = np.load(chunk_dir / "voxel_diag.npz")
    keys_diag = diag["keys"]
    cls = diag["classification"]
    p_occ = diag["p_occ"]
    n_obs = diag["n_obs"]
    diag_origin = diag["origin"]
    voxel_size = float(diag["voxel_size"])

    print(f"voxel_diag: {keys_diag.size} voxels, voxel_size={voxel_size:.3f}m")
    print("  per-class totals (whole chunk):")
    for code, name in CLASS_NAMES.items():
        print(f"    {name:35s} {int((cls == code).sum()):>10d}")
    print()

    sweep_files = sorted(proc_dir.glob("*_world.npz"))
    step = max(1, len(sweep_files) // args.sample_sweeps)
    sampled = sweep_files[::step][: args.sample_sweeps]
    print(f"sampling {len(sampled)} sweeps from {len(sweep_files)} total")
    print(f"near-ego radius: {args.radius_m:.1f}m\n")

    bucket_totals = {code: 0 for code in CLASS_NAMES}
    fallthrough_total = 0
    suspect_total = 0
    p_occ_static_flips: list[float] = []
    n_obs_dynamic: list[int] = []

    for wpath in sampled:
        sweep_id = int(wpath.stem.split("_")[0])
        mask_path = proc_dir / f"{sweep_id:06d}_dynamic_mask.npy"
        if not mask_path.exists():
            continue
        w = np.load(wpath)
        xyz = np.stack([w["x"], w["y"], w["z"]], axis=1)
        origin = w["origin"]
        dyn_mask = np.load(mask_path)
        if dyn_mask.shape[0] != xyz.shape[0]:
            print(f"  sweep {sweep_id}: mask/xyz length mismatch, skipping")
            continue

        # near-ego dynamic points
        d = np.linalg.norm(xyz[:, :2] - origin[:2], axis=1)
        near = d < args.radius_m
        suspects_mask = near & dyn_mask
        suspect_xyz = xyz[suspects_mask]
        if suspect_xyz.shape[0] == 0:
            continue
        suspect_total += suspect_xyz.shape[0]

        suspect_keys = voxel_indices(suspect_xyz, diag_origin, voxel_size)
        pos = np.searchsorted(keys_diag, suspect_keys)
        pos = np.clip(pos, 0, keys_diag.size - 1)
        found = keys_diag[pos] == suspect_keys
        fallthrough_total += int((~found).sum())

        found_pos = pos[found]
        found_cls = cls[found_pos]
        for code in CLASS_NAMES:
            bucket_totals[code] += int((found_cls == code).sum())

        # collect distribution detail for the two interesting buckets
        is_static_flip = found_cls == 0
        if is_static_flip.any():
            p_occ_static_flips.extend(p_occ[found_pos[is_static_flip]].tolist())
        is_dynamic = found_cls == 4
        if is_dynamic.any():
            n_obs_dynamic.extend(n_obs[found_pos[is_dynamic]].tolist())

    print("== near-ego leakage breakdown ==")
    print(
        f"  total suspect points (dynamic & within {args.radius_m:.0f}m): {suspect_total}"
    )
    if suspect_total == 0:
        return
    for code, name in CLASS_NAMES.items():
        n = bucket_totals[code]
        pct = 100.0 * n / suspect_total
        print(f"    {name:35s} {n:>8d}  ({pct:5.1f}%)")
    pct_ft = 100.0 * fallthrough_total / suspect_total
    print(
        f"    {'NOT-IN-KEYS (fall-through)':35s} {fallthrough_total:>8d}  ({pct_ft:5.1f}%)"
    )
    print()

    if p_occ_static_flips:
        arr = np.asarray(p_occ_static_flips)
        print(
            f"  STATIC-bucket flips ({len(arr)} pts): p_occ "
            f"min={arr.min():.3f} median={np.median(arr):.3f} max={arr.max():.3f}"
        )
        print(
            "    -> these are voxels AW called static but MF-MOS / fusion flipped to dynamic"
        )
    if n_obs_dynamic:
        arr = np.asarray(n_obs_dynamic, dtype=np.float64)
        print(
            f"  DYNAMIC-bucket pts  ({len(arr):d}): n_obs "
            f"min={int(arr.min())} median={int(np.median(arr))} max={int(arr.max())}"
        )
        print("    -> if median n_obs is low, raising min_observations may help")


if __name__ == "__main__":
    main()
