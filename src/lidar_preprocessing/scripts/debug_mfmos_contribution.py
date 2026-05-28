"""Measure MF-MOS's marginal contribution under union/mfmos_only fusion.

For each sweep, partition the points by what made them dynamic:
  - AW alone: point's voxel is AW-DYNAMIC (or absent) AND raw MF-MOS mask = False
  - MF-MOS rescue: AW says not-dynamic, MF-MOS raw mask = True
  - Both agree: AW says dynamic, MF-MOS raw mask = True
  - Neither (sanity): both say static — should be 0 in the dynamic_mask

Then bucket each MF-MOS rescue by what AW thought the voxel was:
  STATIC          -> likely MF-MOS false positive (overruling confident static)
  AMBIGUOUS       -> borderline, MF-MOS could be right
  UNDER_EVIDENCED -> AW couldn't decide; MF-MOS info is genuinely additive
  FREE_ONLY       -> voxel had no hits; MF-MOS flagging it is suspicious
  not-in-keys     -> voxel never traversed; MF-MOS flagging is uninformed

Note: uses the raw per-point _mf_mos_mask.npy as a proxy for "MF-MOS thinks
this voxel is dynamic".  The actual voxel-aggregated MF-MOS verdict
(mf_mos_dynamic_arr) requires min_mf_mos_votes + vote_fraction agreement
across sweeps, so this slightly over-estimates MF-MOS's contribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wato_lidar_preprocessing.voxel import voxel_indices

CLASS_NAMES = {
    0: "STATIC",
    1: "AMBIGUOUS",
    2: "UNDER_EVIDENCED",
    3: "FREE_ONLY",
    4: "DYNAMIC",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_id")
    ap.add_argument("chunk_id")
    ap.add_argument("--sample-sweeps", type=int, default=20)
    ap.add_argument(
        "--radius-m",
        type=float,
        default=None,
        help="if set, restrict to near-ego points within this distance",
    )
    ap.add_argument("--root", default="/data/artifacts/raw")
    args = ap.parse_args()

    chunk_dir = Path(args.root) / args.bag_id / "chunks" / args.chunk_id
    proc_dir = chunk_dir / "lidar_proc"
    diag = np.load(chunk_dir / "voxel_diag.npz")
    keys_diag = diag["keys"]
    cls = diag["classification"]
    diag_origin = diag["origin"]
    voxel_size = float(diag["voxel_size"])

    sweep_files = sorted(proc_dir.glob("*_world.npz"))
    step = max(1, len(sweep_files) // args.sample_sweeps)
    sampled = sweep_files[::step][: args.sample_sweeps]
    print(f"sampling {len(sampled)} sweeps from {len(sweep_files)} total")
    if args.radius_m is not None:
        print(f"restricting to near-ego radius {args.radius_m:.1f}m")
    print()

    aw_only = 0
    mfmos_rescue = 0
    both_agree = 0
    neither_in_dyn = 0  # both agree static, point is NOT in dynamic_mask (sanity check)
    inconsistent = 0  # both say static yet dynamic_mask=True (would indicate a bug)

    # bucket the MF-MOS rescues by AW voxel class
    rescue_by_aw_class = {code: 0 for code in CLASS_NAMES}
    rescue_not_in_keys = 0

    for wpath in sampled:
        sweep_id = int(wpath.stem.split("_")[0])
        mask_path = proc_dir / f"{sweep_id:06d}_dynamic_mask.npy"
        mfmos_path = proc_dir / f"{sweep_id:06d}_mf_mos_mask.npy"
        if not mask_path.exists() or not mfmos_path.exists():
            continue
        w = np.load(wpath)
        xyz = np.stack([w["x"], w["y"], w["z"]], axis=1)
        origin = w["origin"]
        dyn_mask = np.load(mask_path)
        mfmos_mask = np.load(mfmos_path)
        if not (dyn_mask.shape[0] == xyz.shape[0] == mfmos_mask.shape[0]):
            continue

        if args.radius_m is not None:
            d = np.linalg.norm(xyz[:, :2] - origin[:2], axis=1)
            sel = d < args.radius_m
            xyz = xyz[sel]
            dyn_mask = dyn_mask[sel]
            mfmos_mask = mfmos_mask[sel]
            if xyz.shape[0] == 0:
                continue

        # AW per-voxel verdict: voxel.classification == DYNAMIC OR voxel absent
        keys = voxel_indices(xyz, diag_origin, voxel_size)
        pos = np.searchsorted(keys_diag, keys)
        pos = np.clip(pos, 0, keys_diag.size - 1)
        found = keys_diag[pos] == keys
        aw_class = np.where(found, cls[pos], -1)  # -1 = not in keys
        is_aw_dyn = (aw_class == 4) | (aw_class == -1)

        # MF-MOS per-point flag (raw, pre-voxel-aggregation)
        is_mf_dyn = mfmos_mask.astype(bool)

        # Partition the dynamic_mask:
        in_dyn = dyn_mask.astype(bool)
        a = is_aw_dyn & ~is_mf_dyn & in_dyn
        m = ~is_aw_dyn & is_mf_dyn & in_dyn
        b = is_aw_dyn & is_mf_dyn & in_dyn
        n = ~is_aw_dyn & ~is_mf_dyn & in_dyn
        aw_only += int(a.sum())
        mfmos_rescue += int(m.sum())
        both_agree += int(b.sum())
        inconsistent += int(n.sum())
        neither_in_dyn += int((~is_aw_dyn & ~is_mf_dyn & ~in_dyn).sum())

        # bucket MF-MOS rescues by what AW thought
        rescue_pts = m
        if rescue_pts.any():
            rc = aw_class[rescue_pts]
            for code in CLASS_NAMES:
                rescue_by_aw_class[code] += int((rc == code).sum())
            rescue_not_in_keys += int((rc == -1).sum())

    print("== dynamic_mask attribution ==")
    total_dyn = aw_only + mfmos_rescue + both_agree + inconsistent
    print(f"  total points in dynamic_mask: {total_dyn}")
    if total_dyn > 0:

        def f(x: int) -> str:
            return f"{x:>8d}  ({100.0 * x / total_dyn:5.1f}%)"

        print(f"    AW only                  {f(aw_only)}")
        print(f"    MF-MOS rescue (AW=no)    {f(mfmos_rescue)}")
        print(f"    Both AW + MF-MOS         {f(both_agree)}")
        print(f"    Inconsistent (bug?)      {f(inconsistent)}")
    print()
    print("== MF-MOS rescue audit (where AW said the voxel was) ==")
    print("    — STATIC / FREE_ONLY rescues likely MF-MOS false-positives")
    print("    — UNDER_EVIDENCED / AMBIGUOUS rescues likely MF-MOS adding value")
    if mfmos_rescue > 0:
        for code, name in CLASS_NAMES.items():
            n = rescue_by_aw_class[code]
            if n:
                print(f"    {name:18s} {n:>8d}  ({100.0*n/mfmos_rescue:5.1f}%)")
        if rescue_not_in_keys:
            print(
                f"    {'not-in-keys':18s} {rescue_not_in_keys:>8d}  "
                f"({100.0*rescue_not_in_keys/mfmos_rescue:5.1f}%)"
            )

    # bottom line
    print()
    if total_dyn > 0 and mfmos_rescue > 0:
        good = rescue_by_aw_class[1] + rescue_by_aw_class[2]
        bad = rescue_by_aw_class[0] + rescue_by_aw_class[3] + rescue_not_in_keys
        print("== bottom line ==")
        print(
            f"  MF-MOS uniquely added {mfmos_rescue} pts to dynamic cloud "
            f"({100.0*mfmos_rescue/total_dyn:.1f}% of total)."
        )
        print(
            f"    likely value-add (AW AMBIGUOUS/UNDER_EVIDENCED): {good} "
            f"({100.0*good/mfmos_rescue:.1f}% of rescues)"
        )
        print(
            f"    likely false-pos (AW STATIC/FREE_ONLY/not-in-keys): {bad} "
            f"({100.0*bad/mfmos_rescue:.1f}% of rescues)"
        )


if __name__ == "__main__":
    main()
