"""Score refined_labels.parquet against ground truth, beside the boxes it refined.

    # in any dev container:
    python /ws/src/label_refinement/scripts/eval_refined.py <bag_id> \
        [--canonical-lidar lidar_cc] [--out FILE.json]

At every labelled instant (wato_common.eval.instants), the owned refined boxes
there and the owned track boxes they came from (tracks.parquet, as proposed)
are each matched to the ground truth within --distance-m
(wato_common.eval.boxes). Reported by range band and by ground-truth class:
recall, precision, and the errors on matched boxes (IoU, centre, size, height
offset, heading modulo pi), so refinement's effect reads straight off the two
rows. Vehicles (car, truck, bus) are also reported by their ground-truth
motion state (by_gt_motion): a parked and a moving vehicle's heading come
from different evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from wato_common.artifact_store import bag_root, chunks_index_path, local_path
from wato_common.artifacts import refined_labels, tracks
from wato_common.chunk_spans import owned_windows
from wato_common.eval.boxes import Scored, Tally, tally_frame
from wato_common.eval.instants import box_of, gt_by_time, labelled_instants
from wato_common.io.parquet_io import read_rows

VEHICLES = ("car", "truck", "bus")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag_id")
    ap.add_argument("--canonical-lidar", default=None)
    ap.add_argument("--max-offset-ms", type=float, default=25.0)
    ap.add_argument("--distance-m", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = gt_by_time(args.bag_id)
    if not gt:
        raise SystemExit(
            f"{args.bag_id}: no gt_boxes.parquet — import ground truth first"
        )
    sources = {
        "tracked": [f for f in tracks.read(args.bag_id)[0] if f.owned],
        "refined": [b for b in refined_labels.read(args.bag_id) if b.owned],
    }
    at: dict[str, dict] = {k: defaultdict(list) for k in sources}
    for k, rows in sources.items():
        for r in rows:
            at[k][(r.chunk_id, r.sweep_id)].append(r)

    bands: dict[str, dict[str, Tally]] = {k: {} for k in sources}
    by_class: dict[str, dict[str, Tally]] = {k: {} for k in sources}
    by_motion: dict[str, dict[str, Tally]] = {k: {} for k in sources}
    n = 0
    index = read_rows(chunks_index_path(args.bag_id))
    windows = owned_windows(index)
    for row in index:
        cid = row["chunk_id"]
        for inst in labelled_instants(
            args.bag_id,
            cid,
            args.canonical_lidar,
            gt,
            int(args.max_offset_ms * 1e6),
            window=windows[cid],
        ):
            n += 1
            motion = [
                f"vehicle:{b.motion_state}" if b.cls in VEHICLES else None
                for b in inst.boxes
            ]
            for k in sources:
                preds = [
                    Scored(
                        box_of(r),
                        score=getattr(r, "score", None)
                        or getattr(r, "confidence", 1.0),
                        cls=r.cls,
                        range_m=inst.range_of(r.cx, r.cy),
                    )
                    for r in at[k].get((cid, inst.frame.reference_sweep_id), [])
                ]
                tally_frame(
                    bands[k],
                    preds,
                    list(inst.gts),
                    args.distance_m,
                    by_class=by_class[k],
                    gt_groups=motion,
                    by_group=by_motion[k],
                )

    out = dict(
        bag_id=args.bag_id,
        instants=n,
        distance_m=args.distance_m,
        by_band={
            k: {b: t.summary() for b, t in sorted(v.items())} for k, v in bands.items()
        },
        by_class={
            k: {c: t.summary() for c, t in sorted(v.items())}
            for k, v in by_class.items()
        },
        by_gt_motion={
            k: {c: t.summary() for c, t in sorted(v.items())}
            for k, v in by_motion.items()
        },
    )
    path = args.out or os.path.join(
        local_path(bag_root(args.bag_id)), "eval", "eval_refined.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    def f(v, fmt="5.2f"):
        return format(v, fmt) if v is not None else "    -"

    print(
        f"{args.bag_id}: {n} labelled instants, matched within {args.distance_m:g} m; errors are medians, pred − gt"
    )
    print(
        f"{'':<8} {'band / class':<16} {'n_gt':>5} {'recall':>6} {'prec':>5} {'IoU':>5} {'ctr':>5} {'dl':>6} {'dw':>6} {'dh':>6} {'dz':>6} {'yaw':>5} {'yaw90':>5}"
    )
    for key, groups in (
        ("by_band", ["all", "0-15", "15-30", "30-50"]),
        ("by_class", ["car", "pedestrian", "truck"]),
        ("by_gt_motion", ["vehicle:moving", "vehicle:parked"]),
    ):
        for g in groups:
            for k in sources:
                s = out[key][k].get(g)
                if not s:
                    continue
                prec = f(s["precision"]) if key == "by_band" else "    -"
                print(
                    f"{k:<8} {g:<16} {s['n_gt']:>5} {f(s['recall'], '6.2f')} {prec} {f(s['median_iou'])} "
                    f"{f(s['median_centre_err_m'])} {f(s['median_dl_m'], '6.2f')} {f(s['median_dw_m'], '6.2f')} "
                    f"{f(s['median_dh_m'], '6.2f')} {f(s['median_dz_m'], '6.2f')} "
                    f"{f(s['median_yaw_err_deg'], '5.1f')} {f(s['p90_yaw_err_deg'], '5.1f')}"
                )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
