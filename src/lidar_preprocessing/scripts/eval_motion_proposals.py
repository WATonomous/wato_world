"""Score Step F's moving-point verdicts against ground truth, point by point.

At every instant gt_boxes.parquet labels (wato_common.artifacts.gt_boxes), the
frame nearest it (within --max-offset-ms) is scored: a non-ground point is
truly moving iff it lies inside a ground-truth box (any class) whose object
moves faster than --min-speed at that instant (wato_common.eval.motion
instant_speeds). Each verdict is scored as Chen et al.'s IoU_MOS
(TP / (TP + FP + FN)) with recall and precision:

  - every source bit, alone, and any bit;
  - clustered points, points of clusters with motion_score > 1, and points of
    clusters passing proposal_generation's lidar_mos gates (--gate-*);
  - Step B's dynamic_mask (the precision artifact) for comparison.

Recall is also split by the moving object's class and by range band.

    # in the lidar_preprocessing_dev container, or on a host with src/common:
    python /ws/src/lidar_preprocessing/scripts/eval_motion_proposals.py <bag_id> \
        [--chunk 0000] [--canonical-lidar lidar_cc] [--out FILE.json]

Results: docs/lidar_preprocessing_measurements.md. The leakage metric of
compare_seg_dynamic.py measures false positives only, this measures misses.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from wato_common.artifact_store import bag_root, chunks_index_path, local_path
from wato_common.chunk_spans import owned_windows
from wato_common.artifacts import gt_boxes, motion_proposals
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.artifacts.motion_proposals import BIT_NAMES
from wato_common.eval.boxes import band_of
from wato_common.eval.motion import instant_speeds
from wato_common.io.parquet_io import read_rows
from wato_common.schemas import Box3D
from wato_common.tracking import points_in_box


class Counts:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0

    def add(self, pred: np.ndarray, truth: np.ndarray) -> None:
        self.tp += int((pred & truth).sum())
        self.fp += int((pred & ~truth).sum())
        self.fn += int((~pred & truth).sum())

    def summary(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        return dict(
            tp=tp,
            fp=fp,
            fn=fn,
            iou_mos=tp / (tp + fp + fn) if tp + fp + fn else None,
            recall=tp / (tp + fn) if tp + fn else None,
            precision=tp / (tp + fp) if tp + fp else None,
        )


def _gt_with_speeds(bag_id: str) -> dict[int, list[tuple[gt_boxes.GtBox, float]]]:
    boxes = gt_boxes.read(bag_id)
    by_inst: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, b in enumerate(boxes):
        by_inst[(b.source, b.instance_id)].append(i)
    speed = np.zeros(len(boxes))
    for idx in by_inst.values():
        t = np.array([boxes[i].timestamp_ns for i in idx])
        xy = np.array([[boxes[i].cx, boxes[i].cy] for i in idx])
        speed[idx] = instant_speeds(t, xy)
    out: dict[int, list] = defaultdict(list)
    for b, v in zip(boxes, speed):
        out[b.timestamp_ns].append((b, float(v)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag_id")
    ap.add_argument("--chunk", default=None)
    ap.add_argument("--canonical-lidar", default=None)
    ap.add_argument("--max-offset-ms", type=float, default=25.0)
    ap.add_argument("--min-speed", type=float, default=0.5, help="m/s")
    ap.add_argument("--gate-motion-score", type=float, default=1.0)
    ap.add_argument("--gate-n-sources", type=int, default=2)
    ap.add_argument("--gate-frac-persistent", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not gt_boxes.exists(args.bag_id):
        raise SystemExit(
            f"{args.bag_id}: no gt_boxes.parquet — import ground truth first"
        )
    gt = _gt_with_speeds(args.bag_id)
    index = read_rows(chunks_index_path(args.bag_id))
    windows = owned_windows(index)
    chunks = [r["chunk_id"] for r in index]
    if args.chunk:
        chunks = [c for c in chunks if c == args.chunk]

    variants: dict[str, Counts] = defaultdict(Counts)
    recall_by: dict[str, Counts] = defaultdict(
        Counts
    )  # gated clusters, by class / band
    n_instants = 0
    max_off = int(args.max_offset_ms * 1e6)
    for cid in chunks:
        sweeps = ProcessedSweeps.load(args.bag_id, cid)
        frames = sweeps.frames(args.canonical_lidar)
        if not frames:
            continue
        clusters = motion_proposals.read_clusters(args.bag_id, cid)
        moving = {
            c.cluster_id for c in clusters if c.motion_score > args.gate_motion_score
        }
        gated = {
            c.cluster_id
            for c in clusters
            if c.motion_score > args.gate_motion_score
            and c.n_sources >= args.gate_n_sources
            and c.frac_persistent < args.gate_frac_persistent
        }
        moving_ids = np.array(sorted(moving), dtype=np.int64)
        gated_ids = np.array(sorted(gated), dtype=np.int64)
        f_t = np.array([f.reference_timestamp_ns for f in frames], dtype=np.int64)
        for t_gt, labelled in sorted(gt.items()):
            k = int(np.argmin(np.abs(f_t - t_gt)))
            if abs(int(f_t[k]) - t_gt) > max_off:
                continue
            if not windows[cid][0] <= f_t[k] < windows[cid][1]:
                continue  # a shared-span instant is scored in its owner
            n_instants += 1
            for sid in frames[k].sweep_ids:
                world, seg = sweeps.world_and_dynamic_mask(sid)
                prop = motion_proposals.read_sweep(
                    args.bag_id, cid, sid, n_points=world.n_points
                )
                keep = (
                    ~world.ground_mask
                    if world.ground_mask is not None
                    else np.ones(world.n_points, bool)
                )
                xyz = world.xyz[keep]
                bits = prop.source_bits[keep]
                cl = prop.cluster_id[keep].astype(np.int64)
                truth = np.zeros(xyz.shape[0], bool)
                truth_cls = np.full(xyz.shape[0], "", dtype=object)
                for b, v in labelled:
                    if v <= args.min_speed:
                        continue
                    inside = points_in_box(
                        xyz,
                        Box3D(
                            cx=b.cx,
                            cy=b.cy,
                            cz=b.cz,
                            w=b.w,
                            l=b.l,
                            h=b.h,
                            heading=b.heading,
                        ),
                    )
                    truth |= inside
                    truth_cls[inside] = b.cls or b.source_class
                rng = (
                    np.linalg.norm(xyz[:, :2] - world.origin[:2], axis=1)
                    if world.origin is not None
                    else np.zeros(xyz.shape[0])
                )
                preds = {
                    name: (bits & np.uint8(v)) != 0 for v, name in BIT_NAMES.items()
                }
                preds["any_bit"] = bits != 0
                preds["clustered"] = cl >= 0
                preds["moving_clusters"] = np.isin(cl, moving_ids)
                preds["gated_clusters"] = np.isin(cl, gated_ids)
                preds["dynamic_mask"] = seg[keep]
                for name, p in preds.items():
                    variants[name].add(p, truth)
                g = preds["gated_clusters"]
                for c in set(truth_cls[truth]):
                    sel = truth_cls == c
                    recall_by[f"class:{c}"].add(g & sel, truth & sel)
                bands = np.array([band_of(r) for r in rng], dtype=object)
                for band in set(bands[truth]):
                    sel = bands == band
                    recall_by[f"band:{band}"].add(g & sel, truth & sel)

    out = dict(
        bag_id=args.bag_id,
        instants=n_instants,
        min_speed_mps=args.min_speed,
        variants={k: v.summary() for k, v in sorted(variants.items())},
        gated_recall_by={k: v.summary() for k, v in sorted(recall_by.items())},
    )
    path = args.out or os.path.join(
        local_path(bag_root(args.bag_id)), "eval", "eval_motion_proposals.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(
        f"{args.bag_id}: {n_instants} labelled instants, moving = faster than {args.min_speed} m/s"
    )
    print(
        f"{'verdict':<16} {'IoU_MOS':>8} {'recall':>7} {'prec':>6} {'tp':>7} {'fp':>8} {'fn':>7}"
    )
    for k, s in out["variants"].items():

        def f(v):
            return f"{v:.3f}" if v is not None else "-"

        print(
            f"{k:<16} {f(s['iou_mos']):>8} {f(s['recall']):>7} {f(s['precision']):>6} {s['tp']:>7} {s['fp']:>8} {s['fn']:>7}"
        )
    print(
        "gated-cluster recall: "
        + ", ".join(
            f"{k} {v['recall']:.2f} ({v['tp'] + v['fn']} pts)"
            for k, v in out["gated_recall_by"].items()
            if v["recall"] is not None
        )
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
