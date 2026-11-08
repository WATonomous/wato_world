"""Score proposals.parquet against ground truth, plus the proxy numbers that
need none.

    # in the proposal_generation_dev container, or on a host with src/common:
    python /ws/src/proposal_generation/scripts/eval_proposals.py <bag_id> \
        [--chunk 0000] [--canonical-lidar lidar_cc] [--out FILE.json]

**Ground truth** (when gt_boxes.parquet exists, wato_common.artifacts.gt_boxes):
at every instant the ground truth labels, each provenance's rows in the frame
nearest that instant (within --max-offset-ms) are matched to it
class-agnostically at each nuScenes centre distance, by range band from the
ego (wato_common.eval.boxes). A ground-truth box with no non-ground LiDAR
point in that frame is ignored, as the nuScenes benchmark drops such boxes.
Per-class recall is reported at --class-distance-m.

**Proxy** (always): the numbers docs/proposal_generation_measurements.md and
the motion-proposal review used — Step F clusters per frame, gated lidar_mos
rows and tracks, wall-shaped and small tracks, camera-classed share by range,
camera-confirmed pedestrian tracks, and the lidar_mos ↔ lifted pair rate.

Chunks overlap, so an instant two chunks both label counts in both.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Optional

import numpy as np

from wato_common.artifact_store import bag_root, chunks_index_path, local_path
from wato_common.artifacts import motion_proposals, proposals
from wato_common.artifacts.lidar_sweeps import Frame, ProcessedSweeps
from wato_common.chunk_spans import owned_windows
from wato_common.eval.boxes import (
    DISTANCE_THRESHOLDS_M,
    Scored,
    band_of,
    tally_frame,
)
from wato_common.eval.instants import LabelledInstant, gt_by_time, labelled_instants
from wato_common.io.parquet_io import read_rows
from wato_common.pose_lookup import PoseLookup
from wato_common.schemas import Box3D

# A track whose median box is longer than this many widths is wall-shaped
# (LISO's length/width limit).
WALL_ASPECT = 4.0
# A track whose median footprint is below this is small (LISO's floor, m²).
SMALL_FOOTPRINT_M2 = 0.35


def _box(p) -> Box3D:
    return Box3D(cx=p.cx, cy=p.cy, cz=p.cz, w=p.w, l=p.l, h=p.h, heading=p.heading)


def _ego_xy(poses: PoseLookup, t_ns: int) -> Optional[np.ndarray]:
    pose = poses.at(t_ns)
    return pose.world_T_ego[:2, 3] if pose.valid else None


# --------------------------------------------------------------------------- #
# Proxy
# --------------------------------------------------------------------------- #


def proxy(
    rows: list[proposals.Proposal],
    clusters: list[motion_proposals.MotionCluster],
    frames: list[Frame],
    poses: PoseLookup,
) -> dict:
    mos = [p for p in rows if p.provenance == "lidar_mos"]
    hint = {c.cluster_id: c.track_hint_id for c in clusters}
    tracks: dict[int, list[proposals.Proposal]] = defaultdict(list)
    for p in mos:
        tracks[hint[int(p.proposal_id.split("-", 1)[1])]].append(p)

    def median_shape(ps):
        return np.median([p.l for p in ps]), np.median([p.w for p in ps])

    shapes = [median_shape(ps) for ps in tracks.values()]
    wall = sum(1 for length, width in shapes if length > WALL_ASPECT * width)
    small = sum(1 for length, width in shapes if length * width < SMALL_FOOTPRINT_M2)
    ped_tracks = sum(
        1 for ps in tracks.values() if any(p.cls == "pedestrian" for p in ps)
    )
    classed_tracks = sum(1 for ps in tracks.values() if any(p.cls for p in ps))

    by_band: dict[str, list[bool]] = defaultdict(list)
    for p in mos:
        ego = _ego_xy(poses, p.reference_timestamp_ns)
        if ego is not None:
            by_band[band_of(math.hypot(p.cx - ego[0], p.cy - ego[1]))].append(
                p.cls is not None
            )
    pairs = sum(
        1
        for p in rows
        if p.provenance == proposals.FUSED and len(p.member_proposal_ids) > 1
    )
    n_frames = max(len(frames), 1)
    return dict(
        frames=len(frames),
        step_f_clusters_per_frame=len(clusters) / n_frames,
        lidar_mos_rows=len(mos),
        lidar_mos_per_frame=len(mos) / n_frames,
        lidar_mos_tracks=len(tracks),
        wall_shaped_tracks=wall,
        small_tracks=small,
        camera_classed_tracks=classed_tracks,
        camera_pedestrian_tracks=ped_tracks,
        classed_share_by_band={
            k: float(np.mean(v)) for k, v in sorted(by_band.items())
        },
        lifted_rows=sum(1 for p in rows if p.provenance == "lifted"),
        fused_rows=sum(1 for p in rows if p.provenance == proposals.FUSED),
        fused_pairs=pairs,
        pairs_per_lidar_mos=pairs / len(mos) if mos else None,
    )


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


def score_chunk(
    rows: list[proposals.Proposal],
    instants: list[LabelledInstant],
    class_distance_m: float,
    tallies: dict,
    class_tallies: dict,
) -> None:
    """Add the chunk's labelled instants to the tallies."""
    by_frame: dict[int, list[proposals.Proposal]] = defaultdict(list)
    for p in rows:
        by_frame[p.sweep_id].append(p)
    for inst in instants:
        frame_rows = by_frame.get(inst.frame.reference_sweep_id, [])
        for prov in proposals.PROVENANCES:
            preds = [
                Scored(
                    _box(p), score=p.score, cls=p.cls, range_m=inst.range_of(p.cx, p.cy)
                )
                for p in frame_rows
                if p.provenance == prov
            ]
            for d in DISTANCE_THRESHOLDS_M:
                tally_frame(
                    tallies[prov].setdefault(f"{d:g}m", {}),
                    preds,
                    list(inst.gts),
                    d,
                    by_class=class_tallies[prov] if d == class_distance_m else None,
                )


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag_id")
    ap.add_argument("--chunk", default=None)
    ap.add_argument("--canonical-lidar", default=None)
    ap.add_argument("--max-offset-ms", type=float, default=25.0)
    ap.add_argument("--class-distance-m", type=float, default=2.0)
    ap.add_argument("--out", default=None, help="JSON path (default: <bag>/eval/)")
    args = ap.parse_args()

    index = read_rows(chunks_index_path(args.bag_id))
    windows = owned_windows(index)
    chunks = [r["chunk_id"] for r in index]
    if args.chunk:
        chunks = [c for c in chunks if c == args.chunk]
    gt = gt_by_time(args.bag_id)

    out: dict = dict(bag_id=args.bag_id, chunks={}, gt=None)
    tallies: dict = defaultdict(dict)
    class_tallies: dict = defaultdict(dict)
    offsets: list[float] = []
    n_instants = 0
    for cid in chunks:
        rows = proposals.read(args.bag_id, cid)
        frames = ProcessedSweeps.load(args.bag_id, cid).frames(args.canonical_lidar)
        poses = PoseLookup.load(args.bag_id, cid)
        clusters = motion_proposals.read_clusters(args.bag_id, cid)
        out["chunks"][cid] = proxy(rows, clusters, frames, poses)
        instants = labelled_instants(
            args.bag_id,
            cid,
            args.canonical_lidar,
            gt,
            int(args.max_offset_ms * 1e6),
            window=windows[cid],
        )
        n_instants += len(instants)
        offsets += [
            (i.frame.reference_timestamp_ns - i.boxes[0].timestamp_ns) * 1e-6
            for i in instants
            if i.boxes
        ]
        score_chunk(rows, instants, args.class_distance_m, tallies, class_tallies)

    if gt:
        out["gt"] = dict(
            instants=n_instants,
            frame_offset_ms_abs_max=float(np.abs(offsets).max()) if offsets else None,
            class_distance_m=args.class_distance_m,
            by_provenance={
                prov: {
                    d: {band: t.summary() for band, t in sorted(bands.items())}
                    for d, bands in by_d.items()
                }
                for prov, by_d in tallies.items()
            },
            by_class={
                prov: {c: t.summary() for c, t in sorted(cs.items())}
                for prov, cs in class_tallies.items()
            },
        )

    path = args.out or os.path.join(
        local_path(bag_root(args.bag_id)), "eval", "eval_proposals.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    _print(out)
    print(f"\nwrote {path}")


def _print(out: dict) -> None:
    for cid, p in out["chunks"].items():
        print(
            f"chunk {cid}: {p['frames']} frames, {p['step_f_clusters_per_frame']:.1f} "
            f"Step F clusters/frame, {p['lidar_mos_per_frame']:.2f} lidar_mos/frame, "
            f"{p['lidar_mos_tracks']} tracks ({p['wall_shaped_tracks']} wall-shaped, "
            f"{p['small_tracks']} small, {p['camera_classed_tracks']} camera-classed, "
            f"{p['camera_pedestrian_tracks']} pedestrian), "
            f"pairs/lidar_mos {p['pairs_per_lidar_mos'] or 0:.2f}"
        )
    g = out["gt"]
    if not g:
        print("no gt_boxes.parquet: proxy numbers only")
        return
    d = f"{g['class_distance_m']:g}m"

    def f(v, fmt):
        return format(v, fmt) if v is not None else "-"

    def row(name, s, gt_side_only=False):
        n_pred = "-" if gt_side_only else s["n_pred"]
        prec = "     -" if gt_side_only else f(s["precision"], "6.2f")
        return (
            f"{name:<22} {s['n_gt']:>5} {n_pred:>6} {f(s['recall'], '7.2f')} "
            f"{prec} {f(s['median_iou'], '5.2f')} "
            f"{f(s['median_centre_err_m'], '5.2f')} {f(s['median_dl_m'], '6.2f')} "
            f"{f(s['median_dw_m'], '6.2f')} {f(s['median_dh_m'], '6.2f')} {f(s['median_dz_m'], '6.2f')}"
        )

    print(
        f"\nground truth: {g['instants']} instants (frame offset ≤ "
        f"{g['frame_offset_ms_abs_max']:.1f} ms); matched within {d}; errors are medians, pred − gt"
    )
    print(
        f"{'source / band or class':<22} {'n_gt':>5} {'n_pred':>6} {'recall':>7} {'prec':>6} {'IoU':>5} {'ctr':>5} {'dl':>6} {'dw':>6} {'dh':>6} {'dz':>6}"
    )
    for prov, by_d in g["by_provenance"].items():
        if not by_d[d]["all"]["n_pred"]:
            continue
        for band, s in by_d[d].items():
            print(row(f"{prov} {band}", s))
        for c, s in g["by_class"][prov].items():
            print(row(f"{prov} [{c}]", s, gt_side_only=True))
    print(
        "recall at each distance (all bands): "
        + "; ".join(
            f"{prov} "
            + " ".join(f"{k} {v['all']['recall']:.2f}" for k, v in by_d.items())
            for prov, by_d in g["by_provenance"].items()
            if by_d[d]["all"]["n_pred"]
        )
    )


if __name__ == "__main__":
    main()
