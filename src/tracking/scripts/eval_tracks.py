"""Score tracks.parquet against ground truth.

    # in the tracking (or any) dev container:
    python /ws/src/tracking/scripts/eval_tracks.py <bag_id> \
        [--canonical-lidar lidar_cc] [--out FILE.json]

At every labelled instant (wato_common.eval.instants), each owned track frame
there is matched to the ground truth within --distance-m (wato_common.eval.boxes,
by the frame's score). From those matches:

- **frames**: recall / precision of the tracked boxes, by range band — what the
  tracker's filter kept of the fused proposals' recall.
- **tracks**: a track seen at a labelled instant is true when most of its
  matched instants hit one ground-truth object, false when most hit nothing;
  tracks matched mostly to ignore-GT (cones, unseen boxes) aren't counted.
- **fragmentation**: distinct tracks per ground-truth object it matched.
- **seams**: each ground-truth object matched within half a shared span on
  both sides of a chunk seam (the midpoint where ownership changes), and
  whether its most-matched track is the same on both sides: the share that
  linking across chunks kept in one track.
- **motion state**: each ground-truth object with a parked / moving state
  against its dominant track's state (moving / parked / unknown).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from wato_common.artifact_store import bag_root, chunks_index_path, local_path
from wato_common.artifacts import tracks
from wato_common.chunk_spans import owned_windows, shared_spans
from wato_common.eval.boxes import Scored, Tally, match_frame, tally_frame
from wato_common.eval.instants import box_of, gt_by_time, labelled_instants
from wato_common.io.parquet_io import read_rows


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
    frames, summaries = tracks.read(args.bag_id)
    state = {s.track_id: s.motion_state for s in summaries}
    by_chunk_sweep: dict[tuple[str, int], list[tracks.TrackFrame]] = defaultdict(list)
    for f in frames:
        if f.owned:
            by_chunk_sweep[(f.chunk_id, f.sweep_id)].append(f)

    tally: dict[str, Tally] = {}
    seen: Counter = Counter()  # track -> labelled instants it appears at
    hits: dict[str, Counter] = defaultdict(Counter)  # track -> gt object -> matches
    ignored: Counter = Counter()  # track -> matches to ignore-GT
    gt_state: dict[tuple, str] = {}
    gt_cls: dict[tuple, str] = {}
    matched_at: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
    n_instants = 0
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
            tf = by_chunk_sweep.get((cid, inst.frame.reference_sweep_id), [])
            if not tf:
                continue
            n_instants += 1
            preds = [
                Scored(
                    box_of(f),
                    score=f.score,
                    cls=f.cls,
                    range_m=inst.range_of(f.cx, f.cy),
                )
                for f in tf
            ]
            gts = list(inst.gts)
            tally_frame(tally, preds, gts, args.distance_m)
            m = match_frame(preds, gts, args.distance_m)
            for i in range(len(tf)):
                seen[tf[i].track_id] += 1
            for i, j in m.pairs:
                b = inst.boxes[j]
                key = (b.source, b.instance_id)
                hits[tf[i].track_id][key] += 1
                matched_at[key].append(
                    (inst.frame.reference_timestamp_ns, tf[i].track_id)
                )
                if b.motion_state is not None:
                    gt_state[key] = b.motion_state
                    gt_cls[key] = b.cls
            for i in m.ignored_preds:
                ignored[tf[i].track_id] += 1

    true_tracks = false_tracks = 0
    for tid, n in seen.items():
        best = max(hits[tid].values(), default=0)
        if ignored[tid] > n / 2:
            continue
        if best > n / 2:
            true_tracks += 1
        else:
            false_tracks += 1
    tracks_of: dict[tuple, Counter] = defaultdict(Counter)
    for tid, c in hits.items():
        for key, n in c.items():
            tracks_of[key][tid] += n
    frag = [len(c) for c in tracks_of.values()]
    # Purity: the share of a track's matches on its main object; under 1 the
    # track switched between objects.
    purity = [max(c.values()) / sum(c.values()) for c in hits.values() if c]
    crossings = kept = 0
    for span in shared_spans(index):
        seam = windows[span.later][0]
        half = (span.end_ns - span.start_ns) / 2
        for at in matched_at.values():
            before = Counter(tid for t, tid in at if seam - half <= t < seam)
            after = Counter(tid for t, tid in at if seam <= t < seam + half)
            if before and after:
                crossings += 1
                kept += before.most_common(1)[0][0] == after.most_common(1)[0][0]
    confusion: Counter = Counter()
    summary_of = {s.track_id: s for s in summaries}
    objects = []
    for key, c in tracks_of.items():
        if key in gt_state:
            dominant = c.most_common(1)[0][0]
            confusion[(gt_state[key], state[dominant])] += 1
            s = summary_of[dominant]
            objects.append(
                dict(
                    gt=list(key),
                    gt_state=gt_state[key],
                    gt_cls=gt_cls[key],
                    track_id=dominant,
                    track_state=s.motion_state,
                    track_cls=s.cls,
                    n_frames=s.n_frames,
                    duration_s=(s.last_timestamp_ns - s.first_timestamp_ns) * 1e-9,
                    net_displacement_m=s.net_displacement_m,
                    centre_var_m2=s.centre_var_m2,
                    frac_lidar_mos=s.frac_lidar_mos,
                )
            )
    judged = sum(n for (g, t), n in confusion.items() if t != tracks.UNKNOWN)
    agree = sum(n for (g, t), n in confusion.items() if g == t)

    out = dict(
        bag_id=args.bag_id,
        instants=n_instants,
        distance_m=args.distance_m,
        frames={k: v.summary() for k, v in sorted(tally.items())},
        tracks=dict(
            seen=len(seen),
            true=true_tracks,
            false=false_tracks,
            precision=true_tracks / (true_tracks + false_tracks)
            if true_tracks + false_tracks
            else None,
        ),
        fragmentation=dict(
            objects=len(frag),
            mean_tracks_per_object=float(np.mean(frag)) if frag else None,
            share_in_one_track=float(np.mean([n == 1 for n in frag])) if frag else None,
        ),
        seams=dict(
            crossings=crossings,
            kept_in_one_track=kept / crossings if crossings else None,
        ),
        purity=dict(
            mean=float(np.mean(purity)) if purity else None,
            share_switched=float(np.mean([p < 0.9 for p in purity]))
            if purity
            else None,
        ),
        motion_state=dict(
            confusion={f"{g}->{t}": n for (g, t), n in sorted(confusion.items())},
            accuracy_when_judged=agree / judged if judged else None,
            objects=objects,
            unknown_share=(
                sum(n for (g, t), n in confusion.items() if t == tracks.UNKNOWN)
                / sum(confusion.values())
                if confusion
                else None
            ),
        ),
    )
    path = args.out or os.path.join(
        local_path(bag_root(args.bag_id)), "eval", "eval_tracks.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    a = out["frames"].get("all", {})
    print(
        f"{args.bag_id}: {n_instants} labelled instants, matched within {args.distance_m:g} m"
    )
    print(
        f"frames: recall {a.get('recall') or 0:.2f}, precision {a.get('precision') or 0:.2f} "
        f"({a.get('n_pred', 0)} owned track frames at labelled instants)"
    )
    for band in ("0-15", "15-30", "30-50", "50+"):
        s = out["frames"].get(band)
        if s:
            print(
                f"  {band:>6}: recall {s['recall'] or 0:.2f}, precision {s['precision'] or 0:.2f}"
            )
    t = out["tracks"]
    print(
        f"tracks: {t['seen']} seen, {t['true']} true, {t['false']} false, precision {t['precision'] or 0:.2f}"
    )
    fr = out["fragmentation"]
    print(
        f"fragmentation: {fr['objects']} objects, {fr['mean_tracks_per_object'] or 0:.2f} tracks each, "
        f"{fr['share_in_one_track'] or 0:.2f} in one track"
    )
    se = out["seams"]
    if se["crossings"]:
        print(
            f"seams: {se['crossings']} objects crossing a chunk seam, "
            f"{se['kept_in_one_track']:.2f} kept in one track"
        )
    pu = out["purity"]
    print(
        f"purity: mean {pu['mean'] or 0:.2f}, tracks switching objects {pu['share_switched'] or 0:.2f}"
    )
    ms = out["motion_state"]
    print(f"motion state (gt->track): {ms['confusion']}")
    print(
        f"  accuracy when judged {ms['accuracy_when_judged'] or 0:.2f}, unknown {ms['unknown_share'] or 0:.2f}"
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
