"""Multi-object tracking metrics: HOTA's association terms, CLEAR and IDF1.

TrackEval's definitions (Luiten et al., HOTA, arXiv 2009.07736 §6; code at
JonathonLuiten/TrackEval@12c8791, MIT: trackeval/metrics/hota.py, clear.py,
identity.py), on one sequence, at one similarity threshold:

- **HOTA at one α.** Per frame, ground truth and tracker boxes are matched by
  the Hungarian method on global alignment × similarity, keeping pairs at
  similarity ≥ α. AssA, AssRe and AssPr then score, per match, how much of its
  ground-truth trajectory and of its tracker trajectory the same id pair
  covers. AssRe falls when one object is split over several tracks; AssPr when
  one track covers several objects.
- **CLEAR.** Per frame, the Hungarian method first keeps the previous frame's
  pairs (score + 1000 · same id), then maximises similarity. IDSW counts a
  ground-truth object matched to a different tracker id than its last one;
  Frag counts an object's untracked → tracked transitions after its first.
  id_switches lists the IDSW events themselves, for attributing their causes.
- **Identity.** One global id-to-id mapping maximising matched detections:
  IDF1 = IDTP / (IDTP + ½IDFP + ½IDFN). An object split into k tracks earns
  credit for its best-mapped one only.

Frames carry ids and a similarity matrix, as TrackEval's per-sequence data
does; distance_similarity turns BEV centre distance into one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

_EPS = np.finfo("float").eps


@dataclass(frozen=True)
class MotFrame:
    """One frame: ground-truth and tracker ids, and their similarity
    (n_gt × n_tracker, in [0, 1])."""

    gt_ids: tuple[Hashable, ...]
    tracker_ids: tuple[Hashable, ...]
    similarity: np.ndarray


def distance_similarity(
    gt_xy: np.ndarray, tracker_xy: np.ndarray, match_dist_m: float
) -> np.ndarray:
    """max(0, 1 − d / (2 · match_dist_m)) for BEV centre distance d: 0.5 at
    match_dist_m, so TrackEval's default threshold 0.5 matches within it."""
    gt_xy = np.asarray(gt_xy, dtype=float).reshape(-1, 2)
    tracker_xy = np.asarray(tracker_xy, dtype=float).reshape(-1, 2)
    d = np.linalg.norm(gt_xy[:, None, :] - tracker_xy[None, :, :], axis=2)
    return np.maximum(0.0, 1.0 - d / (2.0 * match_dist_m))


def _indexed(frames: Sequence[MotFrame]):
    """Frames with ids replaced by dense ints, plus the id counts."""
    gt_index: dict[Hashable, int] = {}
    tr_index: dict[Hashable, int] = {}
    out = []
    for f in frames:
        g = np.array([gt_index.setdefault(i, len(gt_index)) for i in f.gt_ids], int)
        t = np.array(
            [tr_index.setdefault(i, len(tr_index)) for i in f.tracker_ids], int
        )
        sim = np.asarray(f.similarity, dtype=float).reshape(len(g), len(t))
        out.append((g, t, sim))
    return out, len(gt_index), len(tr_index)


def hota(frames: Sequence[MotFrame], alpha: float = 0.5) -> dict:
    """DetA, AssA, AssRe, AssPr and HOTA at one α (TrackEval hota.py)."""
    data, n_gt, n_tr = _indexed(frames)
    potential = np.zeros((n_gt, n_tr))
    gt_count = np.zeros((n_gt, 1))
    tr_count = np.zeros((1, n_tr))
    for g, t, sim in data:
        denom = sim.sum(0)[np.newaxis, :] + sim.sum(1)[:, np.newaxis] - sim
        sim_iou = np.zeros_like(sim)
        mask = denom > 0 + _EPS
        sim_iou[mask] = sim[mask] / denom[mask]
        potential[g[:, np.newaxis], t[np.newaxis, :]] += sim_iou
        gt_count[g] += 1
        tr_count[0, t] += 1
    global_alignment = potential / np.maximum(_EPS, gt_count + tr_count - potential)
    matches = np.zeros((n_gt, n_tr))
    tp = fn = fp = 0
    for g, t, sim in data:
        if len(g) == 0 or len(t) == 0:
            fn += len(g)
            fp += len(t)
            continue
        score = global_alignment[g[:, np.newaxis], t[np.newaxis, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        ok = sim[rows, cols] >= alpha - _EPS
        rows, cols = rows[ok], cols[ok]
        tp += len(rows)
        fn += len(g) - len(rows)
        fp += len(t) - len(rows)
        matches[g[rows], t[cols]] += 1
    denom_tp = max(1, tp)
    ass_a = matches / np.maximum(1, gt_count + tr_count - matches)
    ass_re = matches / np.maximum(1, gt_count)
    ass_pr = matches / np.maximum(1, tr_count)
    det_a = tp / max(1, tp + fn + fp)
    out = dict(
        TP=tp,
        FN=fn,
        FP=fp,
        DetA=det_a,
        AssA=float(np.sum(matches * ass_a) / denom_tp),
        AssRe=float(np.sum(matches * ass_re) / denom_tp),
        AssPr=float(np.sum(matches * ass_pr) / denom_tp),
    )
    out["HOTA"] = float(np.sqrt(out["DetA"] * out["AssA"]))
    return out


@dataclass(frozen=True)
class IdSwitch:
    """One IDSW as clear() counts it: at frame index `frame`, ground-truth
    object `gt_id` is matched to `tracker_id`, not to `previous_tracker_id`,
    the last tracker id it was matched to (in any earlier frame)."""

    frame: int
    gt_id: Hashable
    previous_tracker_id: Hashable
    tracker_id: Hashable


def _clear(frames: Sequence[MotFrame], threshold: float) -> tuple[dict, list[IdSwitch]]:
    """TrackEval clear.py's matching, with the IDSW events it counts."""
    data, n_gt, n_tr = _indexed(frames)
    prev_tracker = np.full(n_gt, np.nan)  # last tracker id each gt matched
    prev_step = np.full(n_gt, np.nan)  # tracker id in the previous frame
    frag_count = np.zeros(n_gt)
    tracker_name: dict[int, Hashable] = {}
    events: list[IdSwitch] = []
    tp = fn = fp = idsw = 0
    for k, (f, (g, t, sim)) in enumerate(zip(frames, data)):
        tracker_name.update(zip(t.tolist(), f.tracker_ids))
        if len(g) == 0:
            fp += len(t)
            continue
        if len(t) == 0:
            fn += len(g)
            continue
        score = (t[np.newaxis, :] == prev_step[g[:, np.newaxis]]).astype(float)
        score = 1000.0 * score + sim
        score[sim < threshold - _EPS] = 0.0
        rows, cols = linear_sum_assignment(-score)
        ok = score[rows, cols] > 0 + _EPS
        rows, cols = rows[ok], cols[ok]
        mg, mt = g[rows], t[cols]
        prev = prev_tracker[mg]
        switched = ~np.isnan(prev) & (mt != prev)
        idsw += int(np.sum(switched))
        for r in np.flatnonzero(switched):
            events.append(
                IdSwitch(
                    frame=k,
                    gt_id=f.gt_ids[rows[r]],
                    previous_tracker_id=tracker_name[int(prev[r])],
                    tracker_id=f.tracker_ids[cols[r]],
                )
            )
        not_before = np.isnan(prev_step)
        prev_tracker[mg] = mt
        prev_step[:] = np.nan
        prev_step[mg] = mt
        frag_count += not_before & ~np.isnan(prev_step)
        tp += len(rows)
        fn += len(g) - len(rows)
        fp += len(t) - len(rows)
    n_gt_dets = tp + fn
    counts = dict(
        TP=tp,
        FN=fn,
        FP=fp,
        IDSW=idsw,
        Frag=int(np.sum(frag_count[frag_count > 0] - 1)),
        MOTA=(n_gt_dets - fn - fp - idsw) / max(1, n_gt_dets),
    )
    return counts, events


def clear(frames: Sequence[MotFrame], threshold: float = 0.5) -> dict:
    """MOTA, IDSW and Frag (TrackEval clear.py)."""
    return _clear(frames, threshold)[0]


def id_switches(frames: Sequence[MotFrame], threshold: float = 0.5) -> list[IdSwitch]:
    """Every IDSW clear() counts, in frame order."""
    return _clear(frames, threshold)[1]


def identity(frames: Sequence[MotFrame], threshold: float = 0.5) -> dict:
    """IDF1, IDTP, IDFP, IDFN (TrackEval identity.py)."""
    data, n_gt, n_tr = _indexed(frames)
    potential = np.zeros((n_gt, n_tr))
    gt_count = np.zeros(n_gt)
    tr_count = np.zeros(n_tr)
    for g, t, sim in data:
        gi, ti = np.nonzero(sim >= threshold - _EPS)
        np.add.at(potential, (g[gi], t[ti]), 1)
        gt_count[g] += 1
        tr_count[t] += 1
    n = n_gt + n_tr
    fp_mat = np.zeros((n, n))
    fn_mat = np.zeros((n, n))
    fp_mat[n_gt:, :n_tr] = 1e10
    fn_mat[:n_gt, n_tr:] = 1e10
    for i in range(n_gt):
        fn_mat[i, :n_tr] = gt_count[i]
        fn_mat[i, n_tr + i] = gt_count[i]
    for j in range(n_tr):
        fp_mat[:n_gt, j] = tr_count[j]
        fp_mat[j + n_gt, j] = tr_count[j]
    fn_mat[:n_gt, :n_tr] -= potential
    fp_mat[:n_gt, :n_tr] -= potential
    rows, cols = linear_sum_assignment(fn_mat + fp_mat)
    idfn = int(fn_mat[rows, cols].sum())
    idfp = int(fp_mat[rows, cols].sum())
    idtp = int(gt_count.sum() - idfn)
    return dict(
        IDTP=idtp,
        IDFP=idfp,
        IDFN=idfn,
        IDF1=idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn),
    )
