"""Competing 2D mask hypotheses lifted onto non-ground LiDAR returns.

semantic_lifting writes two sparse relations per sweep, including empty
sweeps; proposal_generation reads them before geometric refinement and
ownership.

**Candidates** (``point_indices``, ``instance_ids``, ``confidences``): every
visible non-ground return a mask covers, per instance. A point can occur for
several instances, but each (point, instance) pair occurs once.
Cross-camera claims of one merged instance are deduplicated. Confidence is
class agreement among independent camera observations, not calibrated
reliability or an exclusive ownership probability.

**Frustum** (``frustum_point_indices``, ``frustum_instance_ids``): every
non-ground return whose pixel falls inside the bounding rectangle of one of
the instance's masks in a camera frame paired with the sweep, visible or
not — the occluder in front and the background behind included. It is the 2D
box frustum FGR's segmentation test measures a LiDAR segment against
(proposal_generation's verification). A mask lies inside its rectangle, so
every candidate pair is also a frustum pair. Only instances with candidates
somewhere in the chunk appear.

Files are semantic_lifting/lifted_candidates/<sweep_id:06d>.npz, each relation
one-dimensional and aligned within itself: indices int32 into the world
sweep, ids unicode, confidences float32. A file without the frustum arrays
predates them and raises with a rerun instruction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import lifted_candidates_path, local_path
from wato_common.artifacts import ArtifactFormatError


@dataclass(frozen=True)
class SweepCandidates:
    point_indices: np.ndarray
    instance_ids: np.ndarray
    confidences: np.ndarray
    frustum_point_indices: np.ndarray
    frustum_instance_ids: np.ndarray

    @classmethod
    def empty(cls) -> "SweepCandidates":
        return cls(
            np.zeros(0, np.int32),
            np.zeros(0, np.str_),
            np.zeros(0, np.float32),
            np.zeros(0, np.int32),
            np.zeros(0, np.str_),
        )


_DTYPES = {
    "point_indices": np.int32,
    "instance_ids": np.str_,
    "confidences": np.float32,
    "frustum_point_indices": np.int32,
    "frustum_instance_ids": np.str_,
}
_CANDIDATES = ("point_indices", "instance_ids", "confidences")
_FRUSTUM = ("frustum_point_indices", "frustum_instance_ids")


def _one_shape(arrays: dict[str, np.ndarray], keys: tuple[str, ...], what: str) -> None:
    shapes = {arrays[k].shape for k in keys}
    if len(shapes) != 1 or any(arrays[k].ndim != 1 for k in keys):
        raise ValueError(f"{what} arrays must share one (M,) shape")


def _check_relation(
    idx: np.ndarray, ids: np.ndarray, n_points: Optional[int], what: str
) -> None:
    if idx.dtype.kind not in "iu" or ids.dtype.kind != "U":
        raise ValueError(f"{what} arrays require integer indices and unicode ids")
    if idx.size and (idx.min() < 0 or (n_points is not None and idx.max() >= n_points)):
        raise ValueError(
            f"{what} point indices outside the world sweep's {n_points} points"
        )
    if np.any(ids == ""):
        raise ValueError(f"{what} instance ids must be nonempty")
    if idx.size and np.unique(_pair_keys(idx, ids, np.unique(ids))).size != idx.size:
        raise ValueError(f"duplicate (point, instance) {what} pair")


def _pair_keys(idx: np.ndarray, ids: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
    """One int64 per (point, instance), comparable across relations that share
    ``vocabulary`` (sorted unique ids covering both)."""
    code = np.searchsorted(vocabulary, ids)
    return idx.astype(np.int64) * len(vocabulary) + code


def _validate(arrays: dict[str, np.ndarray], n_points: Optional[int] = None) -> None:
    _one_shape(arrays, _CANDIDATES, "candidate")
    _one_shape(arrays, _FRUSTUM, "frustum")
    idx, ids, conf = (arrays[k] for k in _CANDIDATES)
    if conf.dtype.kind != "f":
        raise ValueError("candidate confidences must be floats")
    _check_relation(idx, ids, n_points, "candidate")
    if not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
        raise ValueError("candidate confidences must be finite and within [0, 1]")
    f_idx, f_ids = (arrays[k] for k in _FRUSTUM)
    _check_relation(f_idx, f_ids, n_points, "frustum")
    if idx.size:
        vocabulary = np.unique(np.concatenate([ids, f_ids]))
        inside = np.isin(
            _pair_keys(idx, ids, vocabulary), _pair_keys(f_idx, f_ids, vocabulary)
        )
        if not inside.all():
            raise ValueError(
                f"{int((~inside).sum())} candidate pairs lie outside their "
                "instance's frustum"
            )


def write_sweep(
    bag_id: str, chunk_id: str, sweep_id: int, candidates: SweepCandidates
) -> None:
    arrays = {
        k: np.asarray(getattr(candidates, k), dtype=t) for k, t in _DTYPES.items()
    }
    _validate(arrays)
    path = local_path(lifted_candidates_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def read_sweep(
    bag_id: str, chunk_id: str, sweep_id: int, *, n_points: Optional[int] = None
) -> SweepCandidates:
    """Read both relations; missing/legacy output requires rerunning semantic_lifting."""
    path = local_path(lifted_candidates_path(bag_id, chunk_id, sweep_id))
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = set(_DTYPES) - set(data.files)
            if missing:
                raise ValueError(f"missing candidate arrays {sorted(missing)}")
            arrays = {k: data[k] for k in _DTYPES}
        _validate(arrays, n_points)
    except (OSError, ValueError) as exc:
        raise ArtifactFormatError(
            path, f"{exc}; re-run semantic_lifting with --force"
        ) from None
    return SweepCandidates(**arrays)
