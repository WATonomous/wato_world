"""Competing 2D mask hypotheses lifted onto non-ground LiDAR returns.

semantic_lifting writes one sparse relation per sweep, including empty sweeps;
proposal_generation reads every hypothesis before geometric refinement and
ownership. A point can occur for several instances, but each (point, instance)
pair occurs once. Cross-camera claims of one merged instance are deduplicated.

Files are semantic_lifting/lifted_candidates/<sweep_id:06d>.npz, with aligned
one-dimensional point_indices (int32), instance_ids (unicode) and confidences
(float32). Confidence is class agreement among independent camera observations,
not calibrated reliability or an exclusive ownership probability.
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

    @classmethod
    def empty(cls) -> "SweepCandidates":
        return cls(np.zeros(0, np.int32), np.zeros(0, np.str_), np.zeros(0, np.float32))


_DTYPES = {
    "point_indices": np.int32,
    "instance_ids": np.str_,
    "confidences": np.float32,
}


def _validate(arrays: dict[str, np.ndarray], n_points: Optional[int] = None) -> None:
    shapes = {a.shape for a in arrays.values()}
    if len(shapes) != 1 or any(a.ndim != 1 for a in arrays.values()):
        raise ValueError("candidate arrays must share one (M,) shape")
    idx, ids, conf = (arrays[k] for k in _DTYPES)
    if idx.dtype.kind not in "iu" or ids.dtype.kind != "U" or conf.dtype.kind != "f":
        raise ValueError(
            "candidate arrays require integer indices, unicode ids and float confidences"
        )
    if idx.size and (idx.min() < 0 or (n_points is not None and idx.max() >= n_points)):
        raise ValueError(f"point_indices outside the world sweep's {n_points} points")
    if np.any(ids == ""):
        raise ValueError("candidate instance_ids must be nonempty")
    if not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
        raise ValueError("candidate confidences must be finite and within [0, 1]")
    if idx.size:
        pairs = np.rec.fromarrays([idx, ids])
        if np.unique(pairs).size != idx.size:
            raise ValueError("duplicate (point, instance) candidate pair")


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
    """Read all claims; missing/legacy output requires rerunning semantic_lifting."""
    path = local_path(lifted_candidates_path(bag_id, chunk_id, sweep_id))
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = set(_DTYPES) - set(data.files)
            if missing:
                raise ValueError(f"missing candidate arrays {sorted(missing)}")
            arrays = {k: data[k] for k in _DTYPES}
        _validate(arrays, n_points)
    except (OSError, ValueError) as exc:
        raise ArtifactFormatError(path, f"{exc}; re-run semantic_lifting") from None
    return SweepCandidates(**arrays)
