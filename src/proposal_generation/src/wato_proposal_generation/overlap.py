"""Resolve overlapping object support after independent 3D refinement.

This is a frame-local adaptation of SAL-4D's volume-ordered flattening: support
cardinality replaces temporal volume. The default strict IoM > 0.5 suppression
follows SAL-4D Appendix A.1.1 (https://arxiv.org/html/2504.00848#A1.SS1.SSS1).
It does not implement SAL-4D's spatio-temporal masklet suppression. Larger
refined hypotheses go first; duplicate hypotheses are suppressed by pairwise
intersection-over-minimum (IoM), then residual overlap goes to earlier ones.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def resolve_support(
    supports: dict[str, np.ndarray],
    *,
    min_points: int,
    max_iom: float = 0.5,
    accept: Callable[[str, np.ndarray, dict[str, np.ndarray]], bool] | None = None,
) -> dict[str, np.ndarray]:
    """Return disjoint supports, using the input's original frame row indices.

    Each support is a set of nonnegative integer indices. Repeated indices are
    counted once; returned arrays are sorted. Descending refined support size
    determines priority, with ascending instance ID breaking ties. A hypothesis
    is suppressed when its IoM with an accepted hypothesis is > ``max_iom``;
    this comparison uses both hypotheses' full support before allocation.

    Otherwise, already-owned points are removed. A hypothesis with fewer than
    ``min_points`` remaining is rejected without owning any points or taking
    part in later suppression, so later hypotheses can still recover its
    support. When supplied, ``accept(iid, remaining, accepted)`` checks the
    candidate's final support against already accepted survivors. Returning
    false rejects it before allocation. The predicate must not change its
    arguments; accepted hypotheses are never subsequently removed. Inputs
    are never changed.
    """
    if (
        isinstance(min_points, (bool, np.bool_))
        or not isinstance(min_points, (int, np.integer))
        or min_points < 1
    ):
        raise ValueError("min_points must be a positive integer")
    if not np.isfinite(max_iom) or not 0 <= max_iom <= 1:
        raise ValueError("max_iom must be finite and in [0, 1]")

    claims = {}
    for iid, indices in supports.items():
        indices = np.asarray(indices)
        if indices.ndim != 1 or (
            indices.size and (indices.dtype.kind not in "iu" or np.any(indices < 0))
        ):
            raise ValueError(
                f"support {iid!r} must contain nonnegative integer indices"
            )
        claims[iid] = np.unique(indices.astype(np.int64))

    accepted: dict[str, np.ndarray] = {}
    owned = np.empty(0, np.int64)
    for iid in sorted(claims, key=lambda i: (-len(claims[i]), i)):
        claim = claims[iid]
        if claim.size < min_points:
            continue
        if any(
            np.intersect1d(claim, claims[other], assume_unique=True).size
            / min(claim.size, claims[other].size)
            > max_iom
            for other in accepted
        ):
            continue
        remaining = np.setdiff1d(claim, owned, assume_unique=True)
        if remaining.size < min_points:
            continue
        if accept is not None and not accept(iid, remaining, accepted):
            continue
        accepted[iid] = remaining
        owned = np.union1d(owned, remaining)
    return accepted
