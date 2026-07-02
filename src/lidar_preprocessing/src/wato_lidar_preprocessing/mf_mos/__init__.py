"""MF-MOS learned moving-object segmentation (`--seg mos`).

Self-contained alternative to classify/ (Amanatides-Woo). Two stages:
  process_chunk  — model inference → per-sweep moving masks (_core).
  classify_chunk — masks → static/dynamic clouds (segment); no ray traversal.
"""

from ._core import (
    MFMosResult,
    _compute_residual,
    _range_project,
    _unproject_mask,
    _unproject_scores,
    process_chunk,
)
from .segment import MosSegmentResult, classify_chunk

__all__ = [
    "process_chunk",
    "MFMosResult",
    "classify_chunk",
    "MosSegmentResult",
    "_compute_residual",
    "_range_project",
    "_unproject_mask",
    "_unproject_scores",
]
