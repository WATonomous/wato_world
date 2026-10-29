"""Fusion segmentation (`--seg union`): AW static map vetoes MF-MOS dynamics.

`aw` (classify/) and `mos` (mf_mos/) never import each other; union, the
fusion layer, imports both.
"""

from .motion_filter import MotionFilterResult, filter_dynamic_artifacts
from .segment import UnionSegmentResult, classify_chunk

__all__ = [
    "classify_chunk",
    "UnionSegmentResult",
    "filter_dynamic_artifacts",
    "MotionFilterResult",
]
