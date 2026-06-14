"""Fusion segmentation (`--seg union`): AW static map vetoes MF-MOS dynamics.

The third Step-B method. Unlike `aw` (classify/) and `mos` (mf_mos/) — which
are mutually exclusive and never import each other — `union` is the fusion
layer, so importing from both is expected and intentional.
"""

from .motion_filter import MotionFilterResult, filter_dynamic_artifacts
from .segment import UnionSegmentResult, classify_chunk

__all__ = [
    "classify_chunk",
    "UnionSegmentResult",
    "filter_dynamic_artifacts",
    "MotionFilterResult",
]
