"""Fusion segmentation (`--seg union`): AW static map vetoes MF-MOS dynamics.

The third Step-B method. Unlike `aw` (classify/) and `mos` (mf_mos/) — which
are mutually exclusive and never import each other — `union` is the fusion
layer, so importing from both is expected and intentional.
"""

from .segment import UnionSegmentResult, classify_chunk

__all__ = ["classify_chunk", "UnionSegmentResult"]
