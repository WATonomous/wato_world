"""Step C — Aggregate per-sweep ground masks + height-grid builder."""

from ._core import (
    GroundResult,
    _build_height_grid,
    process_chunk,
)

__all__ = [
    "process_chunk",
    "GroundResult",
    "_build_height_grid",
]
