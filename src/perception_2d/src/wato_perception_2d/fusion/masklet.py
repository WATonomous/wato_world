"""Masklet: one temporally-associated object track within a single camera view.

Produced by the detector + SAM2 tracker (one per tracked object id) and consumed
by the parquet writer in pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Masklet:
    """One temporally-associated object track within a single camera view."""

    masklet_id: str
    bag_id: str
    chunk_id: str
    cam_id: str
    cls: str
    score: float
    frames_present: list[int]  # camera_seq values where the mask exists
    mask_paths: list[str]  # parallel to frames_present
    tracker_backend: str  # segmentation.backend that produced it: sam2 / dam4sam
