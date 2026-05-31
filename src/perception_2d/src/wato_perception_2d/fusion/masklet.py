"""Masklet: one temporally-associated object track within a single camera view.

Produced by the SAM 3.1 concept tracker (one per tracked object id) and consumed
by cross_cam_merge + the parquet writer in pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


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
    dino_feature: Optional[np.ndarray] = None  # (D,) float32 DINOv2 embedding
    global_object_id: Optional[str] = None
    tracker_backend: str = "sam3"  # which tracker produced this masklet
