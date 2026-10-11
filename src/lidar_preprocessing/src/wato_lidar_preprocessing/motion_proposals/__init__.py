"""Step F — recall-oriented moving-object proposals (see _core.py).

The bit layout (AW_DYNAMIC … UNMAPPED, SEED_BITS, ATTACH_BITS, decode_bits) and
the artifacts' read/write halves live in wato_common.artifacts.motion_proposals.
"""

from wato_lidar_preprocessing.motion_proposals._core import (
    BOX_FILL_MAX_PERSISTENT,
    BOX_FILL_MIN_TRACK_LIFE,
    MOVING_SCORE,
    ProposalResult,
    cluster_points,
    compute_sweep_bits,
    fill_box,
    process_chunk,
    proposals_up_to_date,
    should_box_fill,
    track_motion,
)

__all__ = [
    "BOX_FILL_MAX_PERSISTENT",
    "BOX_FILL_MIN_TRACK_LIFE",
    "MOVING_SCORE",
    "ProposalResult",
    "cluster_points",
    "compute_sweep_bits",
    "fill_box",
    "process_chunk",
    "proposals_up_to_date",
    "should_box_fill",
    "track_motion",
]
