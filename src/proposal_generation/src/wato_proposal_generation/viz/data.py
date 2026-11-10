"""What every viewer draws: the chunk's frames, the proposals shown in each,
and their colours."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import chunk_root, local_path
from wato_common.artifacts import proposals as proposals_artifact
from wato_common.artifacts.lidar_sweeps import Frame, ProcessedSweeps
from wato_common.artifacts.proposals import FUSED, Proposal
from wato_proposal_generation.config import ComponentConfig

SHOW = ("sources", "fused")

# Sources, when the source rows are shown.
PROVENANCE_RGB = {
    "lidar_mos": (255, 140, 0),
    "lifted": (0, 200, 255),
    "fusion_detector": (190, 90, 255),
    "lidar_detector": (255, 230, 0),
}
# Classes, when the fused rows are shown (semantic_lifting's viewer palette).
CLASS_RGB = {
    "car": (66, 135, 245),
    "truck": (245, 160, 40),
    "bus": (200, 80, 200),
    "pedestrian": (232, 77, 61),
    "bicycle": (80, 210, 120),
    "motorcycle": (240, 210, 60),
}
OTHER_RGB = (200, 200, 200)  # a class outside the palette, or no class
POINT_RGB = (120, 126, 136)


def color(p: Proposal, show: str) -> tuple[int, int, int]:
    if show == "fused":
        return CLASS_RGB.get(p.cls or "", OTHER_RGB)
    return PROVENANCE_RGB.get(p.provenance, OTHER_RGB)


def label(p: Proposal, show: str) -> str:
    """Short text drawn beside a box."""
    cls = p.cls or "?"
    if show == "fused":
        return f"{cls} x{len(p.member_proposal_ids)}"
    return f"{'mos' if p.provenance == 'lidar_mos' else p.provenance} {cls}"


def legend(show: str) -> list[tuple[str, tuple[int, int, int]]]:
    if show == "fused":
        return list(CLASS_RGB.items()) + [("none / other", OTHER_RGB)]
    return [
        (k, v)
        for k, v in PROVENANCE_RGB.items()
        if k in ("lidar_mos", "lifted", "fusion_detector")
    ]


@dataclass(frozen=True)
class ChunkView:
    frames: list[Frame]
    sweeps: ProcessedSweeps
    by_frame: dict[int, list[Proposal]]  # reference sweep_id -> shown rows

    def frame_points(self, frame: Frame) -> np.ndarray:
        """The frame's non-ground world points, (N, 3)."""
        parts = []
        for sid in frame.sweep_ids:
            world = self.sweeps.world(sid)
            xyz = world.xyz
            if world.ground_mask is not None:
                xyz = xyz[~world.ground_mask]
            parts.append(xyz)
        return np.concatenate(parts) if parts else np.zeros((0, 3))


def load(
    cfg: ComponentConfig, bag_id: str, chunk_id: str, show: str, every: int = 1
) -> ChunkView:
    """Every `every`-th frame and the proposals `show` selects in it."""
    if show not in SHOW:
        raise ValueError(f"show must be one of {SHOW}, not {show!r}")
    rows = proposals_artifact.read(bag_id, chunk_id)
    sweeps = ProcessedSweeps.load(bag_id, chunk_id)
    frames = sweeps.frames(cfg.frames.canonical_lidar)[:: max(1, every)]
    by_frame: dict[int, list[Proposal]] = {}
    for p in rows:
        if (p.provenance == FUSED) == (show == "fused"):
            by_frame.setdefault(p.sweep_id, []).append(p)
    return ChunkView(frames, sweeps, by_frame)


def boxes_in(view: ChunkView, frame: Frame) -> list[Proposal]:
    return view.by_frame.get(frame.reference_sweep_id, [])


def default_dir(bag_id: str, chunk_id: str, name: str) -> str:
    """<chunk>/viz/<name>, where every component's viewers write."""
    return f"{local_path(chunk_root(bag_id, chunk_id))}/viz/{name}"
