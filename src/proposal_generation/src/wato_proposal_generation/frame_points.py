"""One frame's points that any proposal source can claim, as one table.

Every non-ground point enters, including returns neither lifting nor motion
labelled: scan-cluster refinement must be able to recover them. Each row keeps
its (sweep_id, point_index), the key both sources share for fusion. Ground is
excluded using each world sweep's stored Patchwork++ mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.artifacts import (
    ArtifactFormatError,
    lifted_candidates,
    lifted_labels,
    motion_proposals,
)
from wato_common.artifacts.lidar_sweeps import Frame, ProcessedSweeps


@dataclass(frozen=True)
class LiftedSeed:
    """One independent hypothesis, with camera confidence per original return."""

    point_indices: np.ndarray  # int64 (S,) into this frame's point table
    confidences: np.ndarray  # float32 (S,), aligned with point_indices


@dataclass(frozen=True)
class FramePoints:
    sweep_id: np.ndarray  # int64 (P,)
    point_index: np.ndarray  # int64 (P,) into the sweep's world NPZ
    xyz: np.ndarray  # float64 (P, 3) world frame
    instance_id: np.ndarray  # str (P,); "" = no lifted label
    cls: np.ndarray  # str (P,); "" = no lifted label
    confidence: np.ndarray  # float32 (P,); 0 = no lifted label
    cluster_id: np.ndarray  # int64 (P,); -1 = in no gated cluster
    # Sparse independent claims; a return may occur in several seeds. None is
    # only for callers constructing a frame from an exclusive label table.
    lifted_seeds: dict[str, LiftedSeed] | None = None

    def seeds(self) -> dict[str, LiftedSeed]:
        if self.lifted_seeds is not None:
            return self.lifted_seeds
        return {
            str(iid): LiftedSeed(
                np.flatnonzero(self.instance_id == iid),
                self.confidence[self.instance_id == iid],
            )
            for iid in np.unique(self.instance_id[self.instance_id != ""])
        }

    @property
    def n(self) -> int:
        return int(self.sweep_id.shape[0])

    @classmethod
    def empty(cls) -> "FramePoints":
        return cls(
            sweep_id=np.zeros(0, np.int64),
            point_index=np.zeros(0, np.int64),
            xyz=np.zeros((0, 3)),
            instance_id=np.zeros(0, np.str_),
            cls=np.zeros(0, np.str_),
            confidence=np.zeros(0, np.float32),
            cluster_id=np.zeros(0, np.int64),
            lifted_seeds={},
        )


def load(
    bag_id: str,
    chunk_id: str,
    frame: Frame,
    sweeps: ProcessedSweeps,
    gated_cluster_ids: np.ndarray,
) -> FramePoints:
    """All non-ground points and independent seeds, preserving sweep point keys.

    The exclusive labels remain a provisional diagnostic view. The pipeline
    rebuilds motion class evidence from accepted 3D owners; lifted proposals
    use retained candidates, including losing labels.
    Missing candidate files require rerunning lifting; owned labels cannot
    reconstruct their lost hypotheses.
    """
    parts: list[FramePoints] = []
    seed_parts: dict[str, list[LiftedSeed]] = {}
    offset = 0
    for sid in frame.sweep_ids:
        meta = sweeps.get(sid)
        # Both artifacts index the world sweep; n_points_total is its size.
        n_points = meta.n_points_total
        labels = lifted_labels.read_sweep(bag_id, chunk_id, sid, n_points=n_points)
        candidates = lifted_candidates.read_sweep(
            bag_id, chunk_id, sid, n_points=n_points
        )
        clusters = motion_proposals.read_sweep(
            bag_id, chunk_id, sid, n_points=n_points
        ).cluster_id
        world = sweeps.world(sid)
        if world.n_points != n_points:
            raise ArtifactFormatError(
                meta.world_path,
                f"{world.n_points} points but lidar_proc_index says {n_points}",
            )
        if world.ground_mask is None:
            raise ArtifactFormatError(
                meta.world_path,
                "no ground_mask: scan refinement needs Patchwork++ ground; "
                "re-run lidar_preprocessing",
            )
        if (
            world.ground_mask[labels.point_indices].any()
            or world.ground_mask[candidates.point_indices].any()
        ):
            raise ArtifactFormatError(
                meta.world_path,
                "lifted labels include ground points; re-run semantic_lifting",
            )
        idx = np.flatnonzero(~world.ground_mask)
        if idx.size == 0:
            continue

        n = idx.size
        instance_id = np.full(n, "", dtype=labels.instance_ids.dtype)
        cls = np.full(n, "", dtype=labels.classes.dtype)
        confidence = np.zeros(n, np.float32)
        at = np.searchsorted(idx, labels.point_indices)
        instance_id[at] = labels.instance_ids
        cls[at] = labels.classes
        confidence[at] = labels.confidences
        for iid in np.unique(candidates.instance_ids):
            claimed = candidates.instance_ids == iid
            at = np.searchsorted(idx, candidates.point_indices[claimed])
            seed_parts.setdefault(str(iid), []).append(
                LiftedSeed(at + offset, candidates.confidences[claimed])
            )
        cluster_id = clusters[idx].astype(np.int64)
        cluster_id[~np.isin(cluster_id, gated_cluster_ids)] = -1
        parts.append(
            FramePoints(
                sweep_id=np.full(n, sid, np.int64),
                point_index=idx,
                xyz=world.xyz[idx],
                instance_id=instance_id,
                cls=cls,
                confidence=confidence,
                cluster_id=cluster_id,
            )
        )
        offset += n
    if not parts:
        return FramePoints.empty()
    return FramePoints(
        sweep_id=np.concatenate([p.sweep_id for p in parts]),
        point_index=np.concatenate([p.point_index for p in parts]),
        xyz=np.concatenate([p.xyz for p in parts]),
        instance_id=np.concatenate([p.instance_id for p in parts]),
        cls=np.concatenate([p.cls for p in parts]),
        confidence=np.concatenate([p.confidence for p in parts]),
        cluster_id=np.concatenate([p.cluster_id for p in parts]),
        lifted_seeds={
            iid: LiftedSeed(
                np.concatenate([p.point_indices for p in claims]),
                np.concatenate([p.confidences for p in claims]),
            )
            for iid, claims in seed_parts.items()
        },
    )
