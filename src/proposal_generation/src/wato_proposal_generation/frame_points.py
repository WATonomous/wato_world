"""One frame's points that any proposal source can claim, as one table.

A point enters when semantic_lifting labelled it or it belongs to a gated
Step F cluster; every other point of the frame's sweeps is left out, so the
world sweeps are read only for frames with something to box. Each row keeps
its (sweep_id, point_index), the key both sources share: fusion matches boxes
on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wato_common.artifacts import ArtifactFormatError, lifted_labels, motion_proposals
from wato_common.artifacts.lidar_sweeps import Frame, ProcessedSweeps


@dataclass(frozen=True)
class FramePoints:
    sweep_id: np.ndarray  # int64 (P,)
    point_index: np.ndarray  # int64 (P,) into the sweep's world NPZ
    xyz: np.ndarray  # float64 (P, 3) world frame
    instance_id: np.ndarray  # str (P,); "" = no lifted label
    cls: np.ndarray  # str (P,); "" = no lifted label
    confidence: np.ndarray  # float32 (P,); 0 = no lifted label
    cluster_id: np.ndarray  # int64 (P,); -1 = in no gated cluster

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
        )


def load(
    bag_id: str,
    chunk_id: str,
    frame: Frame,
    sweeps: ProcessedSweeps,
    gated_cluster_ids: np.ndarray,
) -> FramePoints:
    """The frame's labelled and gated-cluster points, across all its sweeps."""
    parts: list[FramePoints] = []
    for sid in frame.sweep_ids:
        meta = sweeps.get(sid)
        # Both artifacts index the world sweep; n_points_total is its size.
        n_points = meta.n_points_total
        labels = lifted_labels.read_sweep(bag_id, chunk_id, sid, n_points=n_points)
        clusters = motion_proposals.read_sweep(
            bag_id, chunk_id, sid, n_points=n_points
        ).cluster_id
        in_gated = np.flatnonzero(np.isin(clusters, gated_cluster_ids))
        idx = np.union1d(labels.point_indices.astype(np.int64), in_gated)
        if idx.size == 0:
            continue
        world = sweeps.world(sid)
        if world.n_points != n_points:
            raise ArtifactFormatError(
                meta.world_path,
                f"{world.n_points} points but lidar_proc_index says {n_points}",
            )

        n = idx.size
        instance_id = np.full(n, "", dtype=labels.instance_ids.dtype)
        cls = np.full(n, "", dtype=labels.classes.dtype)
        confidence = np.zeros(n, np.float32)
        at = np.searchsorted(idx, labels.point_indices)
        instance_id[at] = labels.instance_ids
        cls[at] = labels.classes
        confidence[at] = labels.confidences
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
    )
