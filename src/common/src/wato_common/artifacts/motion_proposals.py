"""lidar_preprocessing Step F motion proposals — the recall artifact.

Written by lidar_preprocessing (`motion_proposals/`); read by its viz today and
by proposal_generation and tracking once they exist. False positives are by
design: gate on the cluster features, and never use this where a point must be
trusted static (that is `dynamic_mask.npy`'s job).

On disk, per chunk:

  lidar_proc/<sweep_id:06d>_motion_proposals.npz
      source_bits  uint8 (N,)  which heuristics flagged each point (bits below)
      cluster_id   int32 (N,)  cluster in motion_clusters.parquet, -1 = none
      N and order are the sweep's world NPZ (lidar_proc/<sweep_id:06d>_world.npz).
  motion_clusters.parquet
      one MotionClusterRow per cluster (schemas.MOTION_CLUSTER_SCHEMA);
      sweep_ids is a JSON list[int] string on disk, a tuple once read.

source_bits (wato_lidar_preprocessing.motion_proposals documents how each is
computed). Seed bits start clusters; attach bits only join one.

  bit  name           role
  0    AW_DYNAMIC     seed
  1    AW_AMBIGUOUS   attach
  2    IWU_EVICTED    seed
  3    MF_MOS         seed
  4    SEG_DYNAMIC    seed
  5    BOX_FILL       -      inside the box of a moving cluster
  6    UNMAPPED       attach
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Optional

import numpy as np

from wato_common.artifact_store import (
    local_path,
    motion_clusters_path,
    motion_proposals_path,
)
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import (
    MOTION_CLUSTER_SCHEMA,
    MotionClusterRow,
    decode_int_list,
    encode_int_list,
)

AW_DYNAMIC = np.uint8(1 << 0)
AW_AMBIGUOUS = np.uint8(1 << 1)
IWU_EVICTED = np.uint8(1 << 2)
MF_MOS = np.uint8(1 << 3)
SEG_DYNAMIC = np.uint8(1 << 4)
BOX_FILL = np.uint8(1 << 5)
UNMAPPED = np.uint8(1 << 6)

SEED_BITS = np.uint8(AW_DYNAMIC | IWU_EVICTED | MF_MOS | SEG_DYNAMIC)
ATTACH_BITS = np.uint8(AW_AMBIGUOUS | UNMAPPED)

BIT_NAMES: dict[int, str] = {
    int(AW_DYNAMIC): "AW_DYNAMIC",
    int(AW_AMBIGUOUS): "AW_AMBIGUOUS",
    int(IWU_EVICTED): "IWU_EVICTED",
    int(MF_MOS): "MF_MOS",
    int(SEG_DYNAMIC): "SEG_DYNAMIC",
    int(BOX_FILL): "BOX_FILL",
    int(UNMAPPED): "UNMAPPED",
}


def decode_bits(bits: np.ndarray) -> dict[str, np.ndarray]:
    """{bit name: bool mask} over a source_bits array."""
    return {name: (bits & np.uint8(v)) != 0 for v, name in BIT_NAMES.items()}


# ---------------------------------------------------------------------------
# Per-sweep point verdicts.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepProposals:
    source_bits: np.ndarray  # uint8 (N,)
    cluster_id: np.ndarray  # int32 (N,), -1 = no cluster


def write_sweep(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    source_bits: np.ndarray,
    cluster_id: np.ndarray,
) -> None:
    path = local_path(motion_proposals_path(bag_id, chunk_id, sweep_id))
    if source_bits.shape != cluster_id.shape or source_bits.ndim != 1:
        raise ValueError(
            f"{path}: source_bits {source_bits.shape} and cluster_id "
            f"{cluster_id.shape} must be the same (N,)"
        )
    np.savez_compressed(
        path,
        source_bits=source_bits.astype(np.uint8),
        cluster_id=cluster_id.astype(np.int32),
    )


def remove_sweep(bag_id: str, chunk_id: str, sweep_id: int) -> None:
    """Delete a sweep's proposals (left over from a run where it was valid)."""
    path = local_path(motion_proposals_path(bag_id, chunk_id, sweep_id))
    if os.path.exists(path):
        os.remove(path)


def read_sweep(
    bag_id: str, chunk_id: str, sweep_id: int, *, n_points: Optional[int] = None
) -> SweepProposals:
    """One sweep's proposals. FileNotFoundError when the sweep has none.

    Pass the world sweep's point count as n_points to check the alignment.
    """
    path = local_path(motion_proposals_path(bag_id, chunk_id, sweep_id))
    with np.load(path) as d:
        for key in ("source_bits", "cluster_id"):
            if key not in d:
                raise ArtifactFormatError(path, f"no {key!r} array")
        bits, cid = d["source_bits"], d["cluster_id"]
    if bits.dtype != np.uint8 or cid.dtype != np.int32 or bits.shape != cid.shape:
        raise ArtifactFormatError(
            path,
            f"source_bits {bits.dtype}{bits.shape} / cluster_id "
            f"{cid.dtype}{cid.shape} are not uint8/int32 of one length",
        )
    if n_points is not None and bits.shape[0] != n_points:
        raise ArtifactFormatError(
            path, f"{bits.shape[0]} points but the world sweep has {n_points}"
        )
    return SweepProposals(source_bits=bits, cluster_id=cid)


# ---------------------------------------------------------------------------
# Per-chunk clusters.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionCluster:
    """A MotionClusterRow without bag/chunk ids, sweep_ids decoded.

    Field meanings are documented on schemas.MotionClusterRow.
    track_hint_id is chunk-local and geometry-only — never an identity.
    """

    frame_id: int
    reference_timestamp_ns: int
    sweep_ids: tuple[int, ...]
    cluster_id: int
    track_hint_id: int
    cx: float
    cy: float
    cz: float
    w: float
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
    h: float
    heading: float
    n_points: int
    n_seed_points: int
    source_bits: int
    n_sources: int
    frac_seg_dynamic: float
    frac_persistent: float
    track_life: int
    net_displacement_m: float
    motion_score: float
    box_filled: bool


def write_clusters(bag_id: str, chunk_id: str, clusters: list[MotionCluster]) -> None:
    rows = []
    for c in clusters:
        row = asdict(c)
        row["sweep_ids"] = encode_int_list(list(c.sweep_ids))
        rows.append(
            MotionClusterRow(bag_id=bag_id, chunk_id=chunk_id, **row).model_dump()
        )
    write_table(rows, MOTION_CLUSTER_SCHEMA, motion_clusters_path(bag_id, chunk_id))


def read_clusters(bag_id: str, chunk_id: str) -> list[MotionCluster]:
    out = []
    for row in read_rows(motion_clusters_path(bag_id, chunk_id)):
        values = {f.name: row[f.name] for f in fields(MotionCluster)}
        values["sweep_ids"] = tuple(decode_int_list(row["sweep_ids"]))
        out.append(MotionCluster(**values))
    return out
