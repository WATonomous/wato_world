"""perception_2d masklets: one object tracked through one camera's frames.

Written by perception_2d; read by semantic_lifting (masks) and tracking
(embeddings).

On disk, per chunk:

  tracklets_2d.parquet   one MaskletRow per masklet (schemas.MASKLET_SCHEMA).
                         frames_present is a JSON list[int] string.
  detections_2d.parquet  the same rows; perception_2d's skip check needs both.
  masks_2d/<masklet_id>/<camera_seq:06d>.png
                         one mask per entry of frames_present: 8-bit grayscale
                         at the image's resolution, 255 = object. Independent
                         per-object claims may overlap; no pixel ownership.
  masklet_embeddings.npz one appearance descriptor per (masklet, embedded
                         image), MaskletEmbeddings: unit-norm float16 rows
                         with their masklet, camera, camera_seq and image
                         time, and the model and pooling that made them.
                         Absent when perception_2d ran with embeddings off.

The row's mask_path column records the masklet's directory as the writer saw
it. Readers derive the directory from the artifact root instead, so a tree
read from a different mount point still resolves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import (
    detections_2d_path,
    local_path,
    masklet_embeddings_path,
    masks_2d_dir,
    tracklets_2d_path,
)
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import (
    MASKLET_SCHEMA,
    MaskletRow,
    decode_int_list,
    encode_int_list,
)


@dataclass(frozen=True)
class MaskletRecord:
    """One masklet: everything in its MaskletRow except the derived paths."""

    masklet_id: str
    cam_id: str
    cls: str  # canonical taxonomy class
    score: float
    frames_present: tuple[int, ...]  # camera_seq values with a mask, ascending
    raw_phrase: str = ""  # detector phrase before canonicalisation
    det_score: float = 0.0  # detector (x SAM2 mask) confidence
    discovery_score: float = 0.0
    centroid_depth_m: float = 0.0
    tracker_backend: str = "sam2"
    mask_semantics: str = "independent"


@dataclass(frozen=True)
class MaskletEmbeddings:
    """A chunk's appearance descriptors, one row per (masklet, embedded image).

    A masklet keeps every embedded image's descriptor rather than their mean:
    consumers compare two objects by their closest pair of descriptors
    (DeepSORT's gallery), which a mean over viewpoints can't give. Rows are
    unit L2 norm, so a dot product is the cosine similarity. Only rows of one
    (model, pooling) are comparable.
    """

    model: str  # backbone, e.g. "dinov2_vitl14"
    pooling: str  # how a mask's tokens become one row, e.g. "ffa"
    masklet_ids: np.ndarray  # (N,) str
    cam_ids: np.ndarray  # (N,) str
    camera_seq: np.ndarray  # (N,) int64
    camera_timestamp_ns: np.ndarray  # (N,) int64, the image's time
    features: np.ndarray  # (N, D) float16, unit L2 norm

    def __len__(self) -> int:
        return int(self.features.shape[0])


# ---------------------------------------------------------------------------
# Write half (perception_2d).
# ---------------------------------------------------------------------------


def _masklet_dir(bag_id: str, chunk_id: str, masklet_id: str) -> str:
    return os.path.join(local_path(masks_2d_dir(bag_id, chunk_id)), masklet_id)


def write_mask(
    bag_id: str, chunk_id: str, masklet_id: str, camera_seq: int, mask: np.ndarray
) -> str:
    """Save one frame's (H, W) bool mask; returns its local path."""
    from PIL import Image

    d = _masklet_dir(bag_id, chunk_id, masklet_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{camera_seq:06d}.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return path


_EMBEDDING_ARRAYS = (
    "masklet_ids",
    "cam_ids",
    "camera_seq",
    "camera_timestamp_ns",
    "features",
)
# Unit norm after the float16 round trip.
_UNIT_NORM_TOL = 1e-2


def _check_embeddings(e: MaskletEmbeddings, path: str) -> None:
    n = len(e.masklet_ids)
    if e.features.ndim != 2:
        raise ArtifactFormatError(path, f"features has shape {e.features.shape}")
    for name in _EMBEDDING_ARRAYS:
        if len(getattr(e, name)) != n:
            raise ArtifactFormatError(
                path, f"{name} has {len(getattr(e, name))} rows, masklet_ids {n}"
            )
    if not e.model or not e.pooling:
        raise ArtifactFormatError(path, "embeddings name no model or pooling")
    norms = np.linalg.norm(e.features.astype(np.float32), axis=1)
    if n and not np.all(np.abs(norms - 1.0) <= _UNIT_NORM_TOL):
        raise ArtifactFormatError(path, "features are not unit L2 norm")


def write_embeddings(bag_id: str, chunk_id: str, embeddings: MaskletEmbeddings) -> str:
    """Save the chunk's descriptors (no rows = a chunk with nothing embedded);
    returns the local path. Written through a temporary file, so a reader
    never sees half of it."""
    path = local_path(masklet_embeddings_path(bag_id, chunk_id))
    e = MaskletEmbeddings(
        model=embeddings.model,
        pooling=embeddings.pooling,
        masklet_ids=np.asarray(embeddings.masklet_ids, dtype=str),
        cam_ids=np.asarray(embeddings.cam_ids, dtype=str),
        camera_seq=np.asarray(embeddings.camera_seq, dtype=np.int64),
        camera_timestamp_ns=np.asarray(embeddings.camera_timestamp_ns, dtype=np.int64),
        features=np.asarray(embeddings.features, dtype=np.float16),
    )
    _check_embeddings(e, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.npz"
    np.savez(
        tmp,
        model=np.array(e.model),
        pooling=np.array(e.pooling),
        **{name: getattr(e, name) for name in _EMBEDDING_ARRAYS},
    )
    os.replace(tmp, path)
    return path


def write(bag_id: str, chunk_id: str, records: list[MaskletRecord]) -> None:
    """Write the chunk's masklet tables (empty list = chunk with no masklets)."""
    rows = [
        MaskletRow(
            masklet_id=r.masklet_id,
            bag_id=bag_id,
            chunk_id=chunk_id,
            cam_id=r.cam_id,
            cls=r.cls,
            score=r.score,
            frames_present=encode_int_list(list(r.frames_present)),
            mask_path=_masklet_dir(bag_id, chunk_id, r.masklet_id),
            raw_phrase=r.raw_phrase,
            det_score=r.det_score,
            discovery_score=r.discovery_score,
            centroid_depth_m=r.centroid_depth_m,
            tracker_backend=r.tracker_backend,
            mask_semantics=r.mask_semantics,
        ).model_dump()
        for r in records
    ]
    write_table(rows, MASKLET_SCHEMA, detections_2d_path(bag_id, chunk_id))
    write_table(rows, MASKLET_SCHEMA, tracklets_2d_path(bag_id, chunk_id))


def exists(bag_id: str, chunk_id: str) -> bool:
    """True when both masklet tables were written for this chunk."""
    return os.path.exists(
        local_path(detections_2d_path(bag_id, chunk_id))
    ) and os.path.exists(local_path(tracklets_2d_path(bag_id, chunk_id)))


# ---------------------------------------------------------------------------
# Read half.
# ---------------------------------------------------------------------------


def _tracker_backend(row: dict, path: str) -> str:
    """Which mask tracker made the masklet. Two backends make different
    masks, so a row without one is unreadable, not "sam2"."""
    backend = row.get("tracker_backend")
    if not backend:
        raise ArtifactFormatError(
            path,
            f"masklet {row.get('masklet_id')} has no tracker_backend; rerun "
            "perception_2d",
        )
    return str(backend)


def _record(row: dict, path: str) -> MaskletRecord:
    if row.get("mask_semantics") != "independent":
        raise ArtifactFormatError(
            path,
            f"masklet {row.get('masklet_id')} does not preserve independent "
            "mask claims; re-run perception_2d with --force",
        )
    det_score = row.get("det_score")
    return MaskletRecord(
        masklet_id=str(row["masklet_id"]),
        cam_id=str(row["cam_id"]),
        cls=str(row["cls"]),
        score=float(row["score"]),
        frames_present=tuple(decode_int_list(row["frames_present"])),
        raw_phrase=row.get("raw_phrase") or "",
        # Rows from before det_score existed carry only the masklet score.
        det_score=float(det_score if det_score is not None else row["score"]),
        discovery_score=float(row.get("discovery_score") or 0.0),
        centroid_depth_m=float(row.get("centroid_depth_m") or 0.0),
        tracker_backend=_tracker_backend(row, path),
        mask_semantics=row["mask_semantics"],
    )


def read(bag_id: str, chunk_id: str) -> list[MaskletRecord]:
    """Every masklet in the chunk (from tracklets_2d.parquet)."""
    path = tracklets_2d_path(bag_id, chunk_id)
    return [_record(r, path) for r in read_rows(path)]


def read_mask(
    bag_id: str, chunk_id: str, masklet_id: str, camera_seq: int
) -> np.ndarray:
    """One frame's (H, W) bool mask. Raises if the PNG is missing."""
    from PIL import Image

    path = os.path.join(
        _masklet_dir(bag_id, chunk_id, masklet_id), f"{camera_seq:06d}.png"
    )
    if not os.path.exists(path):
        raise ArtifactFormatError(
            path, f"masklet {masklet_id} lists frame {camera_seq} but has no mask"
        )
    return np.asarray(Image.open(path)) > 0


def embeddings_exist(bag_id: str, chunk_id: str) -> bool:
    return os.path.exists(local_path(masklet_embeddings_path(bag_id, chunk_id)))


def read_embeddings(bag_id: str, chunk_id: str) -> MaskletEmbeddings:
    """The chunk's descriptors. Raises FileNotFoundError when perception_2d
    wrote none, ArtifactFormatError when the file isn't what it writes."""
    path = local_path(masklet_embeddings_path(bag_id, chunk_id))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path}: no masklet embeddings (run perception_2d, or its `embed` "
            "command on a chunk tracked before they existed)"
        )
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ("model", "pooling", *_EMBEDDING_ARRAYS) if k not in z]
        if missing:
            raise ArtifactFormatError(path, f"missing arrays {missing}")
        e = MaskletEmbeddings(
            model=str(z["model"]),
            pooling=str(z["pooling"]),
            **{name: z[name] for name in _EMBEDDING_ARRAYS},
        )
    _check_embeddings(e, path)
    return e


class MaskletIndex:
    """A chunk's masklets indexed by the camera frames they appear in.

    Reads tracklets_2d.parquet once; mask PNGs are read on request.
    """

    def __init__(self, bag_id: str, chunk_id: str, records: list[MaskletRecord]):
        self.bag_id = bag_id
        self.chunk_id = chunk_id
        self.records = records
        self._by_frame: dict[tuple[str, int], list[MaskletRecord]] = {}
        for r in records:
            for seq in r.frames_present:
                self._by_frame.setdefault((r.cam_id, seq), []).append(r)

    @classmethod
    def load(cls, bag_id: str, chunk_id: str) -> "MaskletIndex":
        return cls(bag_id, chunk_id, read(bag_id, chunk_id))

    def at(self, cam_id: str, camera_seq: int) -> list[MaskletRecord]:
        """Masklets with a mask in this camera frame, in table order."""
        return self._by_frame.get((cam_id, camera_seq), [])

    def mask(self, record: MaskletRecord, camera_seq: int) -> np.ndarray:
        return read_mask(self.bag_id, self.chunk_id, record.masklet_id, camera_seq)
