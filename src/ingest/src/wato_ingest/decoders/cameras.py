"""Decode camera image messages from a rosbag chunk into JPEG/PNG files.

Strategy:
- For CompressedImage messages: write the original compressed bytes verbatim.
  No re-encoding — preserves quality and skips the decode/encode round-trip.
- For raw Image messages: decode with numpy + Pillow and write a JPEG.

Writes one row per emitted image to `camera_frames.parquet` so perception_2d can
read directly from disk without touching the bag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

from wato_common.artifact_store import (
    camera_dir,
    camera_frames_path,
    camera_image_path,
    ensure_local_dir,
    local_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.io.rosbag_reader import messages
from wato_common.schemas import CAMERA_FRAMES_SCHEMA, CameraFrameRow
from wato_ingest.config import IngestConfig


@dataclass
class CameraDecodeResult:
    cam_id: str
    rows_written: int
    output_dir: str


def _header_ts_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _ext_for_format(fmt: str) -> str:
    """Map sensor_msgs/CompressedImage.format to a file extension."""
    f = (fmt or "").lower()
    if "jpeg" in f or "jpg" in f:
        return "jpg"
    if "png" in f:
        return "png"
    log.warning(
        "CompressedImage.format %r is not jpeg/png — writing raw bytes as .bin; "
        "downstream perception will not be able to open this file",
        fmt,
    )
    return "bin"


def decode_chunk(
    bag_path: str,
    bag_id: str,
    chunk_id: str,
    *,
    t_start_ns: int,
    t_end_ns: int,
    cfg: IngestConfig,
) -> list[CameraDecodeResult]:
    """Decode every configured camera image topic for a chunk's time range."""
    # Logical name -> image topic (CameraInfo is handled at bag scope by
    # inputs/calibration.py, not here).
    cam_topics = {cam_id: c.image for cam_id, c in cfg.topics.cameras.items()}
    topic_to_cam = {v: k for k, v in cam_topics.items()}
    if not topic_to_cam:
        return []

    # Pre-create output directories.
    for cam_id in cam_topics:
        ensure_local_dir(camera_dir(bag_id, chunk_id, cam_id))

    seq_per_cam: dict[str, int] = {cam: 0 for cam in cam_topics}
    rows: list[dict] = []
    written_per_cam: dict[str, int] = {cam: 0 for cam in cam_topics}

    with messages(
        bag_path,
        storage_id=cfg.storage_id,
        topics=list(topic_to_cam.keys()),
        t_start_ns=t_start_ns,
        t_end_ns=t_end_ns,
    ) as iterator:
        for topic, msg, record_ts_ns in iterator:
            cam_id = topic_to_cam[topic]
            row, written = _write_one_image(
                msg=msg,
                bag_id=bag_id,
                chunk_id=chunk_id,
                cam_id=cam_id,
                seq=seq_per_cam[cam_id],
                record_ts_ns=record_ts_ns,
            )
            if row is not None:
                rows.append(row)
            if written:
                seq_per_cam[cam_id] += 1
                written_per_cam[cam_id] += 1

    write_table(rows, CAMERA_FRAMES_SCHEMA, camera_frames_path(bag_id, chunk_id))

    return [
        CameraDecodeResult(
            cam_id=cam_id,
            rows_written=written_per_cam[cam_id],
            output_dir=camera_dir(bag_id, chunk_id, cam_id),
        )
        for cam_id in cam_topics
    ]


def _dims_from_compressed(data: bytes) -> tuple[int, int]:
    """Read (width, height) from compressed image bytes without full decode."""
    try:
        from io import BytesIO

        from PIL import Image as _PILImage

        with _PILImage.open(BytesIO(data)) as img:
            return img.size
    except Exception:
        return 0, 0


def _write_one_image(
    *,
    msg,
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    seq: int,
    record_ts_ns: int,
) -> tuple[dict | None, bool]:
    """Returns (row, written?).  Row is included even on drops (with reason)."""
    msg_type = type(msg).__name__
    is_compressed = msg_type == "CompressedImage"

    try:
        header_ts = _header_ts_ns(msg.header)
    except AttributeError:
        return None, False

    if is_compressed:
        ext = _ext_for_format(getattr(msg, "format", ""))
        out_uri = camera_image_path(bag_id, chunk_id, cam_id, seq, ext)
        img_bytes = bytes(msg.data)
        with open(local_path(out_uri), "wb") as fh:
            fh.write(img_bytes)
        width, height = _dims_from_compressed(img_bytes)
        row = CameraFrameRow(
            bag_id=bag_id,
            chunk_id=chunk_id,
            cam_id=cam_id,
            camera_seq=seq,
            image_path=out_uri,
            header_timestamp_ns=header_ts,
            record_timestamp_ns=record_ts_ns,
            width=width,
            height=height,
            encoding=getattr(msg, "format", "compressed"),
            is_compressed=True,
            valid=True,
        )
        return row.model_dump(), True

    # Raw Image — minimal path.  Re-encode to PNG to preserve precision.
    if msg_type == "Image":
        try:
            import numpy as np
            from PIL import Image
        except ImportError as e:
            raise RuntimeError("Pillow + numpy required for raw Image decode") from e

        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
            msg.height, msg.step
        )
        # Strip row padding if step > width * channels.
        bytes_per_pixel = max(msg.step // max(msg.width, 1), 1)
        arr = arr[:, : msg.width * bytes_per_pixel]
        if (
            "rgb" in (msg.encoding or "").lower()
            or "bgr" in (msg.encoding or "").lower()
        ):
            arr = arr.reshape(msg.height, msg.width, 3)
        else:
            arr = arr.reshape(msg.height, msg.width)
        out_uri = camera_image_path(bag_id, chunk_id, cam_id, seq, "png")
        Image.fromarray(arr).save(local_path(out_uri), format="PNG")
        row = CameraFrameRow(
            bag_id=bag_id,
            chunk_id=chunk_id,
            cam_id=cam_id,
            camera_seq=seq,
            image_path=out_uri,
            header_timestamp_ns=header_ts,
            record_timestamp_ns=record_ts_ns,
            width=msg.width,
            height=msg.height,
            encoding=msg.encoding or "raw",
            is_compressed=False,
            valid=True,
        )
        return row.model_dump(), True

    return None, False
