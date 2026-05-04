"""Ingest component pipeline.

The pipeline owns orchestration. Leaf modules under `inputs`, `decoders`, and
`artifacts` own one kind of IO or artifact transformation each.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from wato_ingest.artifacts import frame_index, manifest, quality
from wato_ingest.config import IngestConfig
from wato_ingest.decoders import cameras, lidar, poses
from wato_ingest.inputs import bags, calibration, chunks, topics

log = logging.getLogger(__name__)


@dataclass
class ChunkRunResult:
    bag_id: str
    chunk_id: str
    quality_tags: list[str]
    valid_camera_count: int
    dropped_camera_count: int


def run_bag(
    *,
    bag_path: str,
    cfg: IngestConfig,
    config_path: str,
    bag_id: str | None = None,
    calibration_source: str | None = None,
    only_chunk: str | None = None,
) -> list[ChunkRunResult]:
    """Run ingest end-to-end for a single bag.

    Steps:
      1. Register the bag and write `bag_meta.json`.
      2. Validate required camera, LiDAR, and pose topics.
      3. Freeze calibration when a calibration source is supplied.
      4. Compute virtual chunks and write `chunks/index.parquet`.
      5. For each selected chunk, decode sensor data, build the frame index,
         compute quality metrics, and write the manifest.
    """
    meta = bags.register(bag_path, bag_id=bag_id, storage_id=cfg.storage_id)
    bag_id = meta.bag_id
    log.info("registered bag_id=%s duration=%.1fs", bag_id, meta.duration_s)

    topic_check = topics.validate(
        {t: "" for t in meta.topics}, cfg
    )
    if not topic_check.ok:
        raise RuntimeError(
            f"bag {bag_id} missing topics required by ingest: {topic_check.missing}"
        )

    # Calibration: prefer auto-extraction from the bag's own CameraInfo +
    # /tf_static.  Allow a hand-authored JSON to override when the bag is
    # missing one of those (or when the extracted values are known wrong).
    if calibration_source is not None:
        calib_uri = calibration.freeze_from_file(bag_id, calibration_source)
        log.info("calibration: froze from authored file -> %s", calib_uri)
    else:
        calib_uri = calibration.freeze_from_bag(bag_path, bag_id, cfg)
        log.info("calibration: auto-extracted from bag -> %s", calib_uri)

    chunk_rows = chunks.compute_chunks(bag_path, bag_id, cfg)
    chunks.write_chunk_index(bag_id, chunk_rows)
    log.info("computed %d chunks", len(chunk_rows))

    if only_chunk:
        chunk_rows = [c for c in chunk_rows if c.chunk_id == only_chunk]
        if not chunk_rows:
            raise ValueError(f"chunk_id {only_chunk} not found")

    results: list[ChunkRunResult] = []
    for c in chunk_rows:
        log.info("processing chunk %s", c.chunk_id)
        cameras.decode_chunk(
            bag_path, bag_id, c.chunk_id,
            t_start_ns=c.t_overlap_start_ns, t_end_ns=c.t_overlap_end_ns, cfg=cfg,
        )
        lidar.decode_chunk(
            bag_path, bag_id, c.chunk_id,
            t_start_ns=c.t_overlap_start_ns, t_end_ns=c.t_overlap_end_ns, cfg=cfg,
        )
        poses.extract(
            bag_path, bag_id, c.chunk_id,
            t_start_ns=c.t_overlap_start_ns, t_end_ns=c.t_overlap_end_ns, cfg=cfg,
        )

        frame_index_result = frame_index.build(
            bag_id, c.chunk_id, max_cam_offset_ms=cfg.max_cam_offset_ms,
        )
        report = quality.compute(bag_id, c.chunk_id, cfg)

        manifest.write(
            bag_id=bag_id,
            chunk_id=c.chunk_id,
            bag_path=bag_path,
            config_path=config_path,
            extra={
                "topic_check": {
                    "found_camera_images": topic_check.found_camera_image_topics,
                    "found_camera_infos":  topic_check.found_camera_info_topics,
                    "found_lidars":        topic_check.found_lidar_topics,
                    "found_pose":          topic_check.found_pose_topics,
                    "found_tf_static":     topic_check.found_tf_static_topics,
                },
                "frame_index_summary": {
                    "rows": frame_index_result.rows_written,
                    "valid_camera_rows": frame_index_result.valid_camera_count,
                    "dropped_camera_rows": frame_index_result.dropped_camera_count,
                },
            },
        )

        results.append(ChunkRunResult(
            bag_id=bag_id,
            chunk_id=c.chunk_id,
            quality_tags=report.tags,
            valid_camera_count=frame_index_result.valid_camera_count,
            dropped_camera_count=frame_index_result.dropped_camera_count,
        ))

    return results


def run(cfg: IngestConfig, *, bag_id: str, chunk_id: str | None = None) -> None:
    """Compatibility shim for older imports.

    Ingest needs a filesystem path for the input bag, so callers should use
    `run_bag(...)` or `python -m wato_ingest run --bag <path>` instead.
    """
    raise NotImplementedError(
        "Ingest needs the bag's filesystem path; call run_bag(...) "
        "or use `python -m wato_ingest run --bag <path>` instead."
    )


__all__ = ["ChunkRunResult", "run", "run_bag"]
