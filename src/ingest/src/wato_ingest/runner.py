"""Top-level ingest orchestration. Composes the modules into one pass."""

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
      1. Register the bag (writes bag_meta.json).
      2. Validate required topics exist.
      3. Freeze calibration (must be supplied via `calibration_source` until
         the in-bag calibration extractor is built).
      4. Compute virtual chunks and write chunks/index.parquet.
      5. For each chunk: decode cameras, decode LiDAR, extract poses, build
         frame_index, compute quality, write manifest.
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

    if calibration_source is None:
        log.warning("no calibration_source provided; skipping calibration freeze")
    else:
        calib_uri = calibration.freeze_from_file(bag_id, calibration_source)
        log.info("froze calibration -> %s", calib_uri)

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
                    "found_cameras": topic_check.found_camera_topics,
                    "found_lidars": topic_check.found_lidar_topics,
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
