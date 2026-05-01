"""Ingest CLI. Subcommands match the architecture document's recipe.

  watod run ingest <bag>               → full pass
  python -m wato_ingest inspect-bag    → dump topics + duration
  python -m wato_ingest split          → write chunks/index.parquet
  python -m wato_ingest decode-chunk   → cameras + LiDAR + poses for one chunk
  python -m wato_ingest build-frame-index
  python -m wato_ingest quality
  python -m wato_ingest freeze-calibration
  python -m wato_ingest run            → full pass (register + chunk + per-chunk decode/index/quality)
"""

from __future__ import annotations

import json
import logging
import sys

import click

from wato_ingest import runner
from wato_ingest.artifacts import frame_index, manifest, quality
from wato_ingest.config import load_config
from wato_ingest.decoders import cameras, lidar, poses
from wato_ingest.inputs import bags, calibration, chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
def main() -> None:
    """Ingest — rosbag ingest (frames, LiDAR, poses, frame_index)."""


# ---------------------------------------------------------------------------
@main.command("inspect-bag")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--storage-id", default="sqlite3")
def inspect_bag(bag_path: str, storage_id: str) -> None:
    """Print the bag's topics + duration without writing anything."""
    from wato_common.io.rosbag_reader import summarize
    summary = summarize(bag_path, storage_id=storage_id)
    out = {
        "starting_time_ns": summary.starting_time_ns,
        "duration_s": summary.duration_ns / 1e9,
        "message_count": summary.message_count,
        "topics": [
            {"name": t.name, "type": t.type, "messages": t.message_count}
            for t in summary.topics
        ],
    }
    click.echo(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
@main.command("register")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--bag-id", default=None)
@click.option("--storage-id", default="sqlite3")
def register_cmd(bag_path: str, bag_id: str | None, storage_id: str) -> None:
    """Inspect + write bag_meta.json under raw/<bag_id>/."""
    meta = bags.register(bag_path, bag_id=bag_id, storage_id=storage_id)
    click.echo(json.dumps(meta.model_dump(), indent=2))


# ---------------------------------------------------------------------------
@main.command("freeze-calibration")
@click.option("--bag-id", required=True)
@click.option("--from-file", "source_path", required=True, type=click.Path(exists=True))
@click.option("--version", default=None)
def freeze_calibration_cmd(bag_id: str, source_path: str, version: str | None) -> None:
    """Copy an authored calibration JSON into raw/<bag_id>/calibration.json."""
    uri = calibration.freeze_from_file(bag_id, source_path, version=version)
    click.echo(uri)


# ---------------------------------------------------------------------------
@main.command("split")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--bag-id", required=True)
@click.option("--config", "config_path", default="/config/pipeline.yaml")
def split_cmd(bag_path: str, bag_id: str, config_path: str) -> None:
    """Compute virtual chunks and write chunks/index.parquet."""
    cfg = load_config(config_path)
    chunk_rows = chunks.compute_chunks(bag_path, bag_id, cfg)
    uri = chunks.write_chunk_index(bag_id, chunk_rows)
    click.echo(json.dumps({"output": uri, "chunks": len(chunk_rows)}, indent=2))


# ---------------------------------------------------------------------------
@main.command("decode-chunk")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--bag-id", required=True)
@click.option("--chunk-id", required=True)
@click.option("--config", "config_path", default="/config/pipeline.yaml")
def decode_chunk_cmd(bag_path: str, bag_id: str, chunk_id: str, config_path: str) -> None:
    """Decode cameras + LiDAR + poses for a single chunk."""
    cfg = load_config(config_path)
    chunk = _load_chunk(bag_id, chunk_id)
    if chunk is None:
        click.echo(f"chunk {chunk_id} not found in chunks/index.parquet", err=True)
        sys.exit(2)
    cam_results = cameras.decode_chunk(
        bag_path, bag_id, chunk_id,
        t_start_ns=int(chunk["t_overlap_start_ns"]),
        t_end_ns=int(chunk["t_overlap_end_ns"]),
        cfg=cfg,
    )
    lid_results = lidar.decode_chunk(
        bag_path, bag_id, chunk_id,
        t_start_ns=int(chunk["t_overlap_start_ns"]),
        t_end_ns=int(chunk["t_overlap_end_ns"]),
        cfg=cfg,
    )
    pose_result = poses.extract(
        bag_path, bag_id, chunk_id,
        t_start_ns=int(chunk["t_overlap_start_ns"]),
        t_end_ns=int(chunk["t_overlap_end_ns"]),
        cfg=cfg,
    )
    click.echo(json.dumps({
        "cameras": [{"cam_id": r.cam_id, "rows": r.rows_written} for r in cam_results],
        "lidars":  [{"lidar_id": r.lidar_id, "sweeps": r.sweeps_written} for r in lid_results],
        "poses":   {"rows": pose_result.rows_written, "uri": pose_result.output_uri},
    }, indent=2))


# ---------------------------------------------------------------------------
@main.command("build-frame-index")
@click.option("--bag-id", required=True)
@click.option("--chunk-id", required=True)
@click.option("--config", "config_path", default="/config/pipeline.yaml")
def build_frame_index_cmd(bag_id: str, chunk_id: str, config_path: str) -> None:
    """Match LiDAR sweeps ↔ nearest camera frame per camera + interpolate poses."""
    cfg = load_config(config_path)
    result = frame_index.build(
        bag_id, chunk_id, max_cam_offset_ms=cfg.max_cam_offset_ms,
    )
    click.echo(json.dumps({
        "output": result.output_uri,
        "rows": result.rows_written,
        "valid_camera_rows": result.valid_camera_count,
        "dropped_camera_rows": result.dropped_camera_count,
    }, indent=2))


# ---------------------------------------------------------------------------
@main.command("quality")
@click.option("--bag-id", required=True)
@click.option("--chunk-id", required=True)
@click.option("--config", "config_path", default="/config/pipeline.yaml")
def quality_cmd(bag_id: str, chunk_id: str, config_path: str) -> None:
    """Compute per-chunk metrics + tags from the existing artifacts."""
    cfg = load_config(config_path)
    report = quality.compute(bag_id, chunk_id, cfg)
    click.echo(json.dumps({"tags": report.tags, "metrics": report.metrics}, indent=2))


# ---------------------------------------------------------------------------
@main.command("manifest")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--bag-id", required=True)
@click.option("--chunk-id", required=True)
@click.option("--config", "config_path", default="/config/pipeline.yaml")
def manifest_cmd(bag_path: str, bag_id: str, chunk_id: str, config_path: str) -> None:
    """Write the per-chunk manifest.json."""
    uri = manifest.write(
        bag_id=bag_id, chunk_id=chunk_id, bag_path=bag_path, config_path=config_path,
    )
    click.echo(uri)


# ---------------------------------------------------------------------------
@main.command("run")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--bag-id", default=None)
@click.option("--chunk", "only_chunk", default=None,
              help="Process only this chunk_id (default: all chunks).")
@click.option("--calibration", "calib_source", default=None, type=click.Path(),
              help="Path to a pre-authored calibration.json to freeze.")
@click.option("--config", "config_path", default="/config/pipeline.yaml")
def run_cmd(
    bag_path: str,
    bag_id: str | None,
    only_chunk: str | None,
    calib_source: str | None,
    config_path: str,
) -> None:
    """Full pass: register, validate, chunk, decode, frame-index, quality, manifest."""
    cfg = load_config(config_path)
    results = runner.run_bag(
        bag_path=bag_path,
        cfg=cfg,
        config_path=config_path,
        bag_id=bag_id,
        calibration_source=calib_source,
        only_chunk=only_chunk,
    )
    click.echo(json.dumps([
        {
            "bag_id": r.bag_id,
            "chunk_id": r.chunk_id,
            "quality_tags": r.quality_tags,
            "valid_camera_rows": r.valid_camera_count,
            "dropped_camera_rows": r.dropped_camera_count,
        } for r in results
    ], indent=2))


def _load_chunk(bag_id: str, chunk_id: str) -> dict | None:
    from wato_common.artifact_store import chunks_index_path
    from wato_common.io.parquet_io import read_rows
    rows = read_rows(chunks_index_path(bag_id))
    for r in rows:
        if r["chunk_id"] == chunk_id:
            return r
    return None


if __name__ == "__main__":
    main()
