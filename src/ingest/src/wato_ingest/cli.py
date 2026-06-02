"""Command line entrypoint for the ingest component.

Primary path:
  python -m wato_ingest run --bag <bag_path>

Debug and rerun commands:
  inspect-bag         Read topics and duration without writing artifacts.
  register            Write bag-level metadata.
  freeze-calibration  Copy authored calibration into the artifact tree.
  split               Compute virtual chunk windows.
  decode-chunk        Decode camera, LiDAR, and pose rows for one chunk.
  build-frame-index   Join LiDAR sweeps to nearest camera frames and poses.
  quality             Compute per-chunk quality metrics and tags.
  manifest            Write per-chunk traceability metadata.
"""

from __future__ import annotations

import json
import logging
import sys

import click

from wato_common.progress import configure_tqdm
from wato_ingest import pipeline
from wato_ingest.artifacts import frame_index, manifest, quality
from wato_ingest.config import load_config
from wato_ingest.decoders import cameras, lidar, poses
from wato_ingest.inputs import bags, calibration, chunks

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

# Single live tqdm line on a terminal, silent when output is captured. See
# wato_common.progress (override with WATO_PROGRESS=on|off).
configure_tqdm()


@click.group()
def main() -> None:
    """Ingest: rosbag ingest for frames, LiDAR, poses, and frame_index."""


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
@click.option(
    "--bag",
    "bag_path",
    default=None,
    type=click.Path(exists=True),
    help="Auto-extract from this bag (CameraInfo + /tf_static).",
)
@click.option(
    "--from-file",
    "source_path",
    default=None,
    type=click.Path(exists=True),
    help="Override: copy this authored calibration JSON in instead.",
)
@click.option("--config", "config_path", default="/ws/src/ingest/config/ingest.yaml")
@click.option("--version", default=None)
def freeze_calibration_cmd(
    bag_id: str,
    bag_path: str | None,
    source_path: str | None,
    config_path: str,
    version: str | None,
) -> None:
    """Build raw/<bag_id>/calibration.json from the bag (default) or a file (override)."""
    if source_path is not None:
        uri = calibration.freeze_from_file(bag_id, source_path, version=version)
    elif bag_path is not None:
        cfg = load_config(config_path)
        uri = calibration.freeze_from_bag(bag_path, bag_id, cfg)
    else:
        raise click.UsageError(
            "provide either --bag (auto-extract) or --from-file (override)"
        )
    click.echo(uri)


# ---------------------------------------------------------------------------
@main.command("split")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--bag-id", required=True)
@click.option("--config", "config_path", default="/ws/src/ingest/config/ingest.yaml")
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
@click.option("--config", "config_path", default="/ws/src/ingest/config/ingest.yaml")
def decode_chunk_cmd(
    bag_path: str, bag_id: str, chunk_id: str, config_path: str
) -> None:
    """Decode cameras + LiDAR + poses for a single chunk."""
    cfg = load_config(config_path)
    chunk = _load_chunk(bag_id, chunk_id)
    if chunk is None:
        click.echo(f"chunk {chunk_id} not found in chunks/index.parquet", err=True)
        sys.exit(2)
    cam_results = cameras.decode_chunk(
        bag_path,
        bag_id,
        chunk_id,
        t_start_ns=int(chunk["t_overlap_start_ns"]),
        t_end_ns=int(chunk["t_overlap_end_ns"]),
        cfg=cfg,
    )
    lid_results = lidar.decode_chunk(
        bag_path,
        bag_id,
        chunk_id,
        t_start_ns=int(chunk["t_overlap_start_ns"]),
        t_end_ns=int(chunk["t_overlap_end_ns"]),
        cfg=cfg,
    )
    pose_result = poses.extract(
        bag_path,
        bag_id,
        chunk_id,
        t_start_ns=int(chunk["t_overlap_start_ns"]),
        t_end_ns=int(chunk["t_overlap_end_ns"]),
        cfg=cfg,
    )
    click.echo(
        json.dumps(
            {
                "cameras": [
                    {"cam_id": r.cam_id, "rows": r.rows_written} for r in cam_results
                ],
                "lidars": [
                    {"lidar_id": r.lidar_id, "sweeps": r.sweeps_written}
                    for r in lid_results
                ],
                "poses": {
                    "rows": pose_result.rows_written,
                    "uri": pose_result.output_uri,
                },
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
@main.command("build-frame-index")
@click.option("--bag-id", required=True)
@click.option("--chunk-id", required=True)
@click.option("--config", "config_path", default="/ws/src/ingest/config/ingest.yaml")
def build_frame_index_cmd(bag_id: str, chunk_id: str, config_path: str) -> None:
    """Match LiDAR sweeps to nearest camera frame per camera and interpolate poses."""
    cfg = load_config(config_path)
    result = frame_index.build(
        bag_id,
        chunk_id,
        max_cam_offset_ms=cfg.max_cam_offset_ms,
    )
    click.echo(
        json.dumps(
            {
                "output": result.output_uri,
                "rows": result.rows_written,
                "valid_camera_rows": result.valid_camera_count,
                "dropped_camera_rows": result.dropped_camera_count,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
@main.command("quality")
@click.option("--bag-id", required=True)
@click.option("--chunk-id", required=True)
@click.option("--config", "config_path", default="/ws/src/ingest/config/ingest.yaml")
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
@click.option("--config", "config_path", default="/ws/src/ingest/config/ingest.yaml")
def manifest_cmd(bag_path: str, bag_id: str, chunk_id: str, config_path: str) -> None:
    """Write the per-chunk manifest.json."""
    uri = manifest.write(
        bag_id=bag_id,
        chunk_id=chunk_id,
        bag_path=bag_path,
        config_path=config_path,
    )
    click.echo(uri)


# ---------------------------------------------------------------------------
@main.command("run")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--bag-id", default=None)
@click.option(
    "--chunk",
    "only_chunk",
    default=None,
    help="Process only this chunk_id (default: all chunks).",
)
@click.option(
    "--calibration",
    "calib_source",
    default=None,
    type=click.Path(),
    help="Path to a pre-authored calibration.json to freeze.",
)
@click.option("--config", "config_path", default="/ws/src/ingest/config/ingest.yaml")
def run_cmd(
    bag_path: str,
    bag_id: str | None,
    only_chunk: str | None,
    calib_source: str | None,
    config_path: str,
) -> None:
    """Full pass: register, validate, chunk, decode, frame-index, quality, manifest."""
    cfg = load_config(config_path)
    results = pipeline.run_bag(
        bag_path=bag_path,
        cfg=cfg,
        config_path=config_path,
        bag_id=bag_id,
        calibration_source=calib_source,
        only_chunk=only_chunk,
    )
    click.echo("")
    click.echo(f"bag: {results[0].bag_id}  ({len(results)} chunk{'s' if len(results) != 1 else ''})")
    click.echo("")
    click.echo(f"  {'chunk':<8}  {'frames':>8}  {'dropped':>7}  tags")
    click.echo(f"  {'-----':<8}  {'------':>8}  {'-------':>7}  ----")
    for r in results:
        total = r.valid_camera_count + r.dropped_camera_count
        tags = ", ".join(r.quality_tags) if r.quality_tags else "ok"
        click.echo(f"  {r.chunk_id:<8}  {r.valid_camera_count:>6}/{total:<6}  {r.dropped_camera_count:>7}  {tags}")
    click.echo("")


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
