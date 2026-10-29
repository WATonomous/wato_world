"""Command line entrypoint for the ingest component.

Primary path:
  python -m wato_ingest run --bag <bag_path>

Bag preparation:
  validate-bag        Report what each topic is missing (coverage, gaps,
                      header skew, camera_info pairing, extrinsics, profile).
  trim-bag            Cut an MCAP bag to the window every topic records in.

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

import dataclasses
import json
import logging
import os
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
@click.option(
    "--storage-id", default="", help="rosbag2 storage plugin; detected when omitted."
)
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
_STATIC_TOPIC_HELP = (
    "Topic that can't span a time window (/tf_static, one-shot topics). "
    "Excluded from the window and gap checks; trim-bag copies all of its "
    "messages. Repeatable."
)


@main.command("validate-bag")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="/ws/src/ingest/config/ingest.yaml")
@click.option("--static-topic", "static_topics", multiple=True, help=_STATIC_TOPIC_HELP)
@click.option(
    "--gap-factor",
    default=1.5,
    show_default=True,
    type=click.FloatRange(min=1.0, min_open=True),
    help="A gap is consecutive messages more than this x the topic's median "
    "period apart. 1.5 flags every single dropped message.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the report as JSON.")
def validate_bag_cmd(
    bag_path: str,
    config_path: str,
    static_topics: tuple[str, ...],
    gap_factor: float,
    as_json: bool,
) -> None:
    """Report what each topic of the bag is missing. Exit 1 on a failure."""
    from wato_ingest.inputs import bag_check

    cfg = load_config(config_path)
    scan = bag_check.scan(
        bag_path, storage_id=cfg.storage_id, tf_static_topic=cfg.topics.tf_static
    )
    try:
        report = bag_check.build_report(
            scan, cfg, static_topics=set(static_topics), gap_factor=gap_factor
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    if as_json:
        out = dataclasses.asdict(report)
        out["failures"] = report.failures
        click.echo(json.dumps(out, indent=2))
    else:
        click.echo(bag_check.render(report))
    sys.exit(1 if report.failures else 0)


@main.command("trim-bag")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--static-topic", "static_topics", multiple=True, help=_STATIC_TOPIC_HELP)
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Output .mcap (default: $BAG_ROOT/trimmed/<bag>_trimmed.mcap).",
)
@click.option(
    "--compression",
    default="none",
    show_default=True,
    type=click.Choice(["none", "zstd", "lz4"]),
    help="Output chunk compression.",
)
@click.option("--overwrite", is_flag=True, help="Replace an existing output.")
@click.option(
    "--storage-id", default="", help="rosbag2 storage plugin; detected when omitted."
)
def trim_bag_cmd(
    bag_path: str,
    static_topics: tuple[str, ...],
    out_path: str | None,
    compression: str,
    overwrite: bool,
    storage_id: str,
) -> None:
    """Cut an MCAP bag to the window in which every non-static topic records."""
    from wato_ingest.inputs import trim

    try:
        src = trim.resolve_mcap_file(bag_path)
        if out_path is None:
            out_dir = os.path.join(os.environ.get("BAG_ROOT", "/data/bags"), "trimmed")
            out_path = str(trim.default_output(src, out_dir))
        result = trim.trim(
            bag_path,
            out_path,
            static_topics=list(static_topics),
            storage_id=storage_id,
            compression=compression,
            overwrite=overwrite,
        )
    except trim.TrimError as e:
        click.echo(f"trim-bag: {e}", err=True)
        sys.exit(1)
    w = result.window
    o = result.input_start_ns
    click.echo(
        json.dumps(
            {
                "output": result.output,
                "window_s_from_input_start": [
                    (w.start_ns - o) / 1e9,
                    (w.end_ns - o) / 1e9,
                ],
                "window_ns": [w.start_ns, w.end_ns],
                "cut_s": {
                    "start": (w.start_ns - o) / 1e9,
                    "end": (result.input_end_ns - w.end_ns) / 1e9,
                },
                "start_set_by": w.started_last,
                "end_set_by": w.ended_first,
                "static_topics": result.static_topics,
                "messages": {
                    t: {"input": a, "output": b} for t, (a, b) in result.counts.items()
                },
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
@main.command("register")
@click.option("--bag", "bag_path", required=True, type=click.Path(exists=True))
@click.option("--bag-id", default=None)
@click.option(
    "--storage-id", default="", help="rosbag2 storage plugin; detected when omitted."
)
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
    # Poses first so a stream failing the pose requirements aborts early.
    pose_result = poses.extract(
        bag_path,
        bag_id,
        chunk_id,
        t_start_ns=int(chunk["t_overlap_start_ns"]),
        t_end_ns=int(chunk["t_overlap_end_ns"]),
        cfg=cfg,
    )
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
                    "dense_fraction": pose_result.dense_fraction,
                    "held_positions_dropped": pose_result.n_held_dropped,
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
    """Match LiDAR sweeps to nearest camera frame per camera and look up sweep poses."""
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
    click.echo(
        f"bag: {results[0].bag_id}  ({len(results)} chunk{'s' if len(results) != 1 else ''})"
    )
    click.echo("")
    click.echo(f"  {'chunk':<8}  {'frames':>8}  {'dropped':>7}  tags")
    click.echo(f"  {'-----':<8}  {'------':>8}  {'-------':>7}  ----")
    for r in results:
        total = r.valid_camera_count + r.dropped_camera_count
        tags = ", ".join(r.quality_tags) if r.quality_tags else "ok"
        click.echo(
            f"  {r.chunk_id:<8}  {r.valid_camera_count:>6}/{total:<6}  {r.dropped_camera_count:>7}  {tags}"
        )
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
