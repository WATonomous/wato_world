"""Command-line entrypoint for lidar_preprocessing."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import click

from wato_common.progress import configure_tqdm
from wato_lidar_preprocessing.config import load_config
from wato_lidar_preprocessing.pipeline import run as run_pipeline
from wato_lidar_preprocessing.reduce import reduce_ground_map, reduce_static_map

log = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-zA-Z0-9_]+")


def _resolve_bag_id(value: str) -> str:
    """Accept either a plain bag_id or a bag directory path.

    Mirrors ingest's derive_bag_id / slugify so both of these work:
        --bag NuScenes_v1_0_mini_scene_1100
        --bag data/bags/NuScenes-v1.0-mini-scene-1100/
    """
    p = Path(value)
    # Path-shaped input → derive bag_id from the directory name the same way
    # ingest does (slugify non-identifier chars).
    if "/" in value or not _SLUG.sub("", p.name) == p.name:
        raw = _SLUG.sub("_", p.name).strip("_") or "bag"
        return raw
    return value


@click.group()
@click.option("--log-level", default="INFO", help="Logging level.")
def main(log_level: str) -> None:
    """LiDAR preprocessing (motion comp, static/dynamic, ground)."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Single live tqdm line on a terminal, silent when output is captured. See
    # wato_common.progress (override with WATO_PROGRESS=on|off).
    configure_tqdm()


@main.command("run")
@click.option("--bag", "bag_id", required=True, help="bag_id or bag path to process.")
@click.option(
    "--chunk", "chunk_id", default=None, help="optional chunk_id; default: all chunks."
)
@click.option(
    "--config",
    "config_path",
    default="/ws/src/lidar_preprocessing/config/lidar_preprocessing.yaml",
)
@click.option(
    "--force",
    "-f",
    "force",
    is_flag=True,
    default=False,
    help="re-process chunks whose ground.npz already exists.",
)
@click.option(
    "--workers",
    "workers",
    default=1,
    type=int,
    help="number of concurrent worker processes (default 1 = sequential).",
)
@click.option(
    "--auto-reduce/--no-auto-reduce",
    "auto_reduce",
    default=True,
    help=(
        "after all chunks finish, automatically run the bag-level reduce "
        "(global_static_map.npz + global_ground.npz). "
        "Disable with --no-auto-reduce when processing chunks in parallel "
        "across multiple machines — run 'reduce' manually once all chunks are done."
    ),
)
@click.option(
    "--two-pass/--no-two-pass",
    "two_pass",
    default=True,
    help=(
        "Enabled by default: run classification twice — pass 1 builds a rough "
        "static map, then reduce builds the bag-level global_static_map.npz, "
        "then pass 2 re-classifies every chunk using that map as a per-sweep "
        "KDTree prior (UniLiPs IWU).  Roughly doubles wall time but improves "
        "static recall on long-range structure sparsely observed in any single "
        "chunk.  Use --no-two-pass for the legacy single-pass behavior."
    ),
)
def run_cmd(
    bag_id: str,
    chunk_id: str | None,
    config_path: str,
    force: bool,
    workers: int,
    auto_reduce: bool,
    two_pass: bool,
) -> None:
    """Run deskew → classify → ground for all chunks (or one chunk) of a bag."""
    bag_id = _resolve_bag_id(bag_id)
    cfg = load_config(config_path)
    run_pipeline(
        cfg,
        bag_id=bag_id,
        chunk_id=chunk_id,
        force=force,
        workers=workers,
        two_pass=two_pass,
    )
    # Two-pass already built one global_static_map.npz to seed pass 2; we
    # re-reduce here so the final on-disk map reflects pass-2 outputs.
    if auto_reduce and chunk_id is None:
        log.info("auto-reduce: building global_static_map.npz + global_ground.npz ...")
        static_out = reduce_static_map(bag_id, cfg)
        ground_out = reduce_ground_map(bag_id, cfg)
        log.info("auto-reduce complete: %s  %s", static_out, ground_out)


@main.command("reduce")
@click.option("--bag", "bag_id", required=True, help="bag_id or bag path to reduce.")
@click.option(
    "--config",
    "config_path",
    default="/ws/src/lidar_preprocessing/config/lidar_preprocessing.yaml",
)
def reduce_cmd(bag_id: str, config_path: str) -> None:
    """Merge per-chunk artifacts into bag-level global maps.

    Produces both ``global_static_map.npz`` (downsampled bag-level static
    cloud) and ``global_ground.npz`` (bag-level height grid for queries
    that span chunk boundaries, e.g. SLF L_ground for boxes near chunk
    seams).
    """
    bag_id = _resolve_bag_id(bag_id)
    cfg = load_config(config_path)
    static_out = reduce_static_map(bag_id, cfg)
    click.echo(f"global static map written to {static_out}")
    ground_out = reduce_ground_map(bag_id, cfg)
    click.echo(f"global ground map written to {ground_out}")


@main.command("viz")
@click.option("--bag", "bag_id", required=True, help="bag_id or bag path.")
@click.option(
    "--chunk",
    "chunk_id",
    default=None,
    help="chunk_id to visualize (default: all chunks).",
)
@click.option(
    "--sweep",
    "sweep_id",
    default=None,
    type=int,
    help="optional specific sweep_id (stages A/B only).",
)
@click.option(
    "--stage",
    default="all",
    type=click.Choice(["A", "B", "C", "D", "all"]),
    help="Pipeline stage to visualize (default: all).",
)
def viz_cmd(
    bag_id: str, chunk_id: str | None, sweep_id: int | None, stage: str
) -> None:
    """Open interactive Open3D / matplotlib windows for pipeline artifacts.

    Each window blocks until you close it. Requires DISPLAY (or WSLg) to be
    forwarded into the container — see modules/docker-compose.dev.yaml.
    """
    from wato_common.artifact_store import chunks_index_path
    from wato_common.io.parquet_io import read_rows
    from wato_lidar_preprocessing.viz import viz_chunk, viz_stage_D

    bag_id = _resolve_bag_id(bag_id)

    if stage == "D":
        viz_stage_D(bag_id)
        return

    if chunk_id is not None:
        chunk_ids = [chunk_id]
    else:
        rows = read_rows(chunks_index_path(bag_id))
        chunk_ids = [r["chunk_id"] for r in rows]

    for cid in chunk_ids:
        viz_chunk(bag_id, cid, sweep_id=sweep_id, stage=stage)

    if stage == "all":
        try:
            viz_stage_D(bag_id)
        except FileNotFoundError:
            click.echo(
                "skipping stage D: global_static_map.npz not found (run 'reduce' first)"
            )
        except ImportError as exc:
            click.echo(f"skipping stage D: {exc}")


if __name__ == "__main__":
    main()
