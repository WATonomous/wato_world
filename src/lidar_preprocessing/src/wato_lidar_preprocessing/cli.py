"""Command-line entrypoint for lidar_preprocessing."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import click

from wato_lidar_preprocessing.config import load_config
from wato_lidar_preprocessing.pipeline import run as run_pipeline
from wato_lidar_preprocessing.reduce import reduce_static_map

_SLUG = re.compile(r"[^a-zA-Z0-9_]+")


def _resolve_bag_id(value: str) -> str:
    """Accept either a plain bag_id or a bag directory path.

    Mirrors ingest's derive_bag_id / slugify so both of these work:
        --bag NuScenes_v1_0_mini_scene_1100
        --bag data/bags/NuScenes-v1.0-mini-scene-1100/
    """
    p = Path(value)
    # If it looks like a path (contains a separator or the name has non-ID chars),
    # derive the bag_id from the directory name the same way ingest does.
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


@main.command("run")
@click.option("--bag", "bag_id", required=True, help="bag_id or bag path to process.")
@click.option(
    "--chunk", "chunk_id", default=None, help="optional chunk_id; default: all chunks."
)
@click.option("--config", "config_path", default="/ws/src/lidar_preprocessing/config/lidar_preprocessing.yaml")
@click.option(
    "--force", "-f", "force", is_flag=True, default=False,
    help="re-process chunks whose ground.npz already exists.",
)
@click.option(
    "--workers", "workers", default=1, type=int,
    help="number of concurrent worker processes (default 1 = sequential).",
)
def run_cmd(
    bag_id: str,
    chunk_id: str | None,
    config_path: str,
    force: bool,
    workers: int,
) -> None:
    """Run deskew → classify → ground for all chunks (or one chunk) of a bag."""
    bag_id = _resolve_bag_id(bag_id)
    cfg = load_config(config_path)
    run_pipeline(cfg, bag_id=bag_id, chunk_id=chunk_id, force=force, workers=workers)


@main.command("reduce")
@click.option("--bag", "bag_id", required=True, help="bag_id or bag path to reduce.")
@click.option("--config", "config_path", default="/ws/src/lidar_preprocessing/config/lidar_preprocessing.yaml")
def reduce_cmd(bag_id: str, config_path: str) -> None:
    """Merge all per-chunk static maps into a bag-level global_static_map.npz."""
    bag_id = _resolve_bag_id(bag_id)
    cfg = load_config(config_path)
    out = reduce_static_map(bag_id, cfg)
    click.echo(f"global static map written to {out}")


if __name__ == "__main__":
    main()
