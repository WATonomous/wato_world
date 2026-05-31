"""Command-line entrypoint for perception_2d."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import click

from wato_perception_2d.config import load_config
from wato_perception_2d.pipeline import run as run_pipeline

_SLUG = re.compile(r"[^a-zA-Z0-9_]+")


def _resolve_bag_id(value: str) -> str:
    """Accept either a plain bag_id or a bag directory path (mirrors ingest)."""
    p = Path(value)
    if "/" in value or _SLUG.sub("", p.name) != p.name:
        return _SLUG.sub("_", p.name).strip("_") or "bag"
    return value


# Without this, every `log.info(...)` across perception_2d is dropped at
# Python's default WARNING level — so you can't see model-load status,
# per-chunk progress, etc. Mirrors src/ingest/src/wato_ingest/cli.py.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@click.group()
def main() -> None:
    """2D perception pass (Florence-2 + SAM 3.1 + Depth Anything V2 + DINOv2 + x-cam merge)."""


@main.command("run")
@click.option("--bag", "bag_id", required=True, help="bag_id to process.")
@click.option(
    "--chunk", "chunk_id", default=None, help="optional chunk_id; default: all chunks."
)
@click.option(
    "--config",
    "config_path",
    default="/ws/src/perception_2d/config/perception_2d.yaml",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="re-run chunks even if outputs exist; discard per-camera resume caches.",
)
def run_cmd(bag_id: str, chunk_id: str | None, config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    run_pipeline(cfg, bag_id=bag_id, chunk_id=chunk_id, force=force)


@main.command("viz")
@click.option("--bag", "bag_id", required=True, help="bag_id or bag path.")
@click.option(
    "--chunk",
    "chunk_id",
    default=None,
    help="chunk_id to visualize (default: all chunks, one window each).",
)
@click.option(
    "--cam",
    "cam_id",
    default=None,
    help="starting camera id (default: first with depth; switchable in-window).",
)
@click.option(
    "--max-side",
    "max_side",
    default=1280,
    type=int,
    help="downscale frames so the longest side ≤ this many pixels (display only).",
)
def viz_cmd(
    bag_id: str, chunk_id: str | None, cam_id: str | None, max_side: int
) -> None:
    """Interactive depth-vs-image viewer for depth_2d artifacts.

    Scrub frames, blend RGB ↔ depth heatmap by opacity, or drag a split
    curtain to compare. The window blocks until closed. Requires DISPLAY
    (or WSLg) forwarded into the container — see modules/docker-compose.dev.yaml
    — or just run it on the host where matplotlib has a GUI backend.
    """
    from wato_perception_2d.viz import viz_depth

    viz_depth(
        _resolve_bag_id(bag_id),
        chunk_id=chunk_id,
        cam_id=cam_id,
        max_side=max_side,
    )


if __name__ == "__main__":
    main()
