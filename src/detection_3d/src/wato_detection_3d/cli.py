"""Command-line entrypoint for detection_3d."""

from __future__ import annotations

import logging
from typing import Optional

import click

from wato_detection_3d.config import load_config

# Python's default WARNING level would drop every log.info (per-chunk progress).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_DEFAULT_CONFIG = "/ws/src/detection_3d/config/detection_3d.yaml"


@click.group()
def main() -> None:
    """Learned 3D boxes from BEVFusion, the camera + LiDAR model the car runs,
    for proposal_generation."""


@main.command("run")
@click.option("--bag", "bag_id", required=True, help="bag_id to process.")
@click.option("--chunk", "chunk_id", default=None, help="one chunk only.")
@click.option("--config", "config_path", default=_DEFAULT_CONFIG)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="re-run even if a chunk's detections exist.",
)
def run_cmd(
    bag_id: str, chunk_id: Optional[str], config_path: str, force: bool
) -> None:
    """Run BEVFusion on every frame of the bag's chunks."""
    cfg = load_config(config_path)
    from wato_detection_3d.pipeline import run as run_pipeline

    run_pipeline(
        cfg, bag_id=bag_id, chunk_id=chunk_id, force=force, config_path=config_path
    )


if __name__ == "__main__":
    main()
