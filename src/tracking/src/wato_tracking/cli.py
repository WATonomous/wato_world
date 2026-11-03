"""Command-line entrypoint for tracking."""

from __future__ import annotations

import logging

import click

from wato_tracking.config import load_config

# Python's default WARNING level would drop every log.info (per-chunk progress).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_DEFAULT_CONFIG = "/ws/src/tracking/config/tracking.yaml"


@click.group()
def main() -> None:
    """Tracks over proposal_generation's fused boxes, parked / moving per track."""


@main.command("run")
@click.option("--bag", "bag_id", required=True, help="bag_id to process.")
@click.option(
    "--chunk",
    "chunk_id",
    default=None,
    help="optional chunk_id: re-track only it, keeping the other chunks' tracks.",
)
@click.option("--config", "config_path", default=_DEFAULT_CONFIG)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="re-run even if the bag's tracks exist.",
)
def run_cmd(bag_id: str, chunk_id: str | None, config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    from wato_tracking.pipeline import run as run_pipeline

    run_pipeline(
        cfg, bag_id=bag_id, chunk_id=chunk_id, force=force, config_path=config_path
    )


if __name__ == "__main__":
    main()
