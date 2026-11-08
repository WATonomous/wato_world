"""Command-line entrypoint for label_refinement."""

from __future__ import annotations

import logging

import click

from wato_label_refinement.config import load_config

# Python's default WARNING level would drop every log.info (per-chunk progress).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_DEFAULT_CONFIG = "/ws/src/label_refinement/config/label_refinement.yaml"


@click.group()
def main() -> None:
    """Training-free track refinement: one box per parked track, one size and
    a smoothed path per moving track."""


@main.command("run")
@click.option("--bag", "bag_id", required=True, help="bag_id to process.")
@click.option("--config", "config_path", default=_DEFAULT_CONFIG)
@click.option(
    "--force", is_flag=True, default=False, help="re-run even if refined labels exist."
)
def run_cmd(bag_id: str, config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    from wato_label_refinement.pipeline import run as run_pipeline

    run_pipeline(cfg, bag_id=bag_id, force=force, config_path=config_path)


if __name__ == "__main__":
    main()
