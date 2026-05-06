"""Command-line entrypoint for label_refinement."""

from __future__ import annotations

import click

from wato_label_refinement.config import load_config
from wato_label_refinement.pipeline import run as run_pipeline


@click.group()
def main() -> None:
    """Track-centric refinement (multimodal LabelFormer)."""


@main.command("run")
@click.option("--bag", "bag_id", required=True, help="bag_id to process.")
@click.option(
    "--chunk", "chunk_id", default=None, help="optional chunk_id; default: all chunks."
)
@click.option("--config", "config_path", default="/config/pipeline.yaml")
def run_cmd(bag_id: str, chunk_id: str | None, config_path: str) -> None:
    cfg = load_config(config_path)
    run_pipeline(cfg, bag_id=bag_id, chunk_id=chunk_id)


if __name__ == "__main__":
    main()
