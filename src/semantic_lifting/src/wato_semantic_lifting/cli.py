"""Command-line entrypoint for semantic_lifting."""

from __future__ import annotations

import click

from wato_semantic_lifting.config import load_config


@click.group()
def main() -> None:
    """Semantic lifting — occlusion-aware LiDAR label assignment from 2D masks."""


@main.command("run")
@click.option("--bag", "bag_id", required=True, help="bag_id to process.")
@click.option(
    "--chunk", "chunk_id", default=None, help="optional chunk_id; default: all chunks."
)
@click.option(
    "--config",
    "config_path",
    default="/ws/src/semantic_lifting/config/semantic_lifting.yaml",
)
def run_cmd(bag_id: str, chunk_id: str | None, config_path: str) -> None:
    cfg = load_config(config_path)
    from wato_semantic_lifting.pipeline import run as run_pipeline
    run_pipeline(cfg, bag_id=bag_id, chunk_id=chunk_id)


if __name__ == "__main__":
    main()
