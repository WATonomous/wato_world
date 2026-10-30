"""Command-line entrypoint for semantic_lifting."""

from __future__ import annotations

import logging

import click

from wato_semantic_lifting.config import load_config

# Python's default WARNING level would drop every log.info (per-chunk progress).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


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
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="re-run chunks even if their lifted labels exist.",
)
def run_cmd(bag_id: str, chunk_id: str | None, config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    from wato_semantic_lifting.pipeline import run as run_pipeline

    run_pipeline(
        cfg, bag_id=bag_id, chunk_id=chunk_id, force=force, config_path=config_path
    )


@main.command("viz")
@click.option("--bag", "bag_id", required=True, help="bag_id to visualize.")
@click.option(
    "--chunk", "chunk_id", default=None, help="optional chunk_id; default: all chunks."
)
@click.option(
    "--color",
    "color_by",
    default="instance",
    type=click.Choice(["instance", "class"]),
    help="Colour labeled points by lifted instance or by class.",
)
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Output file or directory. Defaults under <chunk>/viz/.",
)
@click.option(
    "--open/--no-open",
    "open_after_write",
    default=False,
    help="Open generated HTML in the default browser.",
)
def viz_cmd(
    bag_id: str,
    chunk_id: str | None,
    color_by: str,
    out_path: str | None,
    open_after_write: bool,
) -> None:
    """Standalone HTML viewer of a chunk's lifted labels."""
    from pathlib import Path

    from wato_common.viz.point_cloud_html import open_html_file
    from wato_semantic_lifting.io import load_chunks
    from wato_semantic_lifting.viz import write_html_viewer

    chunk_ids = [chunk_id] if chunk_id else [c["chunk_id"] for c in load_chunks(bag_id)]
    for cid in chunk_ids:
        target = out_path
        if out_path is not None and len(chunk_ids) > 1:
            target = str(Path(out_path) / cid)
        written = write_html_viewer(bag_id, cid, color_by=color_by, out_path=target)
        click.echo(f"wrote {written}")
        if open_after_write:
            if open_html_file(written):
                click.echo(f"opened {written.resolve()}")
            else:
                click.echo(f"could not open automatically; open {written.resolve()}")


if __name__ == "__main__":
    main()
