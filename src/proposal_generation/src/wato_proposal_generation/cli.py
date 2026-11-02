"""Command-line entrypoint for proposal_generation."""

from __future__ import annotations

import logging

import click

from wato_proposal_generation.config import load_config

# Python's default WARNING level would drop every log.info (per-chunk progress).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_DEFAULT_CONFIG = "/ws/src/proposal_generation/config/proposal_generation.yaml"


@click.group()
def main() -> None:
    """3D box proposals from Step F clusters and semantic_lifting's labels."""


@main.command("run")
@click.option("--bag", "bag_id", required=True, help="bag_id to process.")
@click.option(
    "--chunk", "chunk_id", default=None, help="optional chunk_id; default: all chunks."
)
@click.option("--config", "config_path", default=_DEFAULT_CONFIG)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="re-run chunks even if their proposals.parquet exists.",
)
def run_cmd(bag_id: str, chunk_id: str | None, config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    from wato_proposal_generation.pipeline import run as run_pipeline

    run_pipeline(
        cfg, bag_id=bag_id, chunk_id=chunk_id, force=force, config_path=config_path
    )


@main.command("viz")
@click.option("--bag", "bag_id", required=True, help="bag_id to visualize.")
@click.option(
    "--chunk", "chunk_id", default=None, help="optional chunk_id; default: all chunks."
)
@click.option(
    "--config",
    "config_path",
    default=_DEFAULT_CONFIG,
    help="the run's config (its canonical LiDAR groups the frames).",
)
@click.option(
    "--view",
    type=click.Choice(["3d", "bev", "camera"]),
    default="3d",
    help="3d: browser page; bev: top-down PNG per frame; camera: boxes on images.",
)
@click.option(
    "--show",
    type=click.Choice(["sources", "fused"]),
    default="sources",
    help="source rows coloured by source, or fused rows coloured by class.",
)
@click.option(
    "--every",
    type=int,
    default=None,
    help="every Nth frame (3d: 5, bev: 10) or image per camera (camera: 10).",
)
@click.option(
    "--camera",
    "cameras",
    multiple=True,
    help="--view camera only: these cameras (repeatable); default all.",
)
@click.option(
    "--out", "out_path", default=None, help="Output path. Defaults under <chunk>/viz/."
)
@click.option(
    "--open/--no-open",
    "open_after_write",
    default=False,
    help="--view 3d only: open the page in the default browser.",
)
def viz_cmd(
    bag_id: str,
    chunk_id: str | None,
    config_path: str,
    view: str,
    show: str,
    every: int | None,
    cameras: tuple[str, ...],
    out_path: str | None,
    open_after_write: bool,
) -> None:
    """Draw a chunk's proposals."""
    from pathlib import Path

    from wato_common.viz.point_cloud_html import open_html_file
    from wato_proposal_generation.io import load_chunks
    from wato_proposal_generation.viz.bev import write_bev
    from wato_proposal_generation.viz.browser import write_browser
    from wato_proposal_generation.viz.camera import write_camera

    cfg = load_config(config_path)
    chunk_ids = [chunk_id] if chunk_id else [c["chunk_id"] for c in load_chunks(bag_id)]
    for cid in chunk_ids:
        target = out_path
        if out_path is not None and len(chunk_ids) > 1:
            target = str(Path(out_path) / cid)
        if view == "3d":
            written = write_browser(
                cfg, bag_id, cid, show=show, every=every or 5, out_path=target
            )
            click.echo(f"wrote {written}")
            if open_after_write and not open_html_file(written):
                click.echo(f"could not open automatically; open {written.resolve()}")
        elif view == "bev":
            files = write_bev(
                cfg, bag_id, cid, show=show, every=every or 10, out_dir=target
            )
            click.echo(
                f"wrote {len(files)} PNGs under {files[0].parent if files else target}"
            )
        else:
            files = write_camera(
                cfg,
                bag_id,
                cid,
                show=show,
                every=every or 10,
                cameras=list(cameras) or None,
                out_dir=target,
            )
            click.echo(
                f"wrote {len(files)} images under "
                f"{files[0].parent.parent if files else target}"
            )


if __name__ == "__main__":
    main()
