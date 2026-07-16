"""Command-line entrypoint for lidar_preprocessing."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import click

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
    "--seg",
    "seg",
    type=click.Choice(["aw", "mos", "union"]),
    default=None,
    help=(
        "Segmentation method for the static/dynamic split (Step B). "
        "'aw' = pure Amanatides-Woo log-odds ray-casting (no MF-MOS). "
        "'mos' = pure MF-MOS learned segmentation (no ray traversal). "
        "'union' = fusion: AW static map + MF-MOS dynamics vetoed by it. "
        "Overrides the config's `segmentation` field; default uses the config."
    ),
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
    seg: str | None,
    force: bool,
    workers: int,
    auto_reduce: bool,
    two_pass: bool,
) -> None:
    """Run deskew → (aw|mos|union) → ground for all chunks (or one chunk) of a bag."""
    bag_id = _resolve_bag_id(bag_id)
    cfg = load_config(config_path)
    if seg is not None:
        cfg.segmentation = seg
    log.info("segmentation method: %s", cfg.segmentation)
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
    help="Write one classified sweep instead of the accumulated chunk.",
)
@click.option(
    "--layer",
    default="classification",
    type=click.Choice(["classification", "dynamic", "static", "ground", "global"]),
    help="Artifact to visualize. Native backends require a non-classification layer.",
)
@click.option(
    "--backend",
    default="html",
    type=click.Choice(["html", "web", "auto", "open3d", "plotly", "matplotlib"]),
    help=(
        "html writes a standalone browser viewer (default); web starts a local "
        "server; auto/open3d/plotly/matplotlib use the legacy layer viewer."
    ),
)
@click.option(
    "--export",
    "export_format",
    default=None,
    type=click.Choice(["ply"]),
    help="Export visualization data for external tools instead of opening a viewer.",
)
@click.option(
    "--out",
    "out_path",
    default=None,
    help=(
        "Output file or directory for --backend html or --export. "
        "Defaults under <chunk>/viz/."
    ),
)
@click.option("--host", default="127.0.0.1", help="Host for --backend web.")
@click.option("--port", default=8765, type=int, help="Port for --backend web.")
@click.option(
    "--open/--no-open",
    "open_after_write",
    default=False,
    help="Open generated HTML in the default browser.",
)
@click.option(
    "--color",
    default="height",
    type=click.Choice(["height", "sweep_id", "intensity"]),
    help="Initial color mode for native layer viewers.",
)
@click.option("--point-size", default=2.0, type=float, help="Native marker size.")
@click.option(
    "--max-points",
    default=2_000_000,
    type=int,
    help="Native viewer point cap (0 = no cap).",
)
def viz_cmd(
    bag_id: str,
    chunk_id: str | None,
    sweep_id: int | None,
    layer: str,
    backend: str,
    export_format: str | None,
    out_path: str | None,
    host: str,
    port: int,
    open_after_write: bool,
    color: str,
    point_size: float,
    max_points: int,
) -> None:
    """Visualize static/dynamic classification and pipeline artifacts."""
    from wato_common.artifact_store import chunks_index_path
    from wato_common.io.parquet_io import read_rows

    bag_id = _resolve_bag_id(bag_id)

    def selected_chunks() -> list[str]:
        if chunk_id is not None:
            return [chunk_id]
        return [row["chunk_id"] for row in read_rows(chunks_index_path(bag_id))]

    if export_format is not None:
        if layer != "classification":
            raise click.ClickException("--export supports --layer classification only")
        from wato_lidar_preprocessing.viz_export import export_ply

        chunk_ids = selected_chunks()
        for cid in chunk_ids:
            target = out_path
            if out_path is not None and len(chunk_ids) > 1:
                target = str(Path(out_path) / cid)
            written = export_ply(bag_id, cid, sweep_id=sweep_id, out_path=target)
            click.echo(f"exported {written}")
        return

    if backend == "web":
        if layer != "classification":
            raise click.ClickException(
                "--backend web supports --layer classification only"
            )
        if chunk_id is None:
            raise click.ClickException("--backend web requires --chunk")
        if sweep_id is not None:
            raise click.ClickException("--backend web is chunk-level; omit --sweep")
        if open_after_write:
            raise click.ClickException("--open is only supported by --backend html")
        from wato_lidar_preprocessing.web_viz import serve_web_viz

        serve_web_viz(bag_id, chunk_id, host=host, port=port)
        return

    if backend == "html":
        if layer != "classification":
            raise click.ClickException(
                "--backend html supports --layer classification only"
            )
        from wato_lidar_preprocessing.html_viz import open_html_file, write_html_viewer

        chunk_ids = selected_chunks()
        for cid in chunk_ids:
            target = out_path
            if out_path is not None and len(chunk_ids) > 1:
                target = str(Path(out_path) / cid)
            written = write_html_viewer(bag_id, cid, sweep_id=sweep_id, out_path=target)
            click.echo(f"wrote {written}")
            if open_after_write:
                if open_html_file(written):
                    click.echo(f"opened {written.resolve()}")
                else:
                    click.echo(
                        f"could not open automatically; open {written.resolve()}"
                    )
        return

    if open_after_write:
        raise click.ClickException("--open is only supported by --backend html")
    if sweep_id is not None:
        raise click.ClickException(
            "--sweep is only supported by --backend html or --export"
        )
    if layer == "classification":
        raise click.ClickException(
            "native backends require --layer dynamic, static, ground, or global"
        )

    from wato_lidar_preprocessing.viz import viz_chunk, viz_global_static_map

    opts = dict(
        color=color, backend=backend, point_size=point_size, max_points=max_points
    )
    if layer == "global":
        try:
            viz_global_static_map(bag_id, **opts)
        except FileNotFoundError:
            click.echo("global_static_map.npz not found (run 'reduce' first)")
        return

    for cid in selected_chunks():
        if layer == "ground":
            viz_chunk(bag_id, cid, layer="ground")
        else:
            viz_chunk(bag_id, cid, layer=layer, **opts)


if __name__ == "__main__":
    main()
