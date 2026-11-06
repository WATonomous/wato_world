"""Command-line entrypoint for perception_2d."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import click

from wato_common.progress import configure_tqdm
from wato_perception_2d.config import load_config
from wato_perception_2d.pipeline import run as run_pipeline

_SLUG = re.compile(r"[^a-zA-Z0-9_]+")

# Single live tqdm line on a terminal, silent when output is captured. See
# wato_common.progress (override with WATO_PROGRESS=on|off).
configure_tqdm()


def _resolve_bag_id(value: str) -> str:
    """Accept either a plain bag_id or a bag path (mirrors ingest/lidar_preprocessing)."""
    p = Path(value)
    if "/" in value or _SLUG.sub("", p.name) != p.name:
        return _SLUG.sub("_", p.name).strip("_") or "bag"
    return value


# Python's default WARNING level would drop every log.info (model loads,
# per-chunk progress).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@click.group()
def main() -> None:
    """2D perception pass (detector + SAM2 + Depth Anything V2 + DINOv2)."""


@main.command("run")
@click.option("--bag", "bag_id", required=True, help="bag_id or bag path to process.")
@click.option(
    "--chunk", "chunk_id", default=None, help="optional chunk_id; default: all chunks."
)
@click.option(
    "--config",
    "config_path",
    default="/ws/src/perception_2d/config/perception_2d.yaml",
)
@click.option(
    "--refine-masks/--no-refine-masks",
    default=None,
    help="Override segmentation.refinement.enabled (experimental; validate on annotated masks).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="re-run chunks even if outputs exist; discard per-camera resume caches.",
)
def run_cmd(
    bag_id: str,
    chunk_id: str | None,
    config_path: str,
    refine_masks: bool | None,
    force: bool,
) -> None:
    cfg = load_config(config_path)
    if refine_masks is not None:
        cfg.segmentation.refinement.enabled = refine_masks
    run_pipeline(
        cfg,
        bag_id=_resolve_bag_id(bag_id),
        chunk_id=chunk_id,
        force=force,
        config_path=config_path,
    )


@main.command("ego-mask")
@click.option("--bag", "bag_id", required=True, help="bag_id or bag path.")
@click.option(
    "--config",
    "config_path",
    default="/ws/src/perception_2d/config/perception_2d.yaml",
    help="config whose ego_mask polygons are checked (the run's config).",
)
@click.option(
    "--out",
    "out_dir",
    default=None,
    help="output directory (default: the bag's ego_mask/ artifact dir).",
)
@click.option(
    "--margin-px", default=6, type=int, help="dilation of the draft polygons."
)
def ego_mask_cmd(
    bag_id: str, config_path: str, out_dir: str | None, margin_px: int
) -> None:
    """Draft or check a vehicle's ego mask from this bag's perception_2d output.

    Per camera: the median image over the drive, an overlay of the masklets
    that held still at the image border (the vehicle's own body) and the
    config's polygons, and a draft ego_mask block (ego_mask.yaml). See
    wato_perception_2d.ego_mask.
    """
    from wato_common.artifact_store import ego_mask_dir, local_path
    from wato_perception_2d import ego_mask

    bag_id = _resolve_bag_id(bag_id)
    out = out_dir or local_path(ego_mask_dir(bag_id))
    uncovered = ego_mask.run(
        load_config(config_path), bag_id=bag_id, out_dir=out, margin_px=margin_px
    )
    missed = {cam: f for cam, f in uncovered.items() if f > 0.05}
    click.echo(f"wrote {out}")
    if missed:
        click.echo(
            "body pixels outside the config's polygons: "
            + ", ".join(f"{cam} {100 * f:.0f}%" for cam, f in sorted(missed.items()))
        )


@main.command("compare")
@click.option("--bag", "bag_id", required=True, help="bag_id or bag path.")
@click.option(
    "--baseline",
    required=True,
    help="artifact root of the baseline run (a path or a file:// URI).",
)
@click.option(
    "--chunk",
    "chunk_ids",
    multiple=True,
    help="chunk_id to compare; repeatable (default: every chunk both runs have).",
)
@click.option(
    "--config",
    "config_path",
    default="/ws/src/perception_2d/config/perception_2d.yaml",
    help="the current run's config (its ego mask pre-marks junk).",
)
@click.option(
    "--out",
    "out_dir",
    default=None,
    help="output directory (default: the bag's compare/ artifact dir).",
)
@click.option(
    "--every", default=5, type=int, help="sample every n-th image of each camera."
)
def compare_cmd(
    bag_id: str,
    baseline: str,
    chunk_ids: tuple[str, ...],
    config_path: str,
    out_dir: str | None,
    every: int,
) -> None:
    """Review what this run removed against a baseline run of the same bag.

    Lists the baseline masklets whose masks no current mask matches, as
    picture sheets and review.csv; give each a verdict (real, junk, covered,
    unsure) and run it again for the per-class loss. See
    wato_perception_2d.compare.
    """
    from wato_common.artifact_store import compare_dir, local_path
    from wato_perception_2d import compare

    bag_id = _resolve_bag_id(bag_id)
    out = out_dir or local_path(compare_dir(bag_id))
    uri = baseline if "://" in baseline else f"file://{Path(baseline).resolve()}"
    summary = compare.run(
        load_config(config_path),
        bag_id=bag_id,
        baseline_uri=uri,
        out_dir=out,
        chunk_ids=list(chunk_ids),
        every=every,
    )
    click.echo(f"wrote {out}")
    click.echo(compare.format_summary(summary), nl=False)


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
