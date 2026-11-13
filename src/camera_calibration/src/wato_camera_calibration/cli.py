"""File-based camera calibration CLI; pipeline default remains opt-in."""

from __future__ import annotations

import json
import logging

import click

from wato_camera_calibration.config import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


@click.group()
def main():
    """Refine camera extrinsics from static LiDAR intensity observations."""


@main.command("run")
@click.option("--bag", "bag_id", required=True)
@click.option(
    "--config",
    "config_path",
    default="/ws/src/camera_calibration/config/camera_calibration.yaml",
)
@click.option(
    "--cameras", multiple=True, help="Camera ID; repeat to select multiple cameras."
)
@click.option(
    "--enable",
    is_flag=True,
    help="Enable a diagnostic run even when the profile is disabled.",
)
@click.option(
    "--prepare", is_flag=True, help="Prepare sampled pairs without running the backend."
)
@click.option("--force", is_flag=True)
def run_cmd(bag_id, config_path, cameras, enable, prepare, force):
    from wato_camera_calibration.config import ComponentConfig
    from wato_camera_calibration.pipeline import run

    cfg = load_config(config_path)
    data = cfg.model_dump()
    if enable:
        data["enabled"] = True
    if prepare:
        data["mode"] = "prepare"
    try:
        cfg = ComponentConfig.model_validate(data)
        report = run(cfg, bag_id=bag_id, cameras=cameras, force=force)
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                cid: {"status": row["status"], "reasons": row.get("reasons", [])}
                for cid, row in report.get("cameras", {}).items()
            },
            indent=2,
        )
    )
    if any(r["status"] == "error" for r in report.get("cameras", {}).values()):
        raise click.ClickException(
            "calibration errors were recorded; inspect report.json and backend.log"
        )


@main.command("viz")
@click.option("--bag", "bag_id", required=True)
def viz_cmd(bag_id):
    """Print the report and diagnostic overlay locations from a completed run."""
    from wato_common.artifacts.calibration_refinement import directory, read
    from wato_camera_calibration.viz import write_index

    _, report, _ = read(bag_id)
    click.echo(
        json.dumps(
            {cid: row["status"] for cid, row in report["cameras"].items()}, indent=2
        )
    )
    click.echo(f"Gallery: {write_index(directory(bag_id), report)}")


@main.command("pick")
@click.option("--bag", "bag_id", required=True)
@click.option(
    "--cameras", multiple=True, help="Camera ID; repeat to select multiple cameras."
)
@click.option(
    "--window-s",
    default=1.0,
    show_default=True,
    help="Accumulate static sweeps within this many seconds of each image.",
)
@click.option("--max-range-m", default=60.0, show_default=True)
@click.option("--voxel-m", default=0.03, show_default=True)
def pick_cmd(bag_id, cameras, window_s, max_range_m, voxel_m):
    """Write a browser picker for independent static correspondences."""
    from wato_camera_calibration.picker import export

    try:
        root = export(
            bag_id,
            cameras=cameras,
            window_s=window_s,
            max_range_m=max_range_m,
            voxel_m=voxel_m,
        )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Picker: {root}")
    click.echo(
        "Serve its parent directory, e.g. from the repo root:\n"
        f"  python3 -m http.server 8000 --bind 127.0.0.1 --directory "
        f"data/artifacts/raw/{bag_id}/camera_calibration\n"
        "then open http://localhost:8000/picker/ and save the download as "
        "correspondences.json in that directory."
    )
