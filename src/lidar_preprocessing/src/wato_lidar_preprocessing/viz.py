"""Visualization utilities for lidar_preprocessing artifacts.

Writes PNG files for visual inspection of each pipeline stage.
Stages A, B, D use Open3D OffscreenRenderer (headless, no display needed).
Stage C uses matplotlib for the 2D height/normal grids.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_MAX_STATIC_MAP_PTS = 500_000
_MAX_GLOBAL_MAP_PTS = 1_000_000
_RENDER_W = 1920
_RENDER_H = 1080


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _o3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError as e:
        raise ImportError("open3d is required for point-cloud visualization: pip install open3d") from e


def _viz_dir_chunk(bag_id: str, chunk_id: str) -> Path:
    from wato_common.artifact_store import chunk_root, local_path
    return Path(local_path(chunk_root(bag_id, chunk_id))) / "viz"


def _viz_dir_bag(bag_id: str) -> Path:
    from wato_common.artifact_store import bag_root, local_path
    return Path(local_path(bag_root(bag_id))) / "viz"


def _height_colors(z: np.ndarray) -> np.ndarray:
    """Map Z values to RGB using RdYlBu (low=red, mid=yellow, high=blue)."""
    import matplotlib.cm as cm
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    norm = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return cm.RdYlBu(1.0 - norm)[:, :3].astype(np.float64)


def _render_bev(pcd, out: Path) -> None:
    """Render a PointCloud to PNG using a top-down (BEV) perspective camera."""
    o3d = _o3d()

    if pcd.is_empty():
        log.warning("empty point cloud, skipping render for %s", out)
        return

    renderer = o3d.visualization.rendering.OffscreenRenderer(_RENDER_W, _RENDER_H)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0

    renderer.scene.add_geometry("pcd", pcd, mat)
    renderer.scene.set_background(np.array([0.08, 0.08, 0.08, 1.0]))

    bb = pcd.get_axis_aligned_bounding_box()
    center = bb.get_center()
    extent = bb.get_extent()
    # Camera sits high above center, looking straight down
    eye_z = center[2] + max(extent[0], extent[1]) * 0.75
    renderer.scene.camera.look_at(
        center.tolist(),
        [center[0], center[1], eye_z],
        [0.0, 1.0, 0.0],
    )

    img = renderer.render_to_image()
    o3d.io.write_image(str(out), img)
    log.info("wrote %s", out)


# ---------------------------------------------------------------------------
# Per-stage public functions
# ---------------------------------------------------------------------------

def viz_stage_A(bag_id: str, chunk_id: str, sweep_id: int) -> Path:
    """BEV of a deskewed world-frame sweep: height-colored, ground in green."""
    o3d = _o3d()
    from wato_lidar_preprocessing.io import load_world_sweep

    data = load_world_sweep(bag_id, chunk_id, sweep_id)
    x, y, z = data["x"], data["y"], data["z"]
    xyz = np.column_stack([x, y, z])
    colors = _height_colors(z)

    ground_mask = data.get("ground_mask")
    if ground_mask is not None:
        colors[ground_mask] = [0.2, 1.0, 0.2]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_dir = _viz_dir_chunk(bag_id, chunk_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"A_sweep_{sweep_id:06d}_bev.png"
    _render_bev(pcd, out)
    return out


def viz_stage_B_sweep(bag_id: str, chunk_id: str, sweep_id: int) -> Path:
    """BEV of a sweep colored by static (blue) vs dynamic (red)."""
    o3d = _o3d()
    from wato_lidar_preprocessing.io import load_dynamic_mask, load_world_sweep

    data = load_world_sweep(bag_id, chunk_id, sweep_id)
    dynamic = load_dynamic_mask(bag_id, chunk_id, sweep_id)
    x, y, z = data["x"], data["y"], data["z"]
    xyz = np.column_stack([x, y, z])

    colors = np.full((len(xyz), 3), [0.29, 0.56, 0.85])  # blue = static
    colors[dynamic] = [0.91, 0.30, 0.24]                  # red = dynamic

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_dir = _viz_dir_chunk(bag_id, chunk_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"B_sweep_{sweep_id:06d}_static_dynamic.png"
    _render_bev(pcd, out)
    return out


def viz_static_map(bag_id: str, chunk_id: str) -> Path:
    """BEV of the accumulated chunk static map (downsampled if large)."""
    o3d = _o3d()
    from wato_lidar_preprocessing.io import load_static_map

    sm = load_static_map(bag_id, chunk_id)
    xyz = sm["xyz"]
    if len(xyz) > _MAX_STATIC_MAP_PTS:
        idx = np.random.choice(len(xyz), _MAX_STATIC_MAP_PTS, replace=False)
        xyz = xyz[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([0.65, 0.65, 0.65])

    out_dir = _viz_dir_chunk(bag_id, chunk_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B_static_map.png"
    _render_bev(pcd, out)
    return out


def viz_stage_C(bag_id: str, chunk_id: str) -> Path | None:
    """Height grid heatmap + HSV-encoded surface normals (matplotlib, headless)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from wato_lidar_preprocessing.io import load_ground

    g = load_ground(bag_id, chunk_id)
    # status may be stored as bytes in numpy string arrays
    status = g.get("status", b"ok")
    if isinstance(status, (bytes, np.bytes_)):
        status = status.decode()
    if status in ("skipped_no_ground_mask", "empty"):
        log.warning("ground status=%s for chunk %s — skipping stage C viz", status, chunk_id)
        return None

    height_grid = g["height_grid"]
    normal_grid = g["normal_grid"]

    # HSV-encode normals: XY angle → hue, Z component → value
    nx, ny, nz = normal_grid[..., 0], normal_grid[..., 1], normal_grid[..., 2]
    hue = (np.arctan2(ny, nx) / (2 * np.pi)) % 1.0
    normal_rgb = mcolors.hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.clip(nz, 0, 1)], axis=-1))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    valid = np.isfinite(height_grid)
    vmin = float(height_grid[valid].min()) if valid.any() else 0.0
    vmax = float(height_grid[valid].max()) if valid.any() else 1.0
    im = axes[0].imshow(height_grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=axes[0], label="Height Z (m)")
    axes[0].set_title("Height grid")

    axes[1].imshow(normal_rgb, origin="lower")
    axes[1].set_title("Surface normals (XY angle → hue, Z → value)")

    fig.suptitle(f"Stage C — Ground | chunk {chunk_id}")
    fig.tight_layout()

    out_dir = _viz_dir_chunk(bag_id, chunk_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "C_ground.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)
    return out


def viz_stage_D(bag_id: str) -> Path:
    """BEV of the bag-level global static map (downsampled if large)."""
    o3d = _o3d()
    from wato_lidar_preprocessing.io import load_global_static_map

    xyz = load_global_static_map(bag_id)
    if len(xyz) > _MAX_GLOBAL_MAP_PTS:
        idx = np.random.choice(len(xyz), _MAX_GLOBAL_MAP_PTS, replace=False)
        xyz = xyz[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([0.55, 0.55, 0.55])

    out_dir = _viz_dir_bag(bag_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "D_global_static_map.png"
    _render_bev(pcd, out)
    return out


def viz_chunk(
    bag_id: str,
    chunk_id: str,
    sweep_id: int | None = None,
    stage: str = "all",
) -> list[Path]:
    """Visualize one or more stages for a single chunk.

    stage: "A", "B", "C", or "all". Stage D is bag-level; call viz_stage_D() separately.
    sweep_id: if None, visualize all valid sweeps (stages A and B).
    """
    from wato_lidar_preprocessing.io import load_proc_index

    rows = load_proc_index(bag_id, chunk_id)
    sweep_ids = [r["sweep_id"] for r in rows if r.get("valid", True)]
    if sweep_id is not None:
        sweep_ids = [s for s in sweep_ids if s == sweep_id]

    outputs: list[Path] = []
    do = lambda s: stage in ("all", s)

    if do("A"):
        for sid in sweep_ids:
            try:
                outputs.append(viz_stage_A(bag_id, chunk_id, sid))
            except Exception as exc:
                log.warning("stage A viz failed for sweep %d: %s", sid, exc)

    if do("B"):
        for sid in sweep_ids:
            try:
                outputs.append(viz_stage_B_sweep(bag_id, chunk_id, sid))
            except Exception as exc:
                log.warning("stage B sweep viz failed for sweep %d: %s", sid, exc)
        try:
            outputs.append(viz_static_map(bag_id, chunk_id))
        except Exception as exc:
            log.warning("stage B static-map viz failed: %s", exc)

    if do("C"):
        try:
            out = viz_stage_C(bag_id, chunk_id)
            if out is not None:
                outputs.append(out)
        except Exception as exc:
            log.warning("stage C viz failed: %s", exc)

    return [o for o in outputs if o is not None]
