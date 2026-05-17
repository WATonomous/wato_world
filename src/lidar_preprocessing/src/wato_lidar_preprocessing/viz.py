"""Visualization utilities for lidar_preprocessing artifacts.

Default behaviour: one window per chunk showing the accumulated classified
point cloud (static blue, dynamic red) — i.e. the merged result of every
processed sweep in the chunk. Pass --sweep N to inspect a single sweep.

Stage C is a matplotlib window for the 2D ground height/normal grids.

Open3D windows accept these keys (on top of the built-in mouse controls):
  1=top-down, 2=front, 3=side, 4=isometric, R=reset
  +/- = point size, S/D/G = layer toggles (where applicable)

For the popup to appear inside Docker, the container needs DISPLAY (and on
WSL2, /mnt/wslg + /tmp/.X11-unix) bind-mounted in. See the
lidar_preprocessing_dev service in modules/docker-compose.dev.yaml.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

_MAX_POINTS = 2_500_000  # downsample threshold for accumulated views
_MAX_GLOBAL_MAP_PTS = 1_000_000
_WIN_W = 1600
_WIN_H = 900

_STATIC_RGB = [0.29, 0.56, 0.85]  # blue
_DYNAMIC_RGB = [0.91, 0.30, 0.24]  # red
_GROUND_RGB = [0.20, 1.00, 0.20]  # green


def _o3d():
    try:
        import open3d as o3d

        return o3d
    except ImportError as e:
        raise ImportError(
            "open3d is required for point-cloud visualization: pip install open3d"
        ) from e


def _height_colors(z: np.ndarray) -> np.ndarray:
    """Map Z values to RGB using RdYlBu (low=red, mid=yellow, high=blue)."""
    import matplotlib.cm as cm

    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    norm = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return cm.RdYlBu(1.0 - norm)[:, :3].astype(np.float64)


def _downsample(xyz: np.ndarray, *cols: np.ndarray, max_pts: int = _MAX_POINTS):
    """Random-subsample xyz (and any parallel arrays) to at most max_pts."""
    if len(xyz) <= max_pts:
        return (xyz,) + cols
    idx = np.random.choice(len(xyz), max_pts, replace=False)
    return (xyz[idx],) + tuple(c[idx] for c in cols)


def _axis_size_for(pcds: Iterable) -> float:
    """Coordinate-frame size scaled to the cloud's extent."""
    pts = []
    for p in pcds:
        if hasattr(p, "points") and len(p.points) > 0:
            pts.append(np.asarray(p.points))
    if not pts:
        return 2.0
    merged = np.vstack(pts)
    extent = float(np.linalg.norm(merged.max(0) - merged.min(0)))
    return max(extent * 0.02, 1.0)


def _scene_center_extent(pcds: Iterable) -> tuple[np.ndarray, float]:
    pts = []
    for p in pcds:
        if hasattr(p, "points") and len(p.points) > 0:
            pts.append(np.asarray(p.points))
    if not pts:
        return np.zeros(3), 1.0
    merged = np.vstack(pts)
    mn, mx = merged.min(0), merged.max(0)
    return (mn + mx) * 0.5, float(np.linalg.norm(mx - mn))


def _set_view(vis, front, up, lookat, zoom: float = 0.5) -> bool:
    ctr = vis.get_view_control()
    ctr.set_front(np.asarray(front, dtype=np.float64))
    ctr.set_up(np.asarray(up, dtype=np.float64))
    ctr.set_lookat(np.asarray(lookat, dtype=np.float64))
    ctr.set_zoom(zoom)
    return False


def _show(
    geoms: dict,
    title: str,
    toggle_keys: dict | None = None,
) -> None:
    """Open an interactive Open3D window. Blocks until closed.

    geoms: {name: geometry} — added initially, all visible.
    toggle_keys: {key_char: geom_name} — pressing the key hides/shows the geometry.

    Always-on view-snap keys: 1=top, 2=front, 3=side, 4=isometric.
    """
    o3d = _o3d()

    nonempty = {
        n: g for n, g in geoms.items() if not (hasattr(g, "is_empty") and g.is_empty())
    }
    if not nonempty:
        log.warning("nothing to show for %s", title)
        return

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=_WIN_W, height=_WIN_H)

    visible = {name: True for name in nonempty}
    for g in nonempty.values():
        vis.add_geometry(g)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=_axis_size_for(nonempty.values())
    )
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.08, 0.08, 0.08])
    opt.point_size = 1.5

    center, _extent = _scene_center_extent(nonempty.values())

    # View presets — Z is up in world frame.
    vis.register_key_callback(
        ord("1"), lambda v: _set_view(v, [0, 0, -1], [0, 1, 0], center, 0.4)
    )  # top-down BEV
    vis.register_key_callback(
        ord("2"), lambda v: _set_view(v, [-1, 0, 0], [0, 0, 1], center, 0.5)
    )  # front (looking -X)
    vis.register_key_callback(
        ord("3"), lambda v: _set_view(v, [0, -1, 0], [0, 0, 1], center, 0.5)
    )  # side (looking -Y)
    vis.register_key_callback(
        ord("4"), lambda v: _set_view(v, [-1, -1, -1], [0, 0, 1], center, 0.5)
    )  # isometric

    if toggle_keys:

        def make_toggle(name):
            def cb(v):
                if visible[name]:
                    v.remove_geometry(nonempty[name], reset_bounding_box=False)
                else:
                    v.add_geometry(nonempty[name], reset_bounding_box=False)
                visible[name] = not visible[name]
                return False

            return cb

        for key, name in toggle_keys.items():
            if name in nonempty:
                vis.register_key_callback(ord(key.upper()), make_toggle(name))

    # Start in top-down BEV — most useful default for ground vehicle scans.
    _set_view(vis, [0, 0, -1], [0, 1, 0], center, 0.4)

    toggle_help = (
        f" | toggle: {', '.join(f'{k}={n}' for k, n in (toggle_keys or {}).items())}"
        if toggle_keys
        else ""
    )
    log.info(
        "viewer open: %s — keys: 1=top 2=front 3=side 4=iso R=reset +/-=ptsize%s — close to continue",
        title,
        toggle_help,
    )

    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Accumulated chunk view — the default
# ---------------------------------------------------------------------------


def viz_chunk_classified(bag_id: str, chunk_id: str) -> None:
    """Single window: every processed sweep in the chunk, merged and classified.

    Static points (blue) come from static_map.npz, dynamic points (red) from
    dynamic_map.npz. Press S to toggle static, D to toggle dynamic.
    """
    o3d = _o3d()
    from wato_lidar_preprocessing.io import load_dynamic_map, load_static_map

    static_xyz = load_static_map(bag_id, chunk_id)["xyz"]
    try:
        dyn_xyz = load_dynamic_map(bag_id, chunk_id)["xyz"]
    except FileNotFoundError:
        dyn_xyz = np.empty((0, 3))

    (static_xyz,) = _downsample(static_xyz)
    (dyn_xyz,) = _downsample(dyn_xyz)

    static_pcd = o3d.geometry.PointCloud()
    static_pcd.points = o3d.utility.Vector3dVector(static_xyz)
    static_pcd.paint_uniform_color(_STATIC_RGB)

    dyn_pcd = o3d.geometry.PointCloud()
    dyn_pcd.points = o3d.utility.Vector3dVector(dyn_xyz)
    dyn_pcd.paint_uniform_color(_DYNAMIC_RGB)

    title = (
        f"chunk {chunk_id} — static={len(static_xyz):,} (blue), "
        f"dynamic={len(dyn_xyz):,} (red)"
    )
    _show(
        {"static": static_pcd, "dynamic": dyn_pcd},
        title,
        toggle_keys={"S": "static", "D": "dynamic"},
    )


# ---------------------------------------------------------------------------
# Per-sweep views — opt-in via --sweep
# ---------------------------------------------------------------------------


def viz_stage_A(bag_id: str, chunk_id: str, sweep_id: int) -> None:
    """Single deskewed sweep, height-colored, ground in green. Press G to toggle ground."""
    o3d = _o3d()
    from wato_lidar_preprocessing.io import load_world_sweep

    data = load_world_sweep(bag_id, chunk_id, sweep_id)
    x, y, z = data["x"], data["y"], data["z"]
    xyz = np.column_stack([x, y, z])
    colors = _height_colors(z)

    ground_mask = data.get("ground_mask")
    if ground_mask is not None:
        ng = o3d.geometry.PointCloud()
        ng.points = o3d.utility.Vector3dVector(xyz[~ground_mask])
        ng.colors = o3d.utility.Vector3dVector(colors[~ground_mask])

        g = o3d.geometry.PointCloud()
        g.points = o3d.utility.Vector3dVector(xyz[ground_mask])
        g.colors = o3d.utility.Vector3dVector(
            np.tile(_GROUND_RGB, (int(ground_mask.sum()), 1))
        )
        _show(
            {"non_ground": ng, "ground": g},
            f"sweep {sweep_id} — deskewed (height-colored, green=ground)",
            toggle_keys={"G": "ground"},
        )
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        _show({"pcd": pcd}, f"sweep {sweep_id} — deskewed (height-colored)")


def viz_stage_B_sweep(bag_id: str, chunk_id: str, sweep_id: int) -> None:
    """Single sweep colored by static (blue) vs dynamic (red). S/D to toggle."""
    o3d = _o3d()
    from wato_lidar_preprocessing.io import load_dynamic_mask, load_world_sweep

    data = load_world_sweep(bag_id, chunk_id, sweep_id)
    dynamic = load_dynamic_mask(bag_id, chunk_id, sweep_id)
    x, y, z = data["x"], data["y"], data["z"]
    xyz = np.column_stack([x, y, z])

    static_pcd = o3d.geometry.PointCloud()
    static_pcd.points = o3d.utility.Vector3dVector(xyz[~dynamic])
    static_pcd.paint_uniform_color(_STATIC_RGB)

    dyn_pcd = o3d.geometry.PointCloud()
    dyn_pcd.points = o3d.utility.Vector3dVector(xyz[dynamic])
    dyn_pcd.paint_uniform_color(_DYNAMIC_RGB)

    n_dyn = int(dynamic.sum())
    n_static = len(xyz) - n_dyn
    title = f"sweep {sweep_id} — static={n_static:,} (blue), dynamic={n_dyn:,} (red)"
    _show(
        {"static": static_pcd, "dynamic": dyn_pcd},
        title,
        toggle_keys={"S": "static", "D": "dynamic"},
    )


# ---------------------------------------------------------------------------
# Stage C / D — ground grid + bag-level global map
# ---------------------------------------------------------------------------


def viz_stage_C(bag_id: str, chunk_id: str) -> None:
    """Stage C: height grid heatmap + HSV-encoded surface normals (matplotlib)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from wato_lidar_preprocessing.io import load_ground

    g = load_ground(bag_id, chunk_id)
    status = g.get("status", b"ok")
    if isinstance(status, (bytes, np.bytes_)):
        status = status.decode()
    if status in ("skipped_no_ground_mask", "empty"):
        log.warning(
            "ground status=%s for chunk %s — skipping stage C", status, chunk_id
        )
        return

    height_grid = g["height_grid"]
    normal_grid = g["normal_grid"]

    nx, ny, nz = normal_grid[..., 0], normal_grid[..., 1], normal_grid[..., 2]
    hue = (np.arctan2(ny, nx) / (2 * np.pi)) % 1.0
    normal_rgb = mcolors.hsv_to_rgb(
        np.stack([hue, np.ones_like(hue), np.clip(nz, 0, 1)], axis=-1)
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    valid = np.isfinite(height_grid)
    vmin = float(height_grid[valid].min()) if valid.any() else 0.0
    vmax = float(height_grid[valid].max()) if valid.any() else 1.0
    im = axes[0].imshow(
        height_grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
    )
    plt.colorbar(im, ax=axes[0], label="Height Z (m)")
    axes[0].set_title("Height grid")

    axes[1].imshow(normal_rgb, origin="lower")
    axes[1].set_title("Surface normals (XY angle → hue, Z → value)")

    fig.suptitle(f"Stage C — Ground | chunk {chunk_id}")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def viz_stage_D(bag_id: str) -> None:
    """Stage D: bag-level global static map (downsampled if large)."""
    o3d = _o3d()
    from wato_lidar_preprocessing.io import load_global_static_map

    xyz = load_global_static_map(bag_id)
    (xyz,) = _downsample(xyz, max_pts=_MAX_GLOBAL_MAP_PTS)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([0.55, 0.55, 0.55])

    _show({"global_static_map": pcd}, f"global static map | bag {bag_id}")


# Back-compat alias — viz_static_map() now just shows the classified accumulated view.
def viz_static_map(bag_id: str, chunk_id: str) -> None:
    viz_chunk_classified(bag_id, chunk_id)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def viz_chunk(
    bag_id: str,
    chunk_id: str,
    sweep_id: int | None = None,
    stage: str = "all",
) -> None:
    """Visualize a chunk.

    Without --sweep:
        Opens ONE window with every processed sweep merged + classified
        (static blue / dynamic red). Stage C also opens if stage in {C, all}.

    With --sweep N:
        Opens per-sweep windows for that sweep id (Stage A then Stage B).
        Stage C still opens if requested.

    stage: "A", "B", "C", or "all". Stage D is bag-level — call viz_stage_D() separately.
    """

    def do(s: str) -> bool:
        return stage in ("all", s)

    try:
        _o3d()
        has_o3d = True
    except ImportError as exc:
        has_o3d = False
        if do("A") or do("B"):
            log.warning(
                "open3d not available — skipping point-cloud views (%s). "
                "Stage C will still render.",
                exc,
            )

    if sweep_id is None:
        # Default: accumulated classified view of the whole chunk.
        if do("B") and has_o3d:
            try:
                viz_chunk_classified(bag_id, chunk_id)
            except Exception as exc:
                log.warning("classified accumulated view failed: %s", exc)
        # Stage A doesn't have a pre-accumulated artifact, so without --sweep
        # we just point users at the per-sweep view.
        if do("A") and not do("B"):
            log.info(
                "stage A is per-sweep only — pass --sweep N to inspect a single deskewed sweep."
            )
    else:
        if do("A") and has_o3d:
            try:
                viz_stage_A(bag_id, chunk_id, sweep_id)
            except Exception as exc:
                log.warning("stage A failed for sweep %d: %s", sweep_id, exc)
        if do("B") and has_o3d:
            try:
                viz_stage_B_sweep(bag_id, chunk_id, sweep_id)
            except Exception as exc:
                log.warning("stage B failed for sweep %d: %s", sweep_id, exc)

    if do("C"):
        try:
            viz_stage_C(bag_id, chunk_id)
        except Exception as exc:
            log.warning("stage C failed: %s", exc)
