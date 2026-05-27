"""Visualization utilities for lidar_preprocessing artifacts.

Per chunk opens two windows in sequence — static (blue) then dynamic. The
dynamic window has a sweep slider, Play/Pause, and color modes
(uniform/p_occ/classification/n_obs/n_hits/intensity/sweep_id). Stage C
opens a matplotlib window for the ground height/normal grids.

Static / per-sweep windows use legacy Open3D VisualizerWithKeyCallback
(1/2/3/4 view presets, R reset, +/- point size, letter keys for layer
toggles). Dynamic window uses Open3D gui.

Docker: requires DISPLAY (and on WSL2, /mnt/wslg + /tmp/.X11-unix) bind-
mounted in — see lidar_preprocessing_dev in modules/docker-compose.dev.yaml.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

import numpy as np

log = logging.getLogger(__name__)

_MAX_POINTS = 2_500_000  # downsample threshold for accumulated views
_MAX_STATIC_CONTEXT_PTS = 500_000  # static-as-overlay in the dynamic viz
_MAX_GLOBAL_MAP_PTS = 1_000_000
_WIN_W = 1600
_WIN_H = 900

_STATIC_RGB = [0.29, 0.56, 0.85]  # blue
_DYNAMIC_RGB = [0.91, 0.30, 0.24]  # red
_GROUND_RGB = [0.20, 1.00, 0.20]  # green
_STATIC_CONTEXT_RGB = [0.30, 0.30, 0.30]  # gray (faint backdrop)
_EGO_RGB = [1.00, 0.20, 0.85]  # magenta
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


def _value_colors(values: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """Map per-point scalar values to RGB using a matplotlib colormap.

    Percentile-normalized (2–98) so a few outliers don't flatten the gradient.
    """
    import matplotlib.cm as cm

    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    lo, hi = np.percentile(values, 2), np.percentile(values, 98)
    norm = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return cm.get_cmap(cmap_name)(norm)[:, :3].astype(np.float64)


def _ego_trajectory_lineset(bag_id: str, chunk_id: str):
    """Polyline through the chunk's ego positions in world frame (magenta).

    Returns None if poses aren't loadable — the viz still opens without it.
    Useful in both static and dynamic views to anchor where the vehicle was
    relative to the classified points.
    """
    o3d = _o3d()
    try:
        from wato_lidar_preprocessing._inputs import load_pose_samples

        samples = load_pose_samples(bag_id, chunk_id)
    except Exception as exc:  # noqa: BLE001 — viz must survive missing poses
        log.warning("ego trajectory unavailable for chunk %s: %s", chunk_id, exc)
        return None
    if len(samples) < 2:
        return None
    pts = np.array([s.translation for s in samples], dtype=np.float64)
    lines = np.array([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(_EGO_RGB, (len(lines), 1)))
    return ls


# Classification → RGB mapping for voxel_diag overlay.
# Must stay aligned with CLASS_* codes in classify/log_odds.py.
_CLASS_COLORS: dict[int, list[float]] = {
    0: [0.30, 0.56, 0.85],   # STATIC          — blue
    1: [0.95, 0.85, 0.20],   # AMBIGUOUS       — yellow
    2: [0.95, 0.55, 0.15],   # UNDER_EVIDENCED — orange
    3: [0.55, 0.55, 0.55],   # FREE_ONLY       — gray
    4: [0.91, 0.30, 0.24],   # DYNAMIC         — red
}
_CLASS_MISSING_RGB = [0.20, 0.20, 0.20]


def _load_voxel_diag_sorted(bag_id: str, chunk_id: str):
    """Load voxel_diag.npz, sorted by key for searchsorted.

    Returns None when the file is missing (save_voxel_diagnostics=False);
    callers must handle None — the diag-based color modes just disable.
    """
    from wato_lidar_preprocessing.io import load_voxel_diag

    try:
        diag = load_voxel_diag(bag_id, chunk_id)
    except FileNotFoundError:
        return None
    keys = diag["keys"].astype(np.int64)
    order = np.argsort(keys)
    return {
        "keys": keys[order],
        "log_odds": diag["log_odds"][order],
        "p_occ": diag["p_occ"][order],
        "n_obs": diag["n_obs"][order],
        "n_hits": diag["n_hits"][order],
        "classification": diag["classification"][order],
        "origin": diag["origin"].astype(np.float64),
        "voxel_size": float(diag["voxel_size"]),
    }


def _voxel_diag_per_point(xyz: np.ndarray, diag) -> dict[str, np.ndarray] | None:
    """Look up each xyz point's voxel stats from a pre-sorted voxel_diag.

    Returns parallel arrays of length len(xyz). Misses (voxel not in diag)
    are NaN for floats, -1 for ints. Voxel-key packing mirrors voxel.py.
    """
    if diag is None or xyz.shape[0] == 0:
        return None
    AXIS_BITS = 20
    AXIS_RANGE = 1 << AXIS_BITS
    origin = diag["origin"]
    voxel_size = diag["voxel_size"]
    keys_sorted = diag["keys"]

    rel = xyz.astype(np.float64) - origin
    idx = np.floor(rel / voxel_size).astype(np.int64)
    in_range = (
        (idx[:, 0] >= 0) & (idx[:, 0] < AXIS_RANGE)
        & (idx[:, 1] >= 0) & (idx[:, 1] < AXIS_RANGE)
        & (idx[:, 2] >= 0) & (idx[:, 2] < AXIS_RANGE)
    )
    pt_keys = np.zeros(xyz.shape[0], dtype=np.int64)
    pt_keys[in_range] = (
        (idx[in_range, 0] << (2 * AXIS_BITS))
        | (idx[in_range, 1] << AXIS_BITS)
        | idx[in_range, 2]
    )

    if keys_sorted.size == 0:
        hit = np.zeros(xyz.shape[0], dtype=bool)
        pos_clip = np.zeros(xyz.shape[0], dtype=np.int64)
    else:
        pos = np.searchsorted(keys_sorted, pt_keys)
        pos_clip = np.clip(pos, 0, keys_sorted.size - 1)
        hit = (keys_sorted[pos_clip] == pt_keys) & in_range

    n = xyz.shape[0]
    log_odds = np.full(n, np.nan, dtype=np.float32)
    p_occ = np.full(n, np.nan, dtype=np.float32)
    n_obs = np.full(n, -1, dtype=np.int32)
    n_hits = np.full(n, -1, dtype=np.int32)
    classification = np.full(n, -1, dtype=np.int8)
    if hit.any():
        h_pos = pos_clip[hit]
        log_odds[hit] = diag["log_odds"][h_pos]
        p_occ[hit] = diag["p_occ"][h_pos]
        n_obs[hit] = diag["n_obs"][h_pos]
        n_hits[hit] = diag["n_hits"][h_pos]
        classification[hit] = diag["classification"][h_pos]

    n_miss = int((~hit).sum())
    if n_miss > 0:
        log.info(
            "voxel_diag lookup: %d/%d points (%.1f%%) had no entry — "
            "rare, usually means voxel_diag was written from a different "
            "voxel_size/origin than the dynamic_map.npz",
            n_miss,
            n,
            100.0 * n_miss / n,
        )
    return {
        "log_odds": log_odds,
        "p_occ": p_occ,
        "n_obs": n_obs,
        "n_hits": n_hits,
        "classification": classification,
        "hit": hit,
    }


def _p_occ_colors(p_occ: np.ndarray) -> np.ndarray:
    """Map p_occ to RGB using RdYlBu: red = carved (low), blue = static (high).

    Diverging colormap centred on 0.5: 0.0=dark red, 0.30=yellow (the
    p_dynamic_threshold edge), 0.70=blue (p_static_threshold edge).
    NaN points (no voxel_diag entry) get faint gray.
    """
    import matplotlib.cm as cm

    colors = np.tile([0.20, 0.20, 0.20], (p_occ.shape[0], 1)).astype(np.float64)
    valid = ~np.isnan(p_occ)
    if valid.any():
        v = np.clip(p_occ[valid], 0.0, 1.0).astype(np.float64)
        colors[valid] = cm.get_cmap("RdYlBu")(v)[:, :3]
    return colors


def _classification_colors(class_arr: np.ndarray) -> np.ndarray:
    """Map int8 classification codes to fixed RGB per category."""
    n = class_arr.shape[0]
    colors = np.tile(_CLASS_MISSING_RGB, (n, 1)).astype(np.float64)
    for code, rgb in _CLASS_COLORS.items():
        mask = class_arr == code
        if mask.any():
            colors[mask] = rgb
    return colors


def _log_p_occ_distribution(
    p_occ: np.ndarray, p_dynamic_threshold: float = 0.30
) -> None:
    """Log percentile breakdown for the currently shown p_occ values.

    Two action-relevant buckets:
      - deeply-carved (<0.10) → real carving (pose drift, beam noise)
      - threshold-edge (p_dynamic ±0.05) → consider dropping p_dynamic_threshold
    """
    valid = p_occ[~np.isnan(p_occ)]
    if valid.size == 0:
        log.info("p_occ distribution: no valid voxel_diag lookups")
        return
    pcts = np.percentile(valid, [10, 25, 50, 75, 90])
    n_deep = int((valid < 0.10).sum())
    edge_lo = max(0.0, p_dynamic_threshold - 0.05)
    edge_hi = min(1.0, p_dynamic_threshold + 0.05)
    n_edge = int(((valid >= edge_lo) & (valid <= edge_hi)).sum())
    log.info(
        "p_occ on %d shown points: p10=%.3f p25=%.3f median=%.3f p75=%.3f p90=%.3f "
        "| deeply-carved (<0.10): %d (%.1f%%) "
        "| threshold-edge ([%.2f, %.2f]): %d (%.1f%%)",
        valid.size,
        pcts[0],
        pcts[1],
        pcts[2],
        pcts[3],
        pcts[4],
        n_deep,
        100.0 * n_deep / valid.size,
        edge_lo,
        edge_hi,
        n_edge,
        100.0 * n_edge / valid.size,
    )


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
    extra_key_callbacks: dict[str, Callable] | None = None,
    hidden_at_start: Iterable[str] | None = None,
) -> None:
    """Open an interactive Open3D window. Blocks until closed.

    geoms: {name: geometry} — added initially.  Names listed in
        ``hidden_at_start`` are *built* but not added to the visualizer until
        the user presses their toggle key (useful for heavy backdrop layers
        you only want on demand).
    toggle_keys: {key_char: geom_name} — pressing the key hides/shows the geometry.
    extra_key_callbacks: {key_char: callable(vis) -> bool} — for custom actions
        that aren't simple add/remove toggles (color-mode swaps, lazy
        compute-then-show, etc.).  The callback receives the visualizer and
        should return False unless it wants the render to be marked dirty.

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

    hidden = set(hidden_at_start or ())
    visible = {name: (name not in hidden) for name in nonempty}
    for name, g in nonempty.items():
        if visible[name]:
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

    if extra_key_callbacks:
        for key, cb in extra_key_callbacks.items():
            vis.register_key_callback(ord(key.upper()), cb)

    # Start in top-down BEV — most useful default for ground vehicle scans.
    _set_view(vis, [0, 0, -1], [0, 1, 0], center, 0.4)

    toggle_help = (
        f" | toggle: {', '.join(f'{k}={n}' for k, n in (toggle_keys or {}).items())}"
        if toggle_keys
        else ""
    )
    extra_help = (
        f" | actions: {','.join(sorted((extra_key_callbacks or {}).keys()))}"
        if extra_key_callbacks
        else ""
    )
    log.info(
        "viewer open: %s — keys: 1=top 2=front 3=side 4=iso R=reset +/-=ptsize%s%s — close to continue",
        title,
        toggle_help,
        extra_help,
    )

    vis.run()
    vis.destroy_window()


def viz_chunk_static(bag_id: str, chunk_id: str) -> None:
    """Window 1 of 2 — static cloud (blue) with optional ego trajectory overlay.

    All sweeps in the chunk, merged: this is the persistent-scene
    reconstruction (no road surface, no moving objects, no low-confidence
    points). Ego trajectory is drawn in magenta so you can see where the
    vehicle was relative to the static reconstruction.

    Toggle keys:  S = static cloud   E = ego trajectory
    """
    o3d = _o3d()
    from wato_lidar_preprocessing.io import load_static_map

    static_xyz = load_static_map(bag_id, chunk_id)["xyz"]
    (static_xyz,) = _downsample(static_xyz)

    static_pcd = o3d.geometry.PointCloud()
    static_pcd.points = o3d.utility.Vector3dVector(static_xyz)
    static_pcd.paint_uniform_color(_STATIC_RGB)

    geoms = {"static": static_pcd}
    toggle = {"S": "static"}

    ego = _ego_trajectory_lineset(bag_id, chunk_id)
    if ego is not None:
        geoms["ego_trajectory"] = ego
        toggle["E"] = "ego_trajectory"

    title = f"chunk {chunk_id} — STATIC cloud ({len(static_xyz):,} pts, blue)"
    _show(geoms, title, toggle_keys=toggle)


def viz_chunk_dynamic(bag_id: str, chunk_id: str) -> None:
    """Sweep-by-sweep dynamic cloud viewer with an interactive slider.

    Opens a gui window with the dynamic points painted red and a horizontal
    slider at the bottom that scrubs through every sweep in the chunk.
    Drag the slider (or hit ◀ / ▶, or press Play) and the cloud updates
    fluidly in place.

    Controls:
      • slider           current sweep
      • ◀ / ▶  buttons   step ±1 sweep
      • Play / Pause     auto-advance
      • speed slider     playback rate (sweeps/sec)
      • Mode             "single sweep" (default), "cumulative" (all sweeps
                         ≤ N coloured by sweep_id turbo), or "trail" (last K)
      • Static / Ego     overlay toggles
    """
    if not _has_dynamic_data(bag_id, chunk_id):
        return
    try:
        _run_sweep_slider(bag_id, chunk_id)
    except Exception as exc:  # noqa: BLE001 — fall back to the legacy aggregated view
        log.warning(
            "sweep-slider viewer failed (%s); falling back to aggregated view", exc
        )
        _viz_chunk_dynamic_aggregated(bag_id, chunk_id)


def _has_dynamic_data(bag_id: str, chunk_id: str) -> bool:
    from wato_lidar_preprocessing.io import load_dynamic_map

    try:
        dyn = load_dynamic_map(bag_id, chunk_id)
    except FileNotFoundError:
        log.warning("no dynamic_map.npz for chunk %s — nothing to visualize", chunk_id)
        return False
    if dyn["xyz"].shape[0] == 0:
        log.warning("dynamic cloud empty for chunk %s — nothing to visualize", chunk_id)
        return False
    if "sweep_id" not in dyn:
        log.warning(
            "dynamic_map.npz for chunk %s has no sweep_id field — "
            "cannot slider-scrub; opening aggregated view instead",
            chunk_id,
        )
        _viz_chunk_dynamic_aggregated(bag_id, chunk_id)
        return False
    return True


def _run_sweep_slider(bag_id: str, chunk_id: str) -> None:
    """Build the gui.SceneWidget + slider window and block until closed."""
    o3d = _o3d()
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    import threading
    import time

    from wato_lidar_preprocessing.io import load_dynamic_map, load_static_map

    dyn = load_dynamic_map(bag_id, chunk_id)
    xyz = dyn["xyz"]
    sweep_id_arr = dyn["sweep_id"].astype(np.int64)
    intensity_arr = dyn.get("intensity")

    # Sort by sweep_id so per-sweep slices are contiguous (boolean masking
    # every slider tick is too slow for big chunks).
    order = np.argsort(sweep_id_arr, kind="stable")
    xyz = xyz[order]
    sweep_id_arr = sweep_id_arr[order]
    if intensity_arr is not None:
        intensity_arr = intensity_arr[order]

    unique_sweeps, starts, counts = np.unique(
        sweep_id_arr, return_index=True, return_counts=True
    )
    sweep_min = int(unique_sweeps.min())
    sweep_max = int(unique_sweeps.max())
    sweep_slices: dict[int, tuple[int, int]] = {
        int(s): (int(st), int(st + c)) for s, st, c in zip(unique_sweeps, starts, counts)
    }
    # Precompute turbo colors so cumulative mode doesn't recompute per tick.
    sweep_colors_all = _value_colors(sweep_id_arr.astype(np.float64), "turbo")

    # Load voxel_diag once at viewer open so color-mode switches are O(1)
    # lookups. None → diag-based color modes stay disabled.
    diag_sorted = _load_voxel_diag_sorted(bag_id, chunk_id)
    diag_stats = _voxel_diag_per_point(xyz, diag_sorted) if diag_sorted else None
    has_diag = diag_stats is not None
    if not has_diag:
        log.warning(
            "voxel_diag.npz missing for chunk %s — the p_occ / classification / "
            "n_obs / n_hits color modes will be UNAVAILABLE in this window. "
            "To enable them, set save_voxel_diagnostics: true in "
            "lidar_preprocessing.yaml (already done if you pulled the latest "
            "config) and re-run with --force:\n"
            "    ./watod run lidar_preprocessing run --bag %s --chunk %s -f",
            chunk_id,
            bag_id,
            chunk_id,
        )
    if has_diag:
        p_occ_colors_all = _p_occ_colors(diag_stats["p_occ"])
        n_obs_colors_all = _value_colors(
            diag_stats["n_obs"].astype(np.float64), "viridis"
        )
        n_hits_colors_all = _value_colors(
            diag_stats["n_hits"].astype(np.float64), "viridis"
        )
        class_colors_all = _classification_colors(diag_stats["classification"])
    else:
        p_occ_colors_all = None
        n_obs_colors_all = None
        n_hits_colors_all = None
        class_colors_all = None

    # Scene-wide bounds so the camera frames the whole chunk regardless of
    # which sweep is current.
    scene_min = xyz.min(axis=0)
    scene_max = xyz.max(axis=0)
    scene_center = (scene_min + scene_max) * 0.5
    scene_bbox = o3d.geometry.AxisAlignedBoundingBox(scene_min, scene_max)

    # Static backdrop is optional; off by default.
    static_xyz = None
    try:
        sx = load_static_map(bag_id, chunk_id)["xyz"]
        if sx.shape[0] > 0:
            (sx,) = _downsample(sx, max_pts=_MAX_STATIC_CONTEXT_PTS)
            static_xyz = sx
    except FileNotFoundError:
        pass

    ego = _ego_trajectory_lineset(bag_id, chunk_id)

    app = gui.Application.instance
    app.initialize()  # idempotent across chunks
    title = f"chunk {chunk_id} — DYNAMIC sweep slider"
    window = app.create_window(title, _WIN_W, _WIN_H)

    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.08, 0.08, 0.08, 1.0])

    dyn_mat = rendering.MaterialRecord()
    dyn_mat.shader = "defaultUnlit"
    dyn_mat.point_size = 3.0

    static_mat = rendering.MaterialRecord()
    static_mat.shader = "defaultUnlit"
    static_mat.point_size = 1.5

    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 2.0

    axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(float(np.linalg.norm(scene_max - scene_min)) * 0.02, 1.0)
    )
    axis_mat = rendering.MaterialRecord()
    axis_mat.shader = "defaultLit"
    scene_widget.scene.add_geometry("__axis", axis_mesh, axis_mat)

    if static_xyz is not None:
        spcd = o3d.geometry.PointCloud()
        spcd.points = o3d.utility.Vector3dVector(static_xyz)
        spcd.paint_uniform_color(_STATIC_CONTEXT_RGB)
        scene_widget.scene.add_geometry("__static", spcd, static_mat)
        scene_widget.scene.show_geometry("__static", False)

    if ego is not None:
        scene_widget.scene.add_geometry("__ego", ego, line_mat)
        scene_widget.scene.show_geometry("__ego", False)

    # Mutable state shared by all GUI callbacks.
    state = {
        "sweep": sweep_min,
        "mode": "single",  # "single" | "cumulative" | "trail"
        "trail_k": 5,
        "color_by": "uniform",  # "uniform" | "intensity"
        "playing": False,
        "fps": 8.0,
    }

    def _slice_color(color_by: str, lo: int, hi: int) -> np.ndarray:
        """Per-point colors for the contiguous [lo, hi) range."""
        n = hi - lo
        if color_by == "p_occ" and p_occ_colors_all is not None:
            return p_occ_colors_all[lo:hi]
        if color_by == "n_obs" and n_obs_colors_all is not None:
            return n_obs_colors_all[lo:hi]
        if color_by == "n_hits" and n_hits_colors_all is not None:
            return n_hits_colors_all[lo:hi]
        if color_by == "classification" and class_colors_all is not None:
            return class_colors_all[lo:hi]
        if color_by == "intensity" and intensity_arr is not None:
            return _value_colors(intensity_arr[lo:hi].astype(np.float64), "viridis")
        if color_by == "sweep_id":
            return sweep_colors_all[lo:hi]
        return np.tile(_DYNAMIC_RGB, (n, 1))

    def points_for_sweep(sid: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (xyz, colors-or-None) for the current mode at sweep sid."""
        mode = state["mode"]
        color_by = state["color_by"]
        if mode == "single":
            if sid not in sweep_slices:
                return np.zeros((0, 3)), None
            lo, hi = sweep_slices[sid]
            return xyz[lo:hi], _slice_color(color_by, lo, hi)
        if mode == "cumulative":
            idx = int(np.searchsorted(unique_sweeps, sid, side="right"))
            if idx == 0:
                return np.zeros((0, 3)), None
            last_sid = int(unique_sweeps[idx - 1])
            _, hi = sweep_slices[last_sid]
            # Default to sweep_id gradient when the user picked "uniform".
            cb = color_by if color_by != "uniform" else "sweep_id"
            return xyz[:hi], _slice_color(cb, 0, hi)
        if mode == "trail":
            k = state["trail_k"]
            lo_sid = sid - k + 1
            idx_lo = int(np.searchsorted(unique_sweeps, lo_sid, side="left"))
            idx_hi = int(np.searchsorted(unique_sweeps, sid, side="right"))
            if idx_lo >= idx_hi:
                return np.zeros((0, 3)), None
            first_sid = int(unique_sweeps[idx_lo])
            last_sid = int(unique_sweeps[idx_hi - 1])
            lo = sweep_slices[first_sid][0]
            hi = sweep_slices[last_sid][1]
            cb = color_by if color_by != "uniform" else "sweep_id"
            return xyz[lo:hi], _slice_color(cb, lo, hi)
        return np.zeros((0, 3)), None

    def update_cloud():
        pts, cols = points_for_sweep(state["sweep"])
        if scene_widget.scene.has_geometry("__dyn"):
            scene_widget.scene.remove_geometry("__dyn")
        if len(pts) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            if cols is not None:
                pcd.colors = o3d.utility.Vector3dVector(cols)
            scene_widget.scene.add_geometry("__dyn", pcd, dyn_mat)
        count_label.text = f"sweep {state['sweep']} / {sweep_max}  ·  {len(pts):,} pts"

    em = window.theme.font_size
    panel = gui.Vert(0.5 * em, gui.Margins(em, 0.5 * em, em, 0.5 * em))

    count_label = gui.Label(f"sweep {sweep_min} / {sweep_max}  ·  ... pts")
    sweep_slider = gui.Slider(gui.Slider.INT)
    sweep_slider.set_limits(sweep_min, sweep_max)
    sweep_slider.int_value = sweep_min

    def on_sweep(v):
        state["sweep"] = int(v)
        update_cloud()

    sweep_slider.set_on_value_changed(on_sweep)

    btn_prev = gui.Button("◀")
    btn_play = gui.Button("Play")
    btn_next = gui.Button("▶")
    btn_prev.horizontal_padding_em = 0.6
    btn_next.horizontal_padding_em = 0.6
    btn_play.horizontal_padding_em = 0.8

    def step(delta: int):
        new = state["sweep"] + delta
        if new < sweep_min:
            new = sweep_max
        elif new > sweep_max:
            new = sweep_min
        state["sweep"] = new
        sweep_slider.int_value = new
        update_cloud()

    btn_prev.set_on_clicked(lambda: step(-1))
    btn_next.set_on_clicked(lambda: step(+1))

    speed_label = gui.Label("speed")
    speed_slider = gui.Slider(gui.Slider.DOUBLE)
    speed_slider.set_limits(1.0, 30.0)
    speed_slider.double_value = state["fps"]

    def on_speed(v):
        state["fps"] = float(v)

    speed_slider.set_on_value_changed(on_speed)

    # Play loop is a background thread; UI updates must go through
    # post_to_main_thread because Open3D's gui isn't reentrant.
    stop_evt = threading.Event()

    def play_loop():
        while not stop_evt.is_set() and state["playing"]:
            time.sleep(max(1.0 / max(state["fps"], 0.1), 0.01))
            if not state["playing"]:
                break
            app.post_to_main_thread(window, lambda: step(+1))

    play_thread = {"t": None}

    def toggle_play():
        state["playing"] = not state["playing"]
        btn_play.text = "Pause" if state["playing"] else "Play"
        if state["playing"]:
            t = threading.Thread(target=play_loop, daemon=True)
            play_thread["t"] = t
            t.start()

    btn_play.set_on_clicked(toggle_play)

    transport = gui.Horiz(0.4 * em)
    transport.add_child(btn_prev)
    transport.add_child(btn_play)
    transport.add_child(btn_next)
    transport.add_fixed(em)
    transport.add_child(speed_label)
    transport.add_child(speed_slider)

    mode_label = gui.Label("mode")
    mode_combo = gui.Combobox()
    mode_combo.add_item("single sweep")
    mode_combo.add_item("cumulative")
    mode_combo.add_item(f"trail (last {state['trail_k']})")
    mode_combo.selected_index = 0

    def on_mode(text, idx):
        state["mode"] = ["single", "cumulative", "trail"][idx]
        update_cloud()

    mode_combo.set_on_selection_changed(on_mode)

    color_label = gui.Label("color")
    color_combo = gui.Combobox()
    # Order = most-useful debug modes first; p_occ is the headline diagnostic.
    color_options: list[tuple[str, str]] = [("red (uniform)", "uniform")]
    if has_diag:
        color_options.append(("p_occ (carving evidence)", "p_occ"))
        color_options.append(("classification (5-bucket)", "classification"))
        color_options.append(("n_obs (sweeps that touched voxel)", "n_obs"))
        color_options.append(("n_hits (endpoint hits in voxel)", "n_hits"))
    if intensity_arr is not None:
        color_options.append(("intensity", "intensity"))
    color_options.append(("sweep_id (turbo)", "sweep_id"))
    for label, _key in color_options:
        color_combo.add_item(label)
    color_combo.selected_index = 0

    def on_color(text, idx):
        new_mode = color_options[idx][1]
        state["color_by"] = new_mode
        update_cloud()
        # Log p_occ distribution when the user switches to it so they can
        # see immediately where the dynamic points sit in the band.
        if new_mode == "p_occ" and diag_stats is not None:
            pts, _cols = points_for_sweep(state["sweep"])
            if len(pts):
                _log_p_occ_distribution(diag_stats["p_occ"])

    color_combo.set_on_selection_changed(on_color)

    static_chk = gui.Checkbox("static backdrop")
    static_chk.checked = False
    static_chk.enabled = static_xyz is not None

    def on_static(checked):
        if scene_widget.scene.has_geometry("__static"):
            scene_widget.scene.show_geometry("__static", checked)

    static_chk.set_on_checked(on_static)

    ego_chk = gui.Checkbox("ego path")
    ego_chk.checked = False
    ego_chk.enabled = ego is not None

    def on_ego(checked):
        if scene_widget.scene.has_geometry("__ego"):
            scene_widget.scene.show_geometry("__ego", checked)

    ego_chk.set_on_checked(on_ego)

    btn_reset = gui.Button("reset view")
    btn_reset.horizontal_padding_em = 0.4

    def reset_view():
        scene_widget.setup_camera(60.0, scene_bbox, scene_center)

    btn_reset.set_on_clicked(reset_view)

    options = gui.Horiz(0.4 * em)
    options.add_child(mode_label)
    options.add_child(mode_combo)
    options.add_fixed(em)
    options.add_child(color_label)
    options.add_child(color_combo)
    options.add_fixed(em)
    options.add_child(static_chk)
    options.add_child(ego_chk)
    options.add_stretch()
    options.add_child(btn_reset)

    panel.add_child(count_label)
    panel.add_child(sweep_slider)
    panel.add_child(transport)
    panel.add_child(options)

    def on_layout(layout_context):
        r = window.content_rect
        panel_height = int(round(7.5 * em))
        scene_widget.frame = gui.Rect(
            r.x, r.y, r.width, r.height - panel_height
        )
        panel.frame = gui.Rect(
            r.x, r.y + r.height - panel_height, r.width, panel_height
        )

    window.set_on_layout(on_layout)
    window.add_child(scene_widget)
    window.add_child(panel)

    def on_close():
        state["playing"] = False
        stop_evt.set()
        return True

    window.set_on_close(on_close)

    update_cloud()
    scene_widget.setup_camera(60.0, scene_bbox, scene_center)
    scene_widget.look_at(scene_center, scene_center + np.array([0, 0, 80.0]), [0, 1, 0])

    log.info(
        "viewer open: %s — drag the slider or press Play; close to continue",
        title,
    )
    app.run()


def _viz_chunk_dynamic_aggregated(bag_id: str, chunk_id: str) -> None:
    """Legacy aggregated dynamic view (turbo by sweep_id, key-toggle overlays).

    Used as a fallback when the gui-based slider viewer can't run (missing
    sweep_id field, gui crash, etc.). Same toggles as the historical viewer:
    D dynamic, S static backdrop, E ego, B DBSCAN boxes, T/I/U color modes.
    """
    o3d = _o3d()
    from wato_lidar_preprocessing.io import load_dynamic_map, load_static_map

    try:
        dyn = load_dynamic_map(bag_id, chunk_id)
    except FileNotFoundError:
        return

    dyn_xyz = dyn["xyz"]
    if dyn_xyz.shape[0] == 0:
        return
    dyn_sweep_id = dyn.get("sweep_id")
    dyn_intensity = dyn.get("intensity")

    parallel = []
    if dyn_sweep_id is not None:
        parallel.append(dyn_sweep_id)
    if dyn_intensity is not None:
        parallel.append(dyn_intensity)
    out = _downsample(dyn_xyz, *parallel)
    dyn_xyz = out[0]
    i = 1
    if dyn_sweep_id is not None:
        dyn_sweep_id = out[i]
        i += 1
    if dyn_intensity is not None:
        dyn_intensity = out[i]

    dyn_pcd = o3d.geometry.PointCloud()
    dyn_pcd.points = o3d.utility.Vector3dVector(dyn_xyz)
    if dyn_sweep_id is not None:
        dyn_pcd.colors = o3d.utility.Vector3dVector(
            _value_colors(dyn_sweep_id.astype(np.float64), "turbo")
        )
    else:
        dyn_pcd.paint_uniform_color(_DYNAMIC_RGB)

    geoms = {"dynamic": dyn_pcd}
    toggle = {"D": "dynamic"}
    hidden: set[str] = set()

    try:
        static_xyz = load_static_map(bag_id, chunk_id)["xyz"]
        if static_xyz.shape[0] > 0:
            (static_xyz,) = _downsample(static_xyz, max_pts=_MAX_STATIC_CONTEXT_PTS)
            static_pcd = o3d.geometry.PointCloud()
            static_pcd.points = o3d.utility.Vector3dVector(static_xyz)
            static_pcd.paint_uniform_color(_STATIC_CONTEXT_RGB)
            geoms["static_context"] = static_pcd
            toggle["S"] = "static_context"
            hidden.add("static_context")
    except FileNotFoundError:
        pass

    ego = _ego_trajectory_lineset(bag_id, chunk_id)
    if ego is not None:
        geoms["ego_trajectory"] = ego
        toggle["E"] = "ego_trajectory"

    title = f"chunk {chunk_id} — DYNAMIC cloud aggregated ({len(dyn_xyz):,} pts)"
    _show(geoms, title, toggle_keys=toggle, hidden_at_start=hidden)


# Per-sweep views (opt-in via --sweep).


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


# Stage C / D — ground grid + bag-level global map.


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


# Orchestrator.


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
        # Static first (sets the spatial reference), then dynamic.
        if do("B") and has_o3d:
            try:
                viz_chunk_static(bag_id, chunk_id)
            except Exception as exc:
                log.warning("static accumulated view failed: %s", exc)
            try:
                viz_chunk_dynamic(bag_id, chunk_id)
            except Exception as exc:
                log.warning("dynamic accumulated view failed: %s", exc)
        # Stage A is per-sweep only; without --sweep we point the user at it.
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
