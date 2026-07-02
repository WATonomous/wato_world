"""Visualization for lidar_preprocessing point-cloud artifacts.

A small multi-backend viewer modelled on a standalone ``dynamic_map.npz``
viewer, but loading clouds from the artifact store by ``(bag_id, chunk_id)``
rather than a raw filesystem path.

Layers (what to look at):
  - ``dynamic`` — per-chunk dynamic cloud (``dynamic_map.npz``): xyz, sweep_id,
    intensity
  - ``static``  — per-chunk static cloud (``static_map.npz``): xyz, intensity
  - ``global``  — bag-level global static map (``global_static_map.npz``): xyz
  - ``ground``  — Stage C height/normal grids (matplotlib, 2D)

Color modes mirror the source script: ``height`` (RdYlBu), ``sweep_id``
(turbo), ``intensity`` (turbo, 2–98 percentile normalised). Only the modes a
layer actually has data for are offered.

Backends:
  - ``open3d``      interactive window; press ``c`` to cycle color, ``+``/``-``
                    to grow/shrink points. Default on Linux.
  - ``plotly``      browser-based 3D scatter; default on macOS where Open3D's
                    GLFW/Cocoa window segfaults.
  - ``matplotlib``  software 3D scatter fallback.

All point-cloud backends need a display (or WSLg). In Docker that means DISPLAY
(and on WSL2 /mnt/wslg + /tmp/.X11-unix) bind-mounted in — see
``lidar_preprocessing_dev`` in modules/docker-compose.dev.yaml.
"""

from __future__ import annotations

import logging
import sys
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

# Map-frame clouds are large; cap what we push to the GPU / scatter so the
# window stays interactive. Open3D handles more than the browser/software paths.
_DEFAULT_MAX_POINTS = 2_000_000
_PLOTLY_MAX_POINTS = 200_000  # browser gl3d gets sluggish past this
_MATPLOTLIB_MAX_POINTS = 300_000  # software 3d scatter chokes well before o3d

_COLOR_MODES = ("height", "sweep_id", "intensity")


# --------------------------------------------------------------------------- #
# colouring
# --------------------------------------------------------------------------- #
def _cmap(name: str):
    """Fetch a colormap across matplotlib versions (modern API, old fallback)."""
    import matplotlib

    try:
        return matplotlib.colormaps[name]  # mpl >= 3.5
    except AttributeError:  # pragma: no cover - very old matplotlib
        import matplotlib.cm as cm

        return cm.get_cmap(name)


def _value_colors(values: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """Per-point scalar -> RGB, percentile-normalised (2–98) like the repo viz."""
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    lo, hi = np.percentile(values, 2), np.percentile(values, 98)
    norm = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return _cmap(cmap_name)(norm)[:, :3].astype(np.float64)


def _height_colors(z: np.ndarray) -> np.ndarray:
    """Z -> RdYlBu (low=red, mid=yellow, high=blue), matching lidar_preprocessing."""
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    norm = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return _cmap("RdYlBu")(1.0 - norm)[:, :3].astype(np.float64)


def _colors_for(arrays: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode == "height":
        return _height_colors(arrays["xyz"][:, 2])
    if mode == "sweep_id":
        return _value_colors(arrays["sweep_id"].astype(np.float64), "turbo")
    if mode == "intensity":
        return _value_colors(arrays["intensity"].astype(np.float64), "turbo")
    raise ValueError(f"unknown color mode: {mode}")


def _available_modes(arrays: dict[str, np.ndarray]) -> list[str]:
    """Color modes the cloud actually has data for, in cycle order."""
    modes = ["height"]
    if "sweep_id" in arrays:
        modes.append("sweep_id")
    if "intensity" in arrays:
        modes.append("intensity")
    return modes


# --------------------------------------------------------------------------- #
# loading + downsampling
# --------------------------------------------------------------------------- #
def _arrays_from_npz(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalise a loaded npz dict to the {xyz[, sweep_id][, intensity]} shape."""
    xyz = np.asarray(data["xyz"], dtype=np.float64)
    out: dict[str, np.ndarray] = {"xyz": xyz}
    if "sweep_id" in data and data["sweep_id"].size == len(xyz):
        out["sweep_id"] = np.asarray(data["sweep_id"])
    if "intensity" in data and data["intensity"].size == len(xyz):
        out["intensity"] = np.asarray(data["intensity"], dtype=np.float64)
    return out


def _downsample(
    arrays: dict[str, np.ndarray], max_points: int
) -> dict[str, np.ndarray]:
    """Uniform random subsample (fixed seed) when over the cap."""
    n = len(arrays["xyz"])
    if max_points <= 0 or n <= max_points:
        return arrays
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    log.info("downsampling %s -> %s points for display", f"{n:,}", f"{max_points:,}")
    return {k: v[idx] for k, v in arrays.items()}


def _summary(arrays: dict[str, np.ndarray]) -> str:
    xyz = arrays["xyz"]
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    lines = [
        f"  points    : {len(xyz):,}",
        f"  x range   : [{lo[0]:8.2f}, {hi[0]:8.2f}]  span {hi[0] - lo[0]:.2f} m",
        f"  y range   : [{lo[1]:8.2f}, {hi[1]:8.2f}]  span {hi[1] - lo[1]:.2f} m",
        f"  z range   : [{lo[2]:8.2f}, {hi[2]:8.2f}]  span {hi[2] - lo[2]:.2f} m",
    ]
    if "sweep_id" in arrays:
        sw = arrays["sweep_id"]
        lines.append(
            f"  sweep_id  : {int(sw.min())}..{int(sw.max())} "
            f"({len(np.unique(sw))} unique)"
        )
    if "intensity" in arrays:
        it = arrays["intensity"]
        lines.append(f"  intensity : [{it.min():.2f}, {it.max():.2f}]")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def _view_open3d(
    arrays: dict[str, np.ndarray],
    modes: Sequence[str],
    mode: str,
    point_size: float,
    title: str,
) -> None:
    import open3d as o3d

    state = {"mode_i": list(modes).index(mode), "size": point_size}
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arrays["xyz"])

    def apply_colors() -> None:
        m = modes[state["mode_i"]]
        pcd.colors = o3d.utility.Vector3dVector(_colors_for(arrays, m))
        log.info("color mode: %s", m)

    apply_colors()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1600, height=900)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = state["size"]
    opt.background_color = np.array([0.06, 0.06, 0.08])

    def cycle_color(v):
        state["mode_i"] = (state["mode_i"] + 1) % len(modes)
        apply_colors()
        v.update_geometry(pcd)
        return False

    def grow(v):
        state["size"] = min(state["size"] + 1.0, 12.0)
        v.get_render_option().point_size = state["size"]
        return False

    def shrink(v):
        state["size"] = max(state["size"] - 1.0, 1.0)
        v.get_render_option().point_size = state["size"]
        return False

    vis.register_key_callback(ord("C"), cycle_color)
    vis.register_key_callback(ord("="), grow)  # '+' without shift
    vis.register_key_callback(ord("+"), grow)
    vis.register_key_callback(ord("-"), shrink)

    log.info(
        "viewer open: %s — keys: [c] cycle color  [+/-] point size  "
        "[R] reset view  [q] quit — close to continue",
        title,
    )
    vis.run()
    vis.destroy_window()


# Per-mode plotly scalar + colorscale (plotly maps the values itself, so we
# skip the per-point RGB conversion and get a colorbar for free).
_PLOTLY_CFG = {
    "height": ("z [m]", "RdBu", True),  # reversed: low=blue, high=red (elevation)
    "sweep_id": ("sweep_id", "Turbo", False),
    "intensity": ("intensity", "Turbo", False),
}


def _view_plotly(
    arrays: dict[str, np.ndarray],
    modes: Sequence[str],
    mode: str,
    point_size: float,
    title: str,
) -> None:
    """Interactive 3D scatter in the browser — reliable where Open3D's
    GLFW/Cocoa window segfaults (recent macOS)."""
    import plotly.graph_objects as go

    arrays = _downsample(arrays, _PLOTLY_MAX_POINTS)
    xyz = arrays["xyz"]
    scalar = {
        "height": xyz[:, 2],
        "sweep_id": arrays.get("sweep_id", np.zeros(len(xyz))).astype(np.float64),
        "intensity": arrays.get("intensity", np.zeros(len(xyz))),
    }[mode]
    label, cmap, reverse = _PLOTLY_CFG[mode]

    fig = go.Figure(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                color=scalar,
                colorscale=cmap,
                reversescale=reverse,
                opacity=0.85,
                colorbar=dict(title=label),
            ),
        )
    )
    fig.update_layout(
        title=f"{title} — color: {mode}  ({len(xyz):,} pts)",
        scene=dict(
            aspectmode="data",  # true metric proportions
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#101014",
        font=dict(color="#d8d8e0"),
    )
    log.info("opening plotly figure in your browser…")
    fig.show()


def _view_matplotlib(
    arrays: dict[str, np.ndarray],
    modes: Sequence[str],
    mode: str,
    point_size: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt  # noqa: F401  (registers 3d projection)

    arrays = _downsample(arrays, _MATPLOTLIB_MAX_POINTS)
    xyz = arrays["xyz"]
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=_colors_for(arrays, mode),
        s=point_size,
        marker=".",
        linewidths=0,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"{title} — color: {mode}")
    try:
        ax.set_box_aspect(np.ptp(xyz, axis=0))  # true metric aspect
    except Exception:  # noqa: BLE001 — older matplotlib lacks set_box_aspect
        pass
    plt.tight_layout()
    plt.show()


_BACKENDS = {
    "open3d": _view_open3d,
    "plotly": _view_plotly,
    "matplotlib": _view_matplotlib,
}


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    # Open3D's GLFW/Cocoa window segfaults on recent macOS, so prefer the
    # browser-based plotly backend there; trust Open3D on Linux.
    resolved = "plotly" if sys.platform == "darwin" else "open3d"
    log.info("backend: %s (auto)", resolved)
    return resolved


def view_cloud(
    arrays: dict[str, np.ndarray],
    title: str,
    color: str = "height",
    backend: str = "auto",
    point_size: float = 2.0,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> None:
    """Show a single cloud with the chosen backend.

    ``arrays`` must hold ``xyz`` and may hold ``sweep_id`` / ``intensity``.
    ``color`` is clamped to a mode the cloud actually has data for.
    """
    if arrays["xyz"].shape[0] == 0:
        log.warning("nothing to show for %s — cloud is empty", title)
        return

    modes = _available_modes(arrays)
    if color not in modes:
        log.warning(
            "color mode %r unavailable for this cloud; falling back to %r",
            color,
            modes[0],
        )
        color = modes[0]

    log.info("%s\n%s", title, _summary(arrays))
    arrays = _downsample(arrays, max_points)
    _BACKENDS[_resolve_backend(backend)](arrays, modes, color, point_size, title)


# --------------------------------------------------------------------------- #
# artifact-store entry points
# --------------------------------------------------------------------------- #
def viz_chunk_dynamic(bag_id: str, chunk_id: str, **opts) -> None:
    """Per-chunk dynamic cloud (dynamic_map.npz)."""
    from wato_lidar_preprocessing.io import load_dynamic_map

    try:
        data = load_dynamic_map(bag_id, chunk_id)
    except FileNotFoundError:
        log.warning("no dynamic_map.npz for chunk %s — nothing to visualize", chunk_id)
        return
    arrays = _arrays_from_npz(data)
    view_cloud(arrays, f"chunk {chunk_id} — dynamic", **opts)


def viz_chunk_static(bag_id: str, chunk_id: str, **opts) -> None:
    """Per-chunk static cloud (static_map.npz)."""
    from wato_lidar_preprocessing.io import load_static_map

    try:
        data = load_static_map(bag_id, chunk_id)
    except FileNotFoundError:
        log.warning("no static_map.npz for chunk %s — nothing to visualize", chunk_id)
        return
    arrays = _arrays_from_npz(data)
    view_cloud(arrays, f"chunk {chunk_id} — static", **opts)


def viz_global_static_map(bag_id: str, **opts) -> None:
    """Bag-level global static map (global_static_map.npz)."""
    from wato_lidar_preprocessing.io import load_global_static_map

    xyz = load_global_static_map(bag_id)
    view_cloud(
        {"xyz": np.asarray(xyz, dtype=np.float64)}, f"bag {bag_id} — global", **opts
    )


# Back-compat alias for the old stage-D name.
viz_stage_D = viz_global_static_map


def viz_ground_grid(bag_id: str, chunk_id: str) -> None:
    """Stage C: height grid heatmap + HSV-encoded surface normals (matplotlib)."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from wato_lidar_preprocessing.io import load_ground

    g = load_ground(bag_id, chunk_id)
    status = g.get("status", b"ok")
    if isinstance(status, (bytes, np.bytes_)):
        status = status.decode()
    if status in ("skipped_no_ground_mask", "empty"):
        log.warning("ground status=%s for chunk %s — skipping", status, chunk_id)
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


def viz_chunk(
    bag_id: str,
    chunk_id: str,
    layer: str = "dynamic",
    color: str = "height",
    backend: str = "auto",
    point_size: float = 2.0,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> None:
    """Visualize one layer of a chunk.

    layer: ``dynamic`` | ``static`` | ``ground``.
    """
    if layer == "ground":
        viz_ground_grid(bag_id, chunk_id)
        return
    opts = dict(
        color=color, backend=backend, point_size=point_size, max_points=max_points
    )
    if layer == "static":
        viz_chunk_static(bag_id, chunk_id, **opts)
    else:
        viz_chunk_dynamic(bag_id, chunk_id, **opts)
