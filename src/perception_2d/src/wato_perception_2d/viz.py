"""Interactive depth-vs-image viewer for perception_2d ``depth_2d`` artifacts.

A matplotlib window (the 2D analogue of lidar_preprocessing's Open3D
``watod viz``) for eyeballing what Depth Anything V2 + the LiDAR-anchored
affine fit actually produced, frame by frame, against the RGB image:

  - frame slider   — scrub through one camera's stream
  - opacity slider — blend RGB ↔ depth heatmap (0 = pure image, 1 = pure depth)
  - SPLIT toggle   — a draggable curtain: RGB on the left, depth on the right
                     (drag anywhere on the image to move the divide)
  - Prev/Next cam  — switch camera when a chunk has more than one
  - keyboard:  ←/→ step frame · ↑/↓ opacity · space toggle split ·
               c cycle colormap · q close

Each ``depth_2d/<cam>/<seq:06d>.npz`` holds ``depth_m`` (H, W metric metres) plus
the fit diagnostics (``fit_status``, ``n_inliers``, ``rmse_inliers_m``); the
matching RGB frame comes from ``frame_index.parquet``.  Depth is normalised per
frame over its finite, positive pixels (2nd–98th percentile) so each frame uses
its full colour range; the colourbar reads true metres.  Pixels with no valid
depth (sky, fit holes) fall through to the underlying image.

Needs DISPLAY (or WSLg) — same as lidar_preprocessing viz.  On WSL2 you can
also just run it on the host, where matplotlib already has a GUI backend:

    PYTHONPATH=src/common/src:src/perception_2d/src \\
        python -m wato_perception_2d viz --bag <bag_id>
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict, defaultdict
from typing import Optional

import numpy as np

from wato_common.artifact_store import (
    chunks_index_path,
    depth_2d_path,
    local_path,
)
from wato_common.io.parquet_io import read_rows
from wato_perception_2d.io import CameraFrameInfo, load_frame_index

log = logging.getLogger(__name__)

# A frame entry to display: the camera-frame record + its on-disk depth npz.
_Entry = tuple[CameraFrameInfo, str]

_FIT_LABELS = {0: "ransac", 1: "fallback", 2: "none"}
# turbo reversed → near (small metres) reads warm/red, far reads cool/blue.
_COLORMAPS = ["turbo_r", "turbo", "magma", "viridis", "inferno", "Spectral"]
_MAX_CACHE = 12  # decoded frames kept resident (LRU); bounds memory on long chunks.

# Dark theme to match the Open3D viewer's charcoal background.
_BG = "#101014"
_PANEL = "#1c1c22"
_FG = "#d8d8e0"
_ACCENT = "#4f8cff"


# ---------------------------------------------------------------------------
# Pure helpers (no matplotlib) — unit-tested in tests/test_viz.py.
# ---------------------------------------------------------------------------
def _depth_to_norm(
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Normalise a metric-depth map to [0, 1] for colour mapping.

    Returns ``(norm, valid, vmin, vmax)`` where ``valid`` marks finite,
    positive pixels and ``norm`` is the robust (2nd–98th percentile) min-max
    scaling clamped to [0, 1] with invalid pixels set to 0.  When too few
    pixels are valid the range collapses to (0, 0) and ``norm`` is all zeros.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if int(valid.sum()) < 16:
        return np.zeros_like(depth), valid, 0.0, 0.0
    vals = depth[valid]
    vmin, vmax = (float(x) for x in np.percentile(vals, [2.0, 98.0]))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    norm = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
    norm[~valid] = 0.0
    return norm, valid, vmin, vmax


def _entries_for_camera(
    bag_id: str, chunk_id: str, cam_id: str, frames: list[CameraFrameInfo]
) -> list[_Entry]:
    """Frames of one camera that have a depth npz on disk, sorted by sequence."""
    out: list[_Entry] = []
    for f in sorted(frames, key=lambda x: x.camera_seq):
        if f.cam_id != cam_id:
            continue
        dpath = local_path(depth_2d_path(bag_id, chunk_id, cam_id, f.camera_seq))
        if os.path.exists(dpath):
            out.append((f, dpath))
    return out


def _cameras_with_depth(
    bag_id: str, chunk_id: str
) -> tuple[list[str], dict[str, list[CameraFrameInfo]]]:
    """Return (cam_ids that have ≥1 depth artifact, frames grouped by camera)."""
    by_cam: dict[str, list[CameraFrameInfo]] = defaultdict(list)
    for f in load_frame_index(bag_id, chunk_id):
        by_cam[f.cam_id].append(f)
    cams = sorted(
        c for c in by_cam if _entries_for_camera(bag_id, chunk_id, c, by_cam[c])
    )
    return cams, by_cam


# ---------------------------------------------------------------------------
# Frame decoding (image + depth → display-ready arrays), with an LRU cache.
# ---------------------------------------------------------------------------
def _load_rgb(path: str, max_side: int) -> Optional[np.ndarray]:
    try:
        from PIL import Image as PILImage

        img = PILImage.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to load image %s: %s", path, exc)
        return None
    return _cap_side(arr, max_side, nearest=False)


def _cap_side(arr: np.ndarray, max_side: int, *, nearest: bool) -> np.ndarray:
    """Downscale so the longest side ≤ max_side (no-op if already smaller)."""
    h, w = arr.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return np.ascontiguousarray(arr)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    try:
        import cv2

        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        return cv2.resize(arr, (new_w, new_h), interpolation=interp)
    except Exception:  # noqa: BLE001 — cv2 missing/unhappy: PIL fallback.
        from PIL import Image as PILImage

        mode_arr = (
            (arr * 255.0).clip(0, 255).astype(np.uint8)
            if arr.dtype != np.uint8
            else arr
        )
        resized = PILImage.fromarray(mode_arr).resize(
            (new_w, new_h),
            PILImage.NEAREST if nearest else PILImage.BILINEAR,
        )
        out = np.asarray(resized, dtype=np.float32)
        return out / 255.0 if arr.dtype != np.uint8 else out


class _FrameStore:
    """LRU-cached decode of (image, depth) → display arrays for one camera."""

    def __init__(self, entries: list[_Entry], max_side: int) -> None:
        self.entries = entries
        self.max_side = max_side
        self._cache: "OrderedDict[int, dict]" = OrderedDict()

    def get(self, idx: int) -> Optional[dict]:
        """Return a decoded frame dict, or None when the image can't be read.

        Dict keys: base (H,W,3 float), norm (H,W float), valid (H,W bool),
        and meta (seq, vmin, vmax, fit_status, n_inliers, rmse, valid_frac).
        """
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        frame, dpath = self.entries[idx]
        base = _load_rgb(frame.image_path, self.max_side)
        if base is None:
            return None
        decoded = self._decode_depth(base, dpath, frame.camera_seq)
        self._cache[idx] = decoded
        if len(self._cache) > _MAX_CACHE:
            self._cache.popitem(last=False)
        return decoded

    @staticmethod
    def _decode_depth(base: np.ndarray, dpath: str, seq: int) -> dict:
        with np.load(dpath) as npz:
            depth = np.asarray(npz["depth_m"], dtype=np.float32)
            fit_status = int(npz["fit_status"]) if "fit_status" in npz else -1
            n_inliers = int(npz["n_inliers"]) if "n_inliers" in npz else -1
            rmse = (
                float(npz["rmse_inliers_m"])
                if "rmse_inliers_m" in npz
                else float("nan")
            )
        # Depth is produced at the source image resolution; re-grid to the
        # (possibly downscaled) display base before normalising.
        bh, bw = base.shape[:2]
        if depth.shape[:2] != (bh, bw):
            depth = _cap_side_to(depth, bw, bh)
        norm, valid, vmin, vmax = _depth_to_norm(depth)
        return {
            "base": base,
            "norm": norm,
            "valid": valid,
            "meta": {
                "seq": seq,
                "vmin": vmin,
                "vmax": vmax,
                "fit_status": fit_status,
                "n_inliers": n_inliers,
                "rmse": rmse,
                "valid_frac": float(valid.mean()),
            },
        }


def _cap_side_to(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize a depth map to an exact (w, h), nearest-neighbour (crisp edges)."""
    try:
        import cv2

        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception:  # noqa: BLE001
        from PIL import Image as PILImage

        return np.asarray(
            PILImage.fromarray(arr).resize((w, h), PILImage.NEAREST),
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# The viewer.
# ---------------------------------------------------------------------------
class DepthViewer:
    """Interactive matplotlib window for one chunk's depth_2d artifacts."""

    def __init__(
        self,
        bag_id: str,
        chunk_id: str,
        cameras: list[str],
        frames_by_cam: dict[str, list[CameraFrameInfo]],
        initial_cam: str,
        max_side: int = 1280,
    ) -> None:
        self.bag_id = bag_id
        self.chunk_id = chunk_id
        self.cameras = cameras
        self.frames_by_cam = frames_by_cam
        self.max_side = max_side

        self.cam_idx = cameras.index(initial_cam)
        self.idx = 0
        self.opacity = 0.6
        self.split = False
        self.split_pos = 0.5  # fraction of width
        self._dragging = False
        self._cmap_idx = 0

        self.store = self._make_store(initial_cam)

    def _make_store(self, cam_id: str) -> _FrameStore:
        entries = _entries_for_camera(
            self.bag_id, self.chunk_id, cam_id, self.frames_by_cam[cam_id]
        )
        return _FrameStore(entries, self.max_side)

    # -- lifecycle ----------------------------------------------------------
    def show(self) -> None:
        plt, cm, Normalize, Slider, Button = _mpl()
        self._plt = plt
        self._cmap = plt.get_cmap(_COLORMAPS[self._cmap_idx])

        fig = plt.figure(figsize=(13.0, 8.6))
        fig.patch.set_facecolor(_BG)
        mgr = getattr(fig.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title(
                f"perception_2d depth — {self.bag_id} / chunk {self.chunk_id}"
            )
        self.fig = fig

        self.ax_img = fig.add_axes([0.035, 0.20, 0.84, 0.72])
        self.ax_img.set_facecolor(_BG)
        self.ax_img.axis("off")

        first = self.store.get(0)
        h, w = first["base"].shape[:2] if first else (720, 1280)
        blank = first["base"] if first else np.zeros((h, w, 3), np.float32)
        self._im = self.ax_img.imshow(blank, interpolation="nearest", animated=True)
        self._divider = self.ax_img.axvline(
            w * 0.5, color="white", lw=1.6, alpha=0.9, visible=False
        )

        # Colourbar (true metres) sharing the live colormap + per-frame range.
        cax = fig.add_axes([0.885, 0.20, 0.018, 0.72])
        self._norm = Normalize(vmin=0.0, vmax=1.0)
        self._sm = cm.ScalarMappable(norm=self._norm, cmap=self._cmap)
        self._cbar = fig.colorbar(self._sm, cax=cax)
        self._cbar.set_label("depth (m)", color=_FG)
        cax.tick_params(colors=_FG, labelsize=8)
        cax.yaxis.label.set_color(_FG)

        self.title = fig.text(
            0.035, 0.945, "", color=_FG, fontsize=12.5, fontweight="bold", ha="left"
        )
        fig.text(
            0.5,
            0.012,
            "←/→ frame   ↑/↓ opacity   space split   c colormap   q quit"
            "    ·    drag the image in SPLIT mode to move the divide",
            color="#7a7a88",
            fontsize=8.5,
            ha="center",
        )

        self._build_widgets(Slider, Button)
        self._connect_keys()
        self._render()

        log.info(
            "depth viewer open: %s / chunk %s / cam %s — close window to continue",
            self.bag_id,
            self.chunk_id,
            self.cameras[self.cam_idx],
        )
        plt.show()

    # -- widgets ------------------------------------------------------------
    def _build_widgets(self, Slider, Button) -> None:
        fig = self.fig
        n = len(self.store.entries)

        ax_frame = fig.add_axes([0.235, 0.115, 0.52, 0.035], facecolor=_PANEL)
        self.s_frame = Slider(
            ax_frame, "frame", 0, max(n - 1, 1), valinit=0, valstep=1, color=_ACCENT
        )
        ax_op = fig.add_axes([0.235, 0.060, 0.52, 0.035], facecolor=_PANEL)
        self.s_op = Slider(
            ax_op, "opacity", 0.0, 1.0, valinit=self.opacity, color=_ACCENT
        )
        for s in (self.s_frame, self.s_op):
            s.label.set_color(_FG)
            s.valtext.set_color(_FG)
        self.s_frame.on_changed(self._on_frame)
        self.s_op.on_changed(self._on_opacity)

        self.b_mode = self._button(Button, [0.035, 0.095, 0.12, 0.052], "SPLIT: off")
        self.b_cmap = self._button(Button, [0.035, 0.035, 0.12, 0.052], "turbo_r")
        self.b_prev = self._button(Button, [0.795, 0.115, 0.075, 0.05], "◄ cam")
        self.b_next = self._button(Button, [0.875, 0.115, 0.075, 0.05], "cam ►")
        self.b_mode.on_clicked(lambda _e: self._toggle_split())
        self.b_cmap.on_clicked(lambda _e: self._cycle_cmap())
        self.b_prev.on_clicked(lambda _e: self._switch_camera(-1))
        self.b_next.on_clicked(lambda _e: self._switch_camera(+1))

    def _button(self, Button, rect, label):
        ax = self.fig.add_axes(rect)
        b = Button(ax, label, color=_PANEL, hovercolor="#2c2c36")
        b.label.set_color(_FG)
        b.label.set_fontsize(9.5)
        return b

    def _connect_keys(self) -> None:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    # -- callbacks ----------------------------------------------------------
    def _on_frame(self, val) -> None:
        self.idx = int(round(val))
        self._render()

    def _on_opacity(self, val) -> None:
        self.opacity = float(val)
        if not self.split:
            self._render()

    def _on_key(self, event) -> None:
        key = event.key
        if key in ("right", "left"):
            self._step_frame(1 if key == "right" else -1)
        elif key in ("up", "down"):
            self.s_op.set_val(
                np.clip(self.opacity + (0.05 if key == "up" else -0.05), 0, 1)
            )
        elif key == " ":
            self._toggle_split()
        elif key == "c":
            self._cycle_cmap()
        elif key in ("q", "escape"):
            self._plt.close(self.fig)

    def _on_press(self, event) -> None:
        if self.split and event.inaxes is self.ax_img and event.xdata is not None:
            self._dragging = True
            self._set_split_from_event(event)

    def _on_motion(self, event) -> None:
        if self._dragging and event.inaxes is self.ax_img and event.xdata is not None:
            self._set_split_from_event(event)

    def _on_release(self, _event) -> None:
        self._dragging = False

    # -- actions ------------------------------------------------------------
    def _step_frame(self, delta: int) -> None:
        n = len(self.store.entries)
        self.s_frame.set_val(int(np.clip(self.idx + delta, 0, n - 1)))

    def _toggle_split(self) -> None:
        self.split = not self.split
        self.b_mode.label.set_text(f"SPLIT: {'on' if self.split else 'off'}")
        self._render()

    def _cycle_cmap(self) -> None:
        self._cmap_idx = (self._cmap_idx + 1) % len(_COLORMAPS)
        name = _COLORMAPS[self._cmap_idx]
        self._cmap = self._plt.get_cmap(name)
        self._sm.set_cmap(self._cmap)
        self.b_cmap.label.set_text(name)
        self._render()

    def _switch_camera(self, delta: int) -> None:
        if len(self.cameras) < 2:
            return
        self.cam_idx = (self.cam_idx + delta) % len(self.cameras)
        self.store = self._make_store(self.cameras[self.cam_idx])
        self.idx = 0
        n = len(self.store.entries)
        self.s_frame.valmax = max(n - 1, 1)
        self.s_frame.ax.set_xlim(self.s_frame.valmin, self.s_frame.valmax)
        self.s_frame.set_val(0)  # fires _on_frame → _render
        self._render()

    def _set_split_from_event(self, event) -> None:
        w = self._im.get_array().shape[1]
        self.split_pos = float(np.clip(event.xdata / max(w - 1, 1), 0.0, 1.0))
        self._render()

    # -- rendering ----------------------------------------------------------
    def _render(self) -> None:
        n = len(self.store.entries)
        if n == 0:
            self.title.set_text(
                f"cam {self.cameras[self.cam_idx]} — no depth artifacts in this chunk"
            )
            self.fig.canvas.draw_idle()
            return
        self.idx = int(np.clip(self.idx, 0, n - 1))
        decoded = self.store.get(self.idx)
        if decoded is None:
            self.title.set_text("frame image could not be loaded")
            self.fig.canvas.draw_idle()
            return

        base, norm, valid = decoded["base"], decoded["norm"], decoded["valid"]
        m = decoded["meta"]

        rgb = self._cmap(norm)[..., :3].astype(np.float32)
        # Depth layer falls back to the photo where depth is invalid, so sky /
        # fit-holes show the image through at any opacity.
        depth_over = np.where(valid[..., None], rgb, base)

        if self.split:
            col = int(round(self.split_pos * (base.shape[1] - 1)))
            out = base.copy()
            out[:, col:] = depth_over[:, col:]
            self._divider.set_xdata([col, col])
            self._divider.set_visible(True)
        else:
            a = self.opacity
            out = base * (1.0 - a) + depth_over * a
            self._divider.set_visible(False)

        self._im.set_data(np.clip(out, 0.0, 1.0))

        # Colourbar tracks this frame's metric range.
        self._norm.vmin, self._norm.vmax = (m["vmin"], max(m["vmax"], m["vmin"] + 1e-6))
        self._sm.set_norm(self._norm)
        self._cbar.update_normal(self._sm)

        cam = self.cameras[self.cam_idx]
        fit = _FIT_LABELS.get(m["fit_status"], "?")
        rmse = m["rmse"]
        rmse_str = f"{rmse:.2f}" if np.isfinite(rmse) else "—"
        self.title.set_text(
            f"{cam}   frame {m['seq']:06d}  ({self.idx + 1}/{n})    "
            f"depth {m['vmin']:.1f}–{m['vmax']:.1f} m    "
            f"fit={fit}  inliers={m['n_inliers']}  rmse={rmse_str} m    "
            f"valid={m['valid_frac'] * 100:.0f}%"
        )
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# matplotlib import shim (mirrors viz._o3d in lidar_preprocessing).
# ---------------------------------------------------------------------------
def _mpl():
    try:
        import matplotlib

        backend = matplotlib.get_backend().lower()
        if "agg" in backend and "tkagg" not in backend and "qtagg" not in backend:
            log.warning(
                "matplotlib is using a non-interactive backend (%s); the window "
                "may not appear. Install python3-tk or run on a host with a "
                "display (DISPLAY / WSLg).",
                backend,
            )
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.widgets import Button, Slider

        return plt, cm, Normalize, Slider, Button
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "matplotlib is required for the depth viewer: pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Orchestrator (called by the CLI).
# ---------------------------------------------------------------------------
def viz_depth(
    bag_id: str,
    chunk_id: Optional[str] = None,
    cam_id: Optional[str] = None,
    max_side: int = 1280,
) -> None:
    """Open a depth viewer per chunk (one window at a time, blocks until closed).

    With ``chunk_id`` set, only that chunk is shown.  ``cam_id`` selects the
    starting camera (defaults to the first with depth; switchable in-window).
    """
    if chunk_id is not None:
        chunk_ids = [chunk_id]
    else:
        try:
            rows = read_rows(chunks_index_path(bag_id))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No chunks index for bag {bag_id!r} — run ingest first."
            ) from exc
        chunk_ids = [r["chunk_id"] for r in rows]

    opened = 0
    for cid in chunk_ids:
        cams, frames_by_cam = _cameras_with_depth(bag_id, cid)
        if not cams:
            log.warning(
                "chunk %s: no depth_2d artifacts — run `perception_2d run` with "
                "depth.enabled first. Skipping.",
                cid,
            )
            continue
        start_cam = cam_id if cam_id in cams else cams[0]
        if cam_id is not None and cam_id not in cams:
            log.warning(
                "chunk %s: camera %r has no depth; using %s instead (available: %s)",
                cid,
                cam_id,
                start_cam,
                ", ".join(cams),
            )
        DepthViewer(
            bag_id, cid, cams, frames_by_cam, initial_cam=start_cam, max_side=max_side
        ).show()
        opened += 1

    if opened == 0:
        log.warning(
            "nothing to show for bag %s — no chunk had depth_2d artifacts.", bag_id
        )
