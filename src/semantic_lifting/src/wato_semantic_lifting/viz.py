"""Browser viewer for a chunk's lifted labels.

Writes <chunk>/viz/lifted_labels.html (wato_common.viz.point_cloud_html):
labeled points coloured by instance (or class) over grey unlabeled context,
world frame, with a slider over the chunk's sweeps in time order.

    python -m wato_semantic_lifting viz --bag BAG --chunk 0000 --open
"""

from __future__ import annotations

import colorsys
import zlib
from pathlib import Path

import numpy as np

from wato_common.artifact_store import chunk_root, local_path
from wato_common.artifacts import lifted_labels
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.viz.point_cloud_html import (
    array_spec,
    empty_spec,
    render_html,
    sample_indices,
)

_MAX_LABELED_PTS = 650_000
_MAX_CONTEXT_PTS = 350_000
# Unlabeled context is drawn from every k-th sweep: enough to show the scene.
_CONTEXT_EVERY_K_SWEEPS = 20
_CONTEXT_COLOR = [0.36, 0.38, 0.42]
_CLASS_RGB = {
    "car": (66, 135, 245),
    "truck": (245, 160, 40),
    "bus": (200, 80, 200),
    "pedestrian": (232, 77, 61),
    "bicycle": (80, 210, 120),
    "motorcycle": (240, 210, 60),
}


def _hashed_rgb(name: str) -> tuple[int, int, int]:
    """A stable, well-spread colour per name (instance ids are UUID prefixes)."""
    hue = (zlib.crc32(name.encode()) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _colors(names: np.ndarray, palette: dict | None) -> np.ndarray:
    uniq, inv = np.unique(names, return_inverse=True)
    table = np.array(
        [(palette or {}).get(n) or _hashed_rgb(n) for n in uniq.tolist()],
        dtype=np.uint8,
    ).reshape(-1, 3)
    return table[inv.reshape(-1)]


def _load_payload(bag_id: str, chunk_id: str, color_by: str) -> dict:
    if not lifted_labels.exists(bag_id, chunk_id):
        raise FileNotFoundError(
            f"no lifted labels for {bag_id}/{chunk_id} — run semantic_lifting first"
        )
    sweeps = ProcessedSweeps.load(bag_id, chunk_id)
    rows = sorted(sweeps.rows, key=lambda r: (r.reference_timestamp_ns, r.sweep_id))
    xyz, step, names, context = [], [], [], []
    for k, row in enumerate(rows):
        world = sweeps.world(row.sweep_id)
        labels = lifted_labels.read_sweep(
            bag_id, chunk_id, row.sweep_id, n_points=world.n_points
        )
        pts = world.xyz
        xyz.append(pts[labels.point_indices])
        step.append(np.full(labels.point_indices.size, k, np.int32))
        names.append(labels.instance_ids if color_by == "instance" else labels.classes)
        if k % _CONTEXT_EVERY_K_SWEEPS == 0:
            unlabeled = np.ones(world.n_points, bool)
            unlabeled[labels.point_indices] = False
            context.append(pts[unlabeled])

    xyz = np.concatenate(xyz) if xyz else np.zeros((0, 3))
    step = np.concatenate(step) if step else np.zeros(0, np.int32)
    names = np.concatenate(names) if names else np.zeros(0, np.str_)
    context = np.concatenate(context) if context else np.zeros((0, 3))
    n_labeled, n_context = len(xyz), len(context)
    idx = sample_indices(n_labeled, _MAX_LABELED_PTS, seed=23)
    cidx = sample_indices(n_context, _MAX_CONTEXT_PTS, seed=17)
    xyz, step, names = xyz[idx].astype(np.float32), step[idx], names[idx]
    context = context[cidx].astype(np.float32)
    rgb = _colors(names, _CLASS_RGB if color_by == "class" else None)

    parts = [a for a in (context, xyz) if len(a)]
    bounds = np.vstack(parts) if parts else np.zeros((1, 3))
    return {
        "title": f"semantic_lifting chunk {chunk_id} — by {color_by}",
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "mode": "chunk",
        "static_xyz": array_spec(context, np.float32),
        "dynamic_xyz": array_spec(xyz, np.float32),
        # The slider steps through sweeps in time order: sweep_ids are one
        # counter per chunk across LiDARs, not time-ordered.
        "dynamic_sweep_id": array_spec(step, np.int32),
        "dynamic_intensity": empty_spec((0,), np.float32),
        "dynamic_rgb": array_spec(rgb, np.uint8),
        "has_intensity": False,
        "bounds_min": bounds.min(axis=0).astype(float).tolist(),
        "bounds_max": bounds.max(axis=0).astype(float).tolist(),
        "static_label": "unlabeled",
        "dynamic_label": "labeled",
        "uniform_label": color_by,
        "all_label": "all sweeps",
        "static_color": _CONTEXT_COLOR,
        "counts": {
            "labeled": int(len(xyz)),
            "labeled_sampled_from": int(n_labeled),
            "context": int(len(context)),
            "context_sampled_from": int(n_context),
            "distinct": int(len(np.unique(names))),
        },
    }


def write_html_viewer(
    bag_id: str,
    chunk_id: str,
    *,
    color_by: str = "instance",
    out_path: str | Path | None = None,
) -> Path:
    """Write the chunk's lifted-label viewer and return its path."""
    if color_by not in ("instance", "class"):
        raise ValueError(f"color_by must be 'instance' or 'class', got {color_by!r}")
    payload = _load_payload(bag_id, chunk_id, color_by)
    out = (
        Path(out_path)
        if out_path is not None
        else Path(local_path(chunk_root(bag_id, chunk_id))) / "viz"
    )
    if out.suffix.lower() != ".html":
        out = out / f"lifted_labels_{color_by}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    return out
