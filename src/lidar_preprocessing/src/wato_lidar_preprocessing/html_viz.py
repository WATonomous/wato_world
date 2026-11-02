"""Self-contained browser viewer for lidar_preprocessing point clouds.

The default `viz` backend: a standalone HTML file with embedded WebGL and
downsampled point buffers, for static vs dynamic clouds with sweep scrubbing.
The page itself is wato_common.viz.point_cloud_html; this module builds its
payloads.

    python -m wato_lidar_preprocessing viz --bag BAG --chunk 0000 --open
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from wato_common.viz.point_cloud_html import (
    array_spec as _array_spec,
    empty_spec as _empty_spec,
    render_html as _render_html,
    sample_indices as _sample_indices,
)

log = logging.getLogger(__name__)

_MAX_STATIC_HTML_PTS = 350_000
_MAX_DYNAMIC_HTML_PTS = 650_000
_STATIC_RGB = np.array([74, 143, 217], dtype=np.uint8)
_DYNAMIC_RGB = np.array([232, 77, 61], dtype=np.uint8)
# Proposal-layer palette, lowest to highest priority (later paints over).
_PROPOSAL_ATTACH_RGB = np.array([140, 140, 150], dtype=np.uint8)  # attach-only
_PROPOSAL_SEED_RGB = np.array([240, 210, 60], dtype=np.uint8)  # seed source
_PROPOSAL_IWU_RGB = np.array([200, 80, 200], dtype=np.uint8)  # IWU_EVICTED
_PROPOSAL_FILL_RGB = np.array([245, 160, 40], dtype=np.uint8)  # BOX_FILL
_PROPOSAL_MOVING_RGB = _DYNAMIC_RGB  # member of a cluster with motion_score > 1


def _default_output_path(bag_id: str, chunk_id: str, sweep_id: int | None) -> Path:
    from wato_common.artifact_store import chunk_root, local_path

    stem = f"sweep_{sweep_id:06d}" if sweep_id is not None else "chunk"
    return Path(local_path(chunk_root(bag_id, chunk_id))) / "viz" / f"{stem}.html"


def _load_chunk_payload(bag_id: str, chunk_id: str) -> dict:
    from wato_lidar_preprocessing.viz_data import load_chunk_viz_data

    data = load_chunk_viz_data(bag_id, chunk_id)
    static_xyz = data.static_xyz
    dynamic_xyz = data.dynamic_xyz
    sweep_id = data.dynamic_sweep_id
    intensity = data.dynamic_intensity

    n_static_total = len(static_xyz)
    n_dynamic_total = len(dynamic_xyz)
    static_idx = _sample_indices(n_static_total, _MAX_STATIC_HTML_PTS, seed=17)
    dynamic_idx = _sample_indices(n_dynamic_total, _MAX_DYNAMIC_HTML_PTS, seed=23)

    static_xyz = static_xyz[static_idx].astype(np.float32, copy=False)
    dynamic_xyz = dynamic_xyz[dynamic_idx].astype(np.float32, copy=False)
    sweep_id = sweep_id[dynamic_idx].astype(np.int32, copy=False)
    if intensity is not None:
        intensity = intensity[dynamic_idx].astype(np.float32, copy=False)

    if len(sweep_id) > 0:
        order = np.argsort(sweep_id, kind="stable")
        dynamic_xyz = dynamic_xyz[order]
        sweep_id = sweep_id[order]
        if intensity is not None:
            intensity = intensity[order]

    xyz_for_bounds = [a for a in (static_xyz, dynamic_xyz) if len(a) > 0]
    bounds_src = np.vstack(xyz_for_bounds) if xyz_for_bounds else np.zeros((1, 3))

    return {
        "title": f"lidar_preprocessing chunk {chunk_id}",
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "mode": "chunk",
        "static_xyz": _array_spec(static_xyz, np.float32),
        "dynamic_xyz": _array_spec(dynamic_xyz, np.float32),
        "dynamic_sweep_id": _array_spec(sweep_id, np.int32),
        "dynamic_intensity": (
            _array_spec(intensity, np.float32)
            if intensity is not None
            else _empty_spec((0,), np.float32)
        ),
        "has_intensity": intensity is not None,
        "bounds_min": bounds_src.min(axis=0).astype(float).tolist(),
        "bounds_max": bounds_src.max(axis=0).astype(float).tolist(),
        "counts": {
            "static": int(len(static_xyz)),
            "dynamic": int(len(dynamic_xyz)),
            "static_sampled_from": int(n_static_total),
            "dynamic_sampled_from": int(n_dynamic_total),
        },
    }


def _load_sweep_payload(bag_id: str, chunk_id: str, sweep_id: int) -> dict:
    from wato_lidar_preprocessing.viz_data import load_sweep_viz_data

    data = load_sweep_viz_data(bag_id, chunk_id, sweep_id)
    xyz = data.xyz.astype(np.float32)
    dynamic = data.dynamic

    idx = _sample_indices(len(xyz), _MAX_DYNAMIC_HTML_PTS, seed=31)
    xyz = xyz[idx]
    dynamic = dynamic[idx]
    sweep_ids = np.full((len(xyz),), sweep_id, dtype=np.int32)
    colors = np.repeat(_STATIC_RGB[None, :], len(xyz), axis=0)
    colors[dynamic] = _DYNAMIC_RGB

    return {
        "title": f"lidar_preprocessing chunk {chunk_id} sweep {sweep_id}",
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "sweep_id": sweep_id,
        "mode": "sweep",
        "static_xyz": _empty_spec((0, 3), np.float32),
        "dynamic_xyz": _array_spec(xyz, np.float32),
        "dynamic_sweep_id": _array_spec(sweep_ids, np.int32),
        "dynamic_intensity": _empty_spec((0,), np.float32),
        "dynamic_rgb": _array_spec(colors, np.uint8),
        "has_intensity": False,
        "bounds_min": xyz.min(axis=0).astype(float).tolist() if len(xyz) else [0, 0, 0],
        "bounds_max": xyz.max(axis=0).astype(float).tolist() if len(xyz) else [1, 1, 1],
        "counts": {
            "static": int((~dynamic).sum()),
            "dynamic": int(dynamic.sum()),
            "sampled_from": int(len(idx)),
        },
    }


def _load_proposals_payload(bag_id: str, chunk_id: str) -> dict:
    """Chunk payload whose "dynamic" cloud is Step F's proposal points.

    Colours (the viewer's static/dynamic colour mode): grey = attach-only
    (AMBIGUOUS/UNMAPPED), yellow = a seed source, magenta = IWU_EVICTED,
    orange = BOX_FILL, red = member of a cluster passing Chen's criterion.
    """
    from wato_common.artifacts.motion_proposals import (
        BOX_FILL,
        IWU_EVICTED,
        SEED_BITS,
    )
    from wato_lidar_preprocessing.viz_data import load_chunk_proposal_viz_data

    data = load_chunk_proposal_viz_data(bag_id, chunk_id)
    static_idx = _sample_indices(len(data.static_xyz), _MAX_STATIC_HTML_PTS, seed=17)
    idx = _sample_indices(len(data.xyz), _MAX_DYNAMIC_HTML_PTS, seed=29)
    xyz = data.xyz[idx].astype(np.float32)
    sweep_id = data.sweep_id[idx]
    bits = data.source_bits[idx]
    moving = data.moving[idx]
    order = np.argsort(sweep_id, kind="stable")
    xyz, sweep_id, bits, moving = (
        xyz[order],
        sweep_id[order],
        bits[order],
        moving[order],
    )

    colors = np.repeat(_PROPOSAL_ATTACH_RGB[None, :], len(xyz), axis=0)
    colors[(bits & SEED_BITS) != 0] = _PROPOSAL_SEED_RGB
    colors[(bits & IWU_EVICTED) != 0] = _PROPOSAL_IWU_RGB
    colors[(bits & BOX_FILL) != 0] = _PROPOSAL_FILL_RGB
    colors[moving] = _PROPOSAL_MOVING_RGB

    static_xyz = data.static_xyz[static_idx].astype(np.float32)
    parts = [a for a in (static_xyz, xyz) if len(a)]
    bounds = np.vstack(parts) if parts else np.zeros((1, 3))
    return {
        "title": f"lidar_preprocessing proposals chunk {chunk_id}",
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "mode": "chunk",
        "static_xyz": _array_spec(static_xyz, np.float32),
        "dynamic_xyz": _array_spec(xyz, np.float32),
        "dynamic_sweep_id": _array_spec(sweep_id.astype(np.int32), np.int32),
        "dynamic_intensity": _empty_spec((0,), np.float32),
        "dynamic_rgb": _array_spec(colors, np.uint8),
        "has_intensity": False,
        "bounds_min": bounds.min(axis=0).astype(float).tolist(),
        "bounds_max": bounds.max(axis=0).astype(float).tolist(),
        "counts": {
            "static": int(len(static_xyz)),
            "dynamic": int(len(xyz)),
            "dynamic_sampled_from": int(len(data.xyz)),
            "moving": int(moving.sum()),
        },
    }


def write_html_viewer(
    bag_id: str,
    chunk_id: str,
    *,
    sweep_id: int | None = None,
    out_path: str | Path | None = None,
    layer: str = "classification",
) -> Path:
    """Write a standalone HTML/WebGL point-cloud viewer and return its path.

    layer="proposals" shows Step F's motion proposals (chunk-level only).
    """
    if layer == "proposals":
        if sweep_id is not None:
            raise ValueError("the proposals layer is chunk-level; omit sweep_id")
        payload = _load_proposals_payload(bag_id, chunk_id)
    else:
        payload = (
            _load_sweep_payload(bag_id, chunk_id, sweep_id)
            if sweep_id is not None
            else _load_chunk_payload(bag_id, chunk_id)
        )
    out = (
        Path(out_path)
        if out_path is not None
        else _default_output_path(bag_id, chunk_id, sweep_id)
    )
    if out.suffix.lower() != ".html":
        out.mkdir(parents=True, exist_ok=True)
        name = (
            "proposals.html"
            if layer == "proposals"
            else f"sweep_{sweep_id:06d}.html"
            if sweep_id is not None
            else "chunk.html"
        )
        out = out / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_html(payload), encoding="utf-8")
    return out
