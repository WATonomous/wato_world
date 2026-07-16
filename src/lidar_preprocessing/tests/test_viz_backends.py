from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from wato_common.artifact_store import (
    dynamic_map_path,
    dynamic_mask_path,
    lidar_world_path,
    local_path,
    static_map_path,
    voxel_diag_path,
)
from wato_lidar_preprocessing.viz_data import (
    load_chunk_viz_data,
    load_sweep_viz_data,
)
from wato_lidar_preprocessing.viz_export import export_ply
from wato_lidar_preprocessing.html_viz import open_html_file


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def test_load_chunk_viz_data_reads_optional_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    bag_id, chunk_id = "bag_viz", "0000"

    static_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    dynamic_xyz = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    static_path = local_path(static_map_path(bag_id, chunk_id))
    dynamic_path = local_path(dynamic_map_path(bag_id, chunk_id))
    _ensure_parent(static_path)
    np.savez_compressed(static_path, xyz=static_xyz, intensity=np.array([0.2]))
    np.savez_compressed(
        dynamic_path,
        xyz=dynamic_xyz,
        sweep_id=np.array([7], dtype=np.int32),
        intensity=np.array([0.9], dtype=np.float32),
    )

    diag_path = local_path(voxel_diag_path(bag_id, chunk_id))
    np.savez_compressed(
        diag_path,
        keys=np.array([1 << 40], dtype=np.int64),
        coords=np.array([[1, 0, 0]], dtype=np.int32),
        origin=np.zeros(3, dtype=np.float64),
        voxel_size=np.float32(1.0),
        log_odds=np.array([2.0], dtype=np.float32),
        p_occ=np.array([0.88], dtype=np.float32),
        n_obs=np.array([4], dtype=np.int32),
        n_hits=np.array([3], dtype=np.int32),
        classification=np.array([4], dtype=np.int8),
    )

    data = load_chunk_viz_data(bag_id, chunk_id)

    np.testing.assert_allclose(data.static_xyz, static_xyz)
    np.testing.assert_allclose(data.dynamic_xyz, dynamic_xyz)
    np.testing.assert_array_equal(data.dynamic_sweep_id, [7])
    np.testing.assert_allclose(data.dynamic_p_occ, [0.88])
    np.testing.assert_array_equal(data.dynamic_n_obs, [4])
    np.testing.assert_array_equal(data.dynamic_n_hits, [3])
    np.testing.assert_array_equal(data.dynamic_classification, [4])


def test_sweep_viz_data_and_ply_export(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    bag_id, chunk_id = "bag_viz_sweep", "0000"
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)

    world_path = local_path(lidar_world_path(bag_id, chunk_id, 3))
    _ensure_parent(world_path)
    np.savez_compressed(
        world_path,
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        intensity=np.array([0.1, 0.2], dtype=np.float32),
        ground_mask=np.array([True, False]),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    np.save(local_path(dynamic_mask_path(bag_id, chunk_id, 3)), np.array([False, True]))

    data = load_sweep_viz_data(bag_id, chunk_id, 3)
    np.testing.assert_allclose(data.xyz, xyz)
    np.testing.assert_array_equal(data.dynamic, [False, True])
    np.testing.assert_array_equal(data.ground_mask, [True, False])

    out = export_ply(bag_id, chunk_id, sweep_id=3, out_path=tmp_path / "sweep.ply")
    assert out == Path(tmp_path / "sweep.ply")
    content = out.read_bytes()
    assert b"format binary_little_endian 1.0" in content
    assert b"element vertex 2" in content
    assert b"property uchar dynamic" in content


def test_open_html_file_resolves_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    html = Path("viz") / "chunk.html"
    html.parent.mkdir()
    html.write_text("<html></html>", encoding="utf-8")

    opened = []
    monkeypatch.setattr(
        "webbrowser.open", lambda url, new=0: opened.append((url, new)) or True
    )

    assert open_html_file(html) is True
    assert opened == [(html.resolve().as_uri(), 2)]
