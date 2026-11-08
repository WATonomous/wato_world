"""wato_common.artifacts.ground: what lidar_preprocessing Step C writes,
consumers read back."""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import ground_path, local_path
from wato_common.artifacts import ArtifactFormatError, ground

BAG, CHUNK = "b", "0000"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _grid(h: int = 3, w: int = 4) -> ground.GroundGrid:
    """Height = 10 * row + col, so a lookup's cell is readable from its value;
    cell (1, 2), when there is one, tilts."""
    height = (10.0 * np.arange(h)[:, None] + np.arange(w)[None, :]).astype(np.float32)
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = 1.0
    if h > 1 and w > 2:
        normals[1, 2] = [0.6, 0.0, 0.8]
    return ground.GroundGrid(height, normals, np.array([100.0, 200.0]), 0.5)


def test_round_trip():
    grid = _grid()
    pts = np.array([[100.1, 200.2, 0.0], [101.0, 200.9, 0.3]])
    ground.write(BAG, CHUNK, grid, pts)
    got = ground.read(BAG, CHUNK)
    np.testing.assert_array_equal(got.height_grid, grid.height_grid)
    np.testing.assert_array_equal(got.normal_grid, grid.normal_grid)
    np.testing.assert_array_equal(got.grid_origin, grid.grid_origin)
    assert got.cell_size == 0.5
    assert got.height_grid.dtype == np.float32 and got.grid_origin.dtype == np.float64
    np.testing.assert_array_equal(ground.read_points(BAG, CHUNK), pts)


def test_lookups_use_row_y_col_x_and_clamp():
    grid = _grid()
    # (x, y) = origin + (col + 0.5, row + 0.5) * cell
    xy = np.array(
        [
            [100.25, 200.25],  # row 0, col 0
            [101.25, 200.75],  # row 1, col 2
            [99.0, 199.0],  # below the grid -> row 0, col 0
            [500.0, 500.0],  # above the grid -> row 2, col 3
        ]
    )
    np.testing.assert_array_equal(grid.height_at(xy), [0.0, 12.0, 0.0, 23.0])
    np.testing.assert_allclose(
        grid.height_above(np.column_stack([xy, [1.0, 13.0, 0.5, 23.0]])),
        [1.0, 1.0, 0.5, 0.0],
    )
    np.testing.assert_allclose(grid.normal_at(xy[1:2]), [[0.6, 0.0, 0.8]])


@pytest.mark.parametrize("status", ["empty", "skipped_no_ground_mask"])
def test_sentinel_raises_unavailable_but_keeps_points(status):
    ground.write(BAG, CHUNK, _grid(1, 1), np.empty((0, 3)), status=status)
    with pytest.raises(ground.GroundUnavailableError) as exc:
        ground.read(BAG, CHUNK)
    assert exc.value.status == status
    assert ground.read_points(BAG, CHUNK).shape == (0, 3)


def test_missing_file_is_file_not_found():
    with pytest.raises(FileNotFoundError):
        ground.read(BAG, CHUNK)


def test_writer_rejects_inconsistent_grids():
    grid = _grid()
    bad_normals = ground.GroundGrid(
        grid.height_grid, grid.normal_grid[:, :2], grid.grid_origin, 0.5
    )
    with pytest.raises(ValueError, match="normal_grid"):
        ground.write(BAG, CHUNK, bad_normals, np.empty((0, 3)))
    with pytest.raises(ValueError, match="status"):
        ground.write(BAG, CHUNK, grid, np.empty((0, 3)), status="done")
    with pytest.raises(ValueError, match="ground_xyz"):
        ground.write(BAG, CHUNK, grid, np.empty((0, 2)))


def test_reader_rejects_files_without_every_array():
    """A ground.npz without status or normal_grid (an older writer, or one not
    going through this module) is rejected, not read as status "ok"."""
    path = local_path(ground_path(BAG, CHUNK))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g = _grid()
    np.savez_compressed(
        path,
        height_grid=g.height_grid,
        grid_origin=g.grid_origin,
        cell_size=np.float32(0.5),
        ground_xyz=np.empty((0, 3)),
    )
    with pytest.raises(ArtifactFormatError, match="normal_grid"):
        ground.read(BAG, CHUNK)
    with pytest.raises(ArtifactFormatError, match="status"):
        ground.read_points(BAG, CHUNK)


def test_reader_rejects_non_finite_heights():
    g = _grid()
    height = g.height_grid.copy()
    height[0, 0] = np.nan
    path = local_path(ground_path(BAG, CHUNK))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        height_grid=height,
        normal_grid=g.normal_grid,
        grid_origin=g.grid_origin,
        cell_size=np.float32(0.5),
        ground_xyz=np.empty((0, 3)),
        status=np.array("ok"),
    )
    with pytest.raises(ArtifactFormatError, match="non-finite"):
        ground.read(BAG, CHUNK)
