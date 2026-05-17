"""Tests for the URI-composition logic in wato_common.artifact_store."""

from __future__ import annotations

import os

import pytest

from wato_common import artifact_store


@pytest.fixture(autouse=True)
def fixed_root(monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", "file:///data/artifacts")
    yield


def test_paths_match_arch_doc_layout():
    assert artifact_store.bag_root("b") == "file:///data/artifacts/raw/b"
    assert (
        artifact_store.chunk_root("b", "0000")
        == "file:///data/artifacts/raw/b/chunks/0000"
    )
    assert (
        artifact_store.calibration_path("b")
        == "file:///data/artifacts/raw/b/calibration.json"
    )
    assert (
        artifact_store.bag_meta_path("b")
        == "file:///data/artifacts/raw/b/bag_meta.json"
    )
    assert (
        artifact_store.chunks_index_path("b")
        == "file:///data/artifacts/raw/b/chunks/index.parquet"
    )


def test_per_chunk_paths():
    assert (
        artifact_store.frame_index_path("b", "0000")
        == "file:///data/artifacts/raw/b/chunks/0000/frame_index.parquet"
    )
    assert (
        artifact_store.poses_path("b", "0000")
        == "file:///data/artifacts/raw/b/chunks/0000/poses.parquet"
    )
    assert (
        artifact_store.lidar_sweeps_path("b", "0000")
        == "file:///data/artifacts/raw/b/chunks/0000/lidar_sweeps.parquet"
    )
    assert (
        artifact_store.camera_frames_path("b", "0000")
        == "file:///data/artifacts/raw/b/chunks/0000/camera_frames.parquet"
    )


def test_camera_image_path_is_zero_padded():
    assert (
        artifact_store.camera_image_path("b", "0000", "CAM_FRONT", 7, "jpg")
        == "file:///data/artifacts/raw/b/chunks/0000/cam_CAM_FRONT/000007.jpg"
    )


def test_lidar_sweep_path_is_zero_padded():
    assert (
        artifact_store.lidar_sweep_path("b", "0000", "LIDAR_TOP", 42)
        == "file:///data/artifacts/raw/b/chunks/0000/lidar/LIDAR_TOP/000042.npz"
    )


def test_local_path_resolves_file_uri():
    assert artifact_store.local_path("file:///foo/bar") == "/foo/bar"


def test_local_path_rejects_remote_uri():
    with pytest.raises(ValueError):
        artifact_store.local_path("s3://bucket/key")


def test_ensure_local_dir_creates(tmp_path):
    uri = f"file://{tmp_path}/sub/dir"
    artifact_store.ensure_local_dir(uri)
    assert os.path.isdir(tmp_path / "sub" / "dir")
