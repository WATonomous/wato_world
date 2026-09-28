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


def test_lidar_processed_sweep_paths_are_versioned_and_sensor_scoped(
    tmp_path, monkeypatch
):
    versions = tmp_path / "component_versions.yaml"
    versions.write_text("lidar_preprocessing: v2\n", encoding="utf-8")
    monkeypatch.setenv("COMPONENT_VERSIONS_PATH", str(versions))

    left = artifact_store.lidar_world_path("b", "0000", "LIDAR_LEFT", 42)
    right = artifact_store.lidar_world_path("b", "0000", "LIDAR_RIGHT", 42)

    assert left == (
        "file:///data/artifacts/lidar_preprocessing/v2/b/0000/"
        "sweeps/LIDAR_LEFT/000042_world.npz"
    )
    assert right != left
    assert artifact_store.static_mask_path("b", "0000", "LIDAR_LEFT", 42).endswith(
        "/sweeps/LIDAR_LEFT/000042_static_mask.npy"
    )


def test_lidar_publication_metadata_lives_in_v2_tree(tmp_path, monkeypatch):
    versions = tmp_path / "component_versions.yaml"
    versions.write_text("lidar_preprocessing: v2\n", encoding="utf-8")
    monkeypatch.setenv("COMPONENT_VERSIONS_PATH", str(versions))

    assert artifact_store.lidar_completion_path("b", "0000").endswith(
        "/lidar_preprocessing/v2/b/0000/completion.json"
    )
    assert artifact_store.lidar_bag_manifest_path("b").endswith(
        "/lidar_preprocessing/v2/b/manifest.json"
    )


def test_lidar_staging_redirect_only_changes_local_v2_artifacts(
    tmp_path, monkeypatch
):
    versions = tmp_path / "component_versions.yaml"
    versions.write_text("lidar_preprocessing: v2\n", encoding="utf-8")
    monkeypatch.setenv("COMPONENT_VERSIONS_PATH", str(versions))
    monkeypatch.setenv("WATO_LIDAR_STAGING_ROOT", str(tmp_path / "stage"))

    staged = artifact_store.local_path(
        artifact_store.ground_path("b", "0000")
    )
    raw = artifact_store.local_path(artifact_store.poses_path("b", "0000"))

    assert staged == str(tmp_path / "stage" / "b" / "0000" / "ground.npz")
    assert raw == "/data/artifacts/raw/b/chunks/0000/poses.parquet"


@pytest.mark.parametrize("lidar_id", ["../escape", "lidar/left", "", "left lidar"])
def test_processed_sweep_paths_reject_unsafe_lidar_ids(lidar_id):
    with pytest.raises(ValueError, match="lidar_id"):
        artifact_store.lidar_world_path("b", "0000", lidar_id, 0)


def test_local_path_resolves_file_uri():
    assert artifact_store.local_path("file:///foo/bar") == "/foo/bar"


def test_local_path_rejects_remote_uri():
    with pytest.raises(ValueError):
        artifact_store.local_path("s3://bucket/key")


def test_ensure_local_dir_creates(tmp_path):
    uri = f"file://{tmp_path}/sub/dir"
    artifact_store.ensure_local_dir(uri)
    assert os.path.isdir(tmp_path / "sub" / "dir")
