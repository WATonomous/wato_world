"""Tests for provenance capture and manifest writing.

These guard the two properties the reproducibility story depends on:
  - provenance never raises, whatever the environment looks like;
  - manifests record enough about their inputs to detect that an upstream
    artifact changed underneath them.
"""

from __future__ import annotations

import json
import os

import pytest

from wato_common import manifest as common_manifest
from wato_common import provenance


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_environment_reads_baked_build_args(monkeypatch):
    monkeypatch.setenv("WATO_GIT_COMMIT", "abc123")
    monkeypatch.setenv("WATO_GIT_DIRTY", "true")
    monkeypatch.setenv("WATO_BUILD_TIME", "2026-09-21T00:00:00Z")
    monkeypatch.setenv("WATO_BASE_IMAGE", "ghcr.io/x/base@sha256:deadbeef")

    env = provenance.environment()

    assert env["git_commit"] == "abc123"
    assert env["git_dirty"] is True
    assert env["build_time"] == "2026-09-21T00:00:00Z"
    assert env["base_image"] == "ghcr.io/x/base@sha256:deadbeef"
    assert env["python"]


def test_environment_degrades_when_nothing_is_baked_in(monkeypatch):
    """A dev container with no build args must not crash a batch run."""
    for var in (
        "WATO_GIT_COMMIT",
        "WATO_GIT_DIRTY",
        "WATO_BUILD_TIME",
        "WATO_BASE_IMAGE",
    ):
        monkeypatch.delenv(var, raising=False)

    env = provenance.environment()

    assert env["git_commit"] is None
    assert env["git_dirty"] is False


def test_empty_env_var_is_treated_as_unset(monkeypatch):
    """Compose passes "" for an unset arg — that must read as None, not ""."""
    monkeypatch.setenv("WATO_GIT_COMMIT", "")
    assert provenance.environment()["git_commit"] is None


def test_config_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("threshold: 0.5\n")
    b.write_text("threshold: 0.6\n")

    assert provenance.config_hash(str(a)) != provenance.config_hash(str(b))
    assert provenance.config_hash(None) is None
    assert provenance.config_hash(str(tmp_path / "missing.yaml")) is None


def test_collect_includes_model_revisions():
    block = provenance.collect(
        component="perception_2d", models={"sam2": "facebook/sam2.1@abc"}
    )
    assert block["component"] == "perception_2d"
    assert block["models"]["sam2"] == "facebook/sam2.1@abc"


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
@pytest.fixture
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    return tmp_path


def test_manifest_hashes_small_inputs(artifact_root, tmp_path):
    src = tmp_path / "frame_index.parquet"
    src.write_bytes(b"some bytes")

    m = common_manifest.build(
        component="perception_2d",
        bag_id="bag1",
        chunk_id="chunk0",
        inputs={"frame_index": f"file://{src}"},
        outputs={"detections": "file:///out/detections.parquet"},
    )

    entry = m["inputs"]["frame_index"]
    assert entry["sha256"] is not None
    assert entry["size_bytes"] == len(b"some bytes")


def test_manifest_input_hash_detects_upstream_change(artifact_root, tmp_path):
    """The whole point: a changed upstream artifact yields a changed manifest."""
    src = tmp_path / "frame_index.parquet"
    src.write_bytes(b"v1")
    before = common_manifest.build(
        component="c", bag_id="b", chunk_id="k", inputs={"fi": f"file://{src}"}
    )
    src.write_bytes(b"v2")
    after = common_manifest.build(
        component="c", bag_id="b", chunk_id="k", inputs={"fi": f"file://{src}"}
    )

    assert before["inputs"]["fi"]["sha256"] != after["inputs"]["fi"]["sha256"]


def test_manifest_records_missing_input_rather_than_raising(artifact_root, tmp_path):
    m = common_manifest.build(
        component="c",
        bag_id="b",
        chunk_id="k",
        inputs={"gone": f"file://{tmp_path / 'nope.parquet'}"},
    )
    assert m["inputs"]["gone"]["present"] is False


def test_large_input_skips_hashing(artifact_root, tmp_path, monkeypatch):
    """Hashing every lidar sweep would cost minutes per chunk — size+mtime instead."""
    monkeypatch.setattr(common_manifest, "HASH_SIZE_LIMIT_BYTES", 4)
    big = tmp_path / "sweeps.npz"
    big.write_bytes(b"12345678")

    entry = common_manifest.build(
        component="c", bag_id="b", chunk_id="k", inputs={"big": f"file://{big}"}
    )["inputs"]["big"]

    assert entry["sha256"] is None
    assert entry["mtime"] is not None
    assert entry["size_bytes"] == 8


def test_write_places_component_manifests_side_by_side(artifact_root):
    """Several components write into one chunk dir without clobbering."""
    a = common_manifest.write(component="ingest", bag_id="b", chunk_id="k")
    b = common_manifest.write(
        component="perception_2d",
        bag_id="b",
        chunk_id="k",
        filename=common_manifest.component_manifest_name("perception_2d"),
    )

    assert a != b
    assert a.endswith("manifest.json")
    assert b.endswith("manifest_perception_2d.json")
    for uri in (a, b):
        with open(uri.replace("file://", ""), encoding="utf-8") as fh:
            assert json.load(fh)["schema_version"] == 2


def test_written_manifest_is_json_serializable(artifact_root):
    uri = common_manifest.write(
        component="perception_2d",
        bag_id="b",
        chunk_id="k",
        models={"sam2": "facebook/sam2.1@abc"},
    )
    path = uri.replace("file://", "")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["provenance"]["models"]["sam2"] == "facebook/sam2.1@abc"
