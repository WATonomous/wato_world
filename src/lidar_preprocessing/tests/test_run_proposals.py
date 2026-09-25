"""Tests for pipeline.run_proposals (Steps E + F orchestration)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    chunk_root,
    global_iwu_path,
    local_path,
    motion_clusters_path,
)
from wato_common.manifest import component_manifest_name
from wato_lidar_preprocessing import pipeline

from ._staging import write_global_static_map
from .test_motion_proposals import _cfg, _stage_scene


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _manifest(bag, chunk):
    path = os.path.join(
        local_path(chunk_root(bag, chunk)),
        component_manifest_name("lidar_preprocessing"),
    )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_run_proposals_runs_iwu_then_f_and_records_manifest(tmp_env):
    bag = "bag_rp"
    _stage_scene(bag, "c0", with_iwu=False)
    write_global_static_map(bag, np.array([[30.0, 0.0, 1.0]]))
    pipeline.run_proposals(_cfg(), bag_id=bag)
    assert os.path.exists(local_path(global_iwu_path(bag)))
    assert os.path.exists(local_path(motion_clusters_path(bag, "c0")))
    m = _manifest(bag, "c0")
    assert "motion_clusters" in json.dumps(m["outputs"])
    assert "global_iwu" in json.dumps(m["inputs"])


def test_run_proposals_single_chunk_skips_iwu(tmp_env):
    bag = "bag_rp_chunk"
    _stage_scene(bag, "c0", with_iwu=False)
    write_global_static_map(bag, np.array([[30.0, 0.0, 1.0]]))
    pipeline.run_proposals(_cfg(), bag_id=bag, chunk_id="c0")
    assert not os.path.exists(local_path(global_iwu_path(bag)))
    assert os.path.exists(local_path(motion_clusters_path(bag, "c0")))


def test_run_proposals_is_idempotent_until_forced(tmp_env):
    bag = "bag_rp_idem"
    _stage_scene(bag, "c0")
    pipeline.run_proposals(_cfg(), bag_id=bag, with_iwu=False)
    out = local_path(motion_clusters_path(bag, "c0"))
    t = os.path.getmtime(out)
    pipeline.run_proposals(_cfg(), bag_id=bag, with_iwu=False)
    assert os.path.getmtime(out) == t
    pipeline.run_proposals(_cfg(), bag_id=bag, with_iwu=False, force=True)
    assert os.path.getmtime(out) > t


def test_run_proposals_disabled_writes_nothing(tmp_env):
    bag = "bag_rp_off"
    _stage_scene(bag, "c0")
    pipeline.run_proposals(_cfg(enabled=False), bag_id=bag, with_iwu=False)
    assert not os.path.exists(local_path(motion_clusters_path(bag, "c0")))
