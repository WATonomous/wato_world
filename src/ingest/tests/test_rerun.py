"""Re-running ingest: a bag whose every chunk has its manifest is skipped."""

from __future__ import annotations

import os
import json

import pytest

from wato_common.artifact_store import chunks_index_path, local_path, manifest_path
from wato_common.io.parquet_io import write_table
from wato_common.schemas import CHUNK_SCHEMA
from wato_ingest.config import load_config
from wato_ingest.pipeline import bag_ingested, chunk_ingested, run_bag

BAG = "some_bag_mcap"
CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "ingest.yaml")


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _index(*chunk_ids):
    write_table(
        [
            dict(
                bag_id=BAG,
                chunk_id=c,
                t_start_ns=0,
                t_end_ns=1,
                t_overlap_start_ns=0,
                t_overlap_end_ns=1,
            )
            for c in chunk_ids
        ],
        CHUNK_SCHEMA,
        chunks_index_path(BAG),
    )


def _manifest(chunk_id, extra_lidar=None):
    path = local_path(manifest_path(BAG, chunk_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        cfg = load_config(CONFIG)
        lidars = list(cfg.topics.lidars.values())
        if extra_lidar:
            lidars.append(extra_lidar)
        json.dump(
            {
                "topic_check": {
                    "found_camera_images": [
                        c.image for c in cfg.topics.cameras.values()
                    ],
                    "found_lidars": lidars,
                }
            },
            fh,
        )


def test_a_bag_is_ingested_when_every_chunk_has_its_manifest():
    assert not bag_ingested(BAG)  # no chunk index
    _index("0000", "0001")
    _manifest("0000")
    assert chunk_ingested(BAG, "0000") and not chunk_ingested(BAG, "0001")
    assert not bag_ingested(BAG)
    assert bag_ingested(BAG, only_chunk="0000")
    _manifest("0001")
    assert bag_ingested(BAG)
    assert not bag_ingested(BAG, only_chunk="0002")


def test_an_ingested_bag_is_skipped_without_opening_it():
    _index("0000")
    _manifest("0000")
    # The bag isn't there: a skip must not read it (or rewrite bag_meta.json,
    # calibration.json or the chunk index).
    missing = "/nonexistent/some_bag.mcap"
    cfg = load_config(CONFIG)
    assert run_bag(bag_path=missing, cfg=cfg, config_path=CONFIG) == []
    with pytest.raises(Exception):
        run_bag(bag_path=missing, cfg=cfg, config_path=CONFIG, force=True)


@pytest.mark.parametrize("partial", [False, True])
def test_completed_chunks_with_other_sensors_cannot_be_silently_reused(partial):
    _index(*(("0000", "0001") if partial else ("0000",)))
    _manifest("0000", extra_lidar="/corner/points")
    with pytest.raises(ValueError, match="sensor selection differs"):
        run_bag(
            bag_path="/nonexistent/some_bag.mcap",
            cfg=load_config(CONFIG),
            config_path=CONFIG,
        )
