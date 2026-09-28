import numpy as np
import pytest

from wato_common.artifact_store import lidar_proc_index_path
from wato_common.io.parquet_io import write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_semantic_lifting.io import load_sweeps


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def test_load_sweeps_preserves_composite_identity_and_reference_timestamp(tmp_env):
    bag_id, chunk_id = "bag_lift", "chunk0"
    rows = []
    for lidar_id, timestamp in (("LIDAR_LEFT", 100), ("LIDAR_RIGHT", 200)):
        rows.append(
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "sweep_id": 0,
                "lidar_id": lidar_id,
                "reference_timestamp_ns": timestamp,
                "n_points_total": 0,
                "n_points_static": 0,
                "n_points_dynamic": 0,
                "world_path": f"/tmp/{lidar_id}.npz",
                "dynamic_mask_path": f"/tmp/{lidar_id}_dynamic.npy",
                "static_mask_path": f"/tmp/{lidar_id}_static.npy",
                "has_intensity": False,
                "deskewed": True,
                "valid": True,
            }
        )
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    sweeps = load_sweeps(bag_id, chunk_id)

    assert [(s.lidar_id, s.sweep_id) for s in sweeps] == [
        ("LIDAR_LEFT", 0),
        ("LIDAR_RIGHT", 0),
    ]
    assert [s.timestamp_ns for s in sweeps] == [100, 200]
