import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    dynamic_mask_path,
    lidar_proc_index_path,
    lidar_world_path,
    local_path,
    static_mask_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_perception_2d.io import clear_lidar_caches, load_static_lidar_points


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def test_static_depth_anchors_use_composite_identity_and_explicit_static_mask(tmp_env):
    bag_id, chunk_id, sweep_id = "bag_depth", "chunk0", 0
    rows = []
    for lidar_id, x, static in (
        ("LIDAR_LEFT", np.array([1.0, 2.0]), np.array([True, False])),
        ("LIDAR_RIGHT", np.array([10.0]), np.array([True])),
    ):
        world_uri = lidar_world_path(bag_id, chunk_id, lidar_id, sweep_id)
        dynamic_uri = dynamic_mask_path(bag_id, chunk_id, lidar_id, sweep_id)
        static_uri = static_mask_path(bag_id, chunk_id, lidar_id, sweep_id)
        os.makedirs(os.path.dirname(local_path(world_uri)), exist_ok=True)
        np.savez_compressed(
            local_path(world_uri), x=x, y=np.zeros_like(x), z=np.zeros_like(x)
        )
        # LEFT point x=2 is unknown: neither static nor dynamic.
        np.save(local_path(dynamic_uri), np.zeros(x.shape[0], dtype=bool))
        np.save(local_path(static_uri), static)
        rows.append(
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "sweep_id": sweep_id,
                "lidar_id": lidar_id,
                "reference_timestamp_ns": 0,
                "n_points_total": x.shape[0],
                "n_points_static": int(static.sum()),
                "n_points_dynamic": 0,
                "world_path": world_uri,
                "dynamic_mask_path": dynamic_uri,
                "static_mask_path": static_uri,
                "has_intensity": False,
                "deskewed": True,
                "valid": True,
            }
        )
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    left = load_static_lidar_points(
        bag_id, chunk_id, "LIDAR_LEFT", sweep_id
    )
    right = load_static_lidar_points(
        bag_id, chunk_id, "LIDAR_RIGHT", sweep_id
    )

    np.testing.assert_array_equal(left[:, 0], np.array([1.0]))
    np.testing.assert_array_equal(right[:, 0], np.array([10.0]))
    clear_lidar_caches(bag_id, chunk_id)
