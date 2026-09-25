"""Tests for decoders/lidar.decode_chunk: sweep_id names one sweep uniquely
within the chunk even with several LiDARs.  No rosbag: `lidar.messages` is
monkeypatched with fake PointCloud2 messages."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wato_common.artifact_store import lidar_sweeps_path, local_path
from wato_common.io.parquet_io import read_rows
from wato_ingest.config import load_config
from wato_ingest.decoders import lidar

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
FLOAT32 = 7


def _cloud(t_ns: int, n: int = 4):
    PointCloud2 = type("PointCloud2", (), {})
    msg = PointCloud2()
    msg.header = SimpleNamespace(
        stamp=SimpleNamespace(sec=t_ns // 10**9, nanosec=t_ns % 10**9)
    )
    msg.fields = [
        SimpleNamespace(name=c, offset=4 * i, datatype=FLOAT32, count=1)
        for i, c in enumerate("xyz")
    ]
    msg.data = np.ones((n, 3), np.float32).tobytes()
    msg.point_step, msg.width, msg.height, msg.is_bigendian = 12, n, 1, False
    return msg


def test_sweep_ids_are_unique_across_lidars(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    cfg = load_config(str(CONFIG_DIR / "ingest.wato.yaml"))
    topic = {lid: t for lid, t in cfg.topics.lidars.items()}
    # Three scanners interleaved in record order, as on the WATO rig.
    stream = [
        (topic[lid], _cloud(k * 100_000_000 + j), k * 100_000_000 + j)
        for k in range(3)
        for j, lid in enumerate(("lidar_cc", "lidar_ne", "lidar_nw"))
    ]

    @contextmanager
    def fake_messages(*_a, **_kw):
        yield iter(stream)

    monkeypatch.setattr(lidar, "messages", fake_messages)
    results = lidar.decode_chunk("bag", "b", "0000", t_start_ns=0, t_end_ns=1, cfg=cfg)

    rows = read_rows(lidar_sweeps_path("b", "0000"))
    assert [r["sweep_id"] for r in rows] == list(range(9))
    assert len({r["lidar_path"] for r in rows}) == 9
    assert all(Path(local_path(r["lidar_path"])).exists() for r in rows)
    assert {r.lidar_id: r.sweeps_written for r in results} == {
        "lidar_cc": 3,
        "lidar_ne": 3,
        "lidar_nw": 3,
    }
