"""Pure-Python PointCloud2 → structured numpy decoder.

Avoids depending on `sensor_msgs_py` so tests and downstream components can read
the decoded sweeps without ROS installed.  Only the wire-decode (during ingest
ingest) needs ROS, and that path uses the rosbag reader's deserializer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# PointField datatype constants (matching sensor_msgs/PointField).
DTYPE_BY_PF = {
    1: ("i1", 1),  # INT8
    2: ("u1", 1),  # UINT8
    3: ("i2", 2),  # INT16
    4: ("u2", 2),  # UINT16
    5: ("i4", 4),  # INT32
    6: ("u4", 4),  # UINT32
    7: ("f4", 4),  # FLOAT32
    8: ("f8", 8),  # FLOAT64
}


@dataclass
class PointFieldSpec:
    name: str
    offset: int
    datatype: int
    count: int = 1


def decode_pointcloud2(
    data: bytes,
    fields: Sequence[PointFieldSpec],
    point_step: int,
    width: int,
    height: int = 1,
    is_bigendian: bool = False,
) -> dict[str, np.ndarray]:
    """Decode the raw byte buffer of a PointCloud2 message into named arrays.

    Returns a dict of field_name -> 1D numpy array.  Only count-1 scalar fields
    are supported; that covers x/y/z/intensity/ring/time which is all ingest
    cares about.
    """
    n = width * height
    if n == 0 or not data:
        return {f.name: np.zeros(0, dtype=DTYPE_BY_PF[f.datatype][0]) for f in fields}

    buf = np.frombuffer(data, dtype=np.uint8, count=n * point_step).reshape(n, point_step)
    out: dict[str, np.ndarray] = {}
    bo = ">" if is_bigendian else "<"

    for f in fields:
        if f.count != 1:
            continue
        kind, size = DTYPE_BY_PF[f.datatype]
        dtype = np.dtype(f"{bo}{kind}")
        view = buf[:, f.offset:f.offset + size].copy()
        out[f.name] = view.view(dtype).reshape(n)

    return out


def fields_to_specs(point_fields_msg) -> list[PointFieldSpec]:
    """Convert a list of sensor_msgs/PointField objects to PointFieldSpec.

    Accepts anything with `.name`, `.offset`, `.datatype`, `.count` attrs so it
    works against rosbag2 messages without importing rclpy types here.
    """
    return [
        PointFieldSpec(
            name=str(f.name),
            offset=int(f.offset),
            datatype=int(f.datatype),
            count=int(f.count),
        )
        for f in point_fields_msg
    ]
