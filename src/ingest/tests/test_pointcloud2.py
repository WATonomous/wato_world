"""Test the pure-Python PointCloud2 byte decoder against a synthetic buffer."""

from __future__ import annotations

import struct

import numpy as np

from wato_common.io.pointcloud2 import PointFieldSpec, decode_pointcloud2


def test_decode_xyz_intensity():
    # Build a 4-point cloud: 4 floats per point (x, y, z, intensity), little-endian.
    point_step = 16
    points = [
        (1.0, 2.0, 3.0, 0.5),
        (4.0, 5.0, 6.0, 0.8),
        (-1.0, -2.0, -3.0, 0.0),
        (10.0, 20.0, 30.0, 1.0),
    ]
    buf = b"".join(struct.pack("<ffff", *p) for p in points)

    fields = [
        PointFieldSpec("x", 0, 7),
        PointFieldSpec("y", 4, 7),
        PointFieldSpec("z", 8, 7),
        PointFieldSpec("intensity", 12, 7),
    ]
    out = decode_pointcloud2(
        buf, fields, point_step=point_step, width=4, height=1, is_bigendian=False
    )

    np.testing.assert_allclose(out["x"], [1.0, 4.0, -1.0, 10.0])
    np.testing.assert_allclose(out["y"], [2.0, 5.0, -2.0, 20.0])
    np.testing.assert_allclose(out["z"], [3.0, 6.0, -3.0, 30.0])
    np.testing.assert_allclose(out["intensity"], [0.5, 0.8, 0.0, 1.0])


def test_decode_with_padding_offsets():
    # 24-byte point: x(0..3), y(4..7), z(8..11), intensity(12..15), ring(16..17), pad(18..23).
    point_step = 24
    pts_packed = []
    for x, y, z, inten, ring in [(1.0, 2.0, 3.0, 0.7, 12), (4.0, 5.0, 6.0, 0.2, 5)]:
        pts_packed.append(struct.pack("<ffffH6x", x, y, z, inten, ring))
    buf = b"".join(pts_packed)
    assert all(len(p) == point_step for p in pts_packed)

    fields = [
        PointFieldSpec("x", 0, 7),
        PointFieldSpec("y", 4, 7),
        PointFieldSpec("z", 8, 7),
        PointFieldSpec("intensity", 12, 7),
        PointFieldSpec("ring", 16, 4),  # UINT16
    ]
    out = decode_pointcloud2(
        buf, fields, point_step=point_step, width=2, height=1, is_bigendian=False
    )

    np.testing.assert_allclose(out["x"], [1.0, 4.0])
    np.testing.assert_allclose(out["intensity"], [0.7, 0.2])
    np.testing.assert_array_equal(out["ring"], [12, 5])


def test_empty_cloud_returns_empty_arrays():
    out = decode_pointcloud2(b"", [PointFieldSpec("x", 0, 7)], point_step=4, width=0)
    assert out["x"].size == 0
