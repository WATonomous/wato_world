"""Tests for decoders/cameras: raw Image encodings become correctly ordered
RGB / grayscale PNGs, compressed images are typed by their bytes, and
anything downstream can't read fails instead of being written."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from wato_common.artifact_store import camera_dir, ensure_local_dir, local_path
from wato_ingest.decoders.cameras import UnsupportedImageError, _write_one_image

HEADER = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=500))


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    ensure_local_dir(camera_dir("b", "0000", "CAM"))


def _raw(encoding: str, pixels: np.ndarray, *, pad: int = 0):
    """sensor_msgs/Image stand-in; `pad` extra bytes at the end of each row."""
    h, w = pixels.shape[:2]
    rows = pixels.reshape(h, -1)
    rows = np.hstack([rows, np.full((h, pad), 7, np.uint8)])
    Img = type("Image", (), {})
    msg = Img()
    msg.header, msg.encoding = HEADER, encoding
    msg.height, msg.width, msg.step = h, w, rows.shape[1]
    msg.data = rows.tobytes()
    return msg


def _compressed(data: bytes, fmt: str = ""):
    Comp = type("CompressedImage", (), {})
    msg = Comp()
    msg.header, msg.format, msg.data = HEADER, fmt, data
    return msg


def _write(msg):
    row, written = _write_one_image(
        msg=msg, bag_id="b", chunk_id="0000", cam_id="CAM", seq=0, record_ts_ns=0
    )
    assert written
    return row


def _read(row) -> np.ndarray:
    return np.asarray(Image.open(local_path(row["image_path"])))


RED = [255, 0, 0]


@pytest.mark.parametrize(
    "encoding,pixel",
    [
        ("rgb8", [255, 0, 0]),
        ("bgr8", [0, 0, 255]),
        ("rgba8", [255, 0, 0, 9]),
        ("bgra8", [0, 0, 255, 9]),
    ],
)
def test_color_encodings_become_rgb(encoding, pixel):
    pixels = np.tile(np.array(pixel, np.uint8), (2, 3, 1))
    row = _write(_raw(encoding, pixels))
    out = _read(row)
    assert out.shape == (2, 3, 3)
    assert (out == RED).all()
    assert row["image_path"].endswith(".png") and row["width"] == 3


def test_row_padding_is_stripped():
    pixels = np.arange(6, dtype=np.uint8).reshape(2, 3)
    out = _read(_write(_raw("mono8", pixels, pad=5)))
    np.testing.assert_array_equal(out, pixels)


@pytest.mark.parametrize("encoding", ["bayer_rggb8", "mono16", "yuv422"])
def test_unsupported_raw_encoding_raises(encoding):
    with pytest.raises(UnsupportedImageError, match=encoding):
        _write_one_image(
            msg=_raw(encoding, np.zeros((2, 4), np.uint8)),
            bag_id="b",
            chunk_id="0000",
            cam_id="CAM",
            seq=0,
            record_ts_ns=0,
        )


def _encoded(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((4, 5, 3), np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt,ext", [("JPEG", ".jpg"), ("PNG", ".png")])
def test_compressed_type_comes_from_the_bytes(fmt, ext):
    # Drivers fill `format` inconsistently (often empty); the bytes decide.
    row = _write(_compressed(_encoded(fmt), fmt=""))
    assert row["image_path"].endswith(ext)
    assert (row["width"], row["height"]) == (5, 4)


def test_compressed_non_image_bytes_raise():
    with pytest.raises(UnsupportedImageError, match="h264"):
        _write_one_image(
            msg=_compressed(b"\x00\x00\x00\x01gd", fmt="h264"),
            bag_id="b",
            chunk_id="0000",
            cam_id="CAM",
            seq=0,
            record_ts_ns=0,
        )
