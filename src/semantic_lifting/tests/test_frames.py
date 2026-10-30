"""Tests for frames.py — per-frame depth + mask inputs, cached across sweeps."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.artifacts import ArtifactFormatError, depth, masklets
from wato_semantic_lifting.frames import FrameCache
from wato_semantic_lifting.io import CameraFrameRef

BAG, CHUNK, CAM = "b", "0000", "CAM_FRONT"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _ref(seq: int) -> CameraFrameRef:
    return CameraFrameRef(CAM, seq, seq * 100, np.eye(4))


def _depth(seq: int, shape=(4, 6)) -> None:
    depth.write_frame(
        BAG, CHUNK, CAM, seq,
        depth.DepthFrame(np.full(shape, 5.0, np.float16), 1.0, 0.0, 1, 1, 0.1,
                         depth.FitStatus.OK),
    )  # fmt: skip


def _masklet(
    masklet_id: str, seqs: tuple[int, ...], shape=(4, 6)
) -> masklets.MaskletRecord:
    for s in seqs:
        m = np.zeros(shape, bool)
        m[1:3, 1:4] = True
        masklets.write_mask(BAG, CHUNK, masklet_id, s, m)
    return masklets.MaskletRecord(masklet_id, CAM, "car", 0.5, seqs)


def _cache(records, max_frames=4) -> FrameCache:
    masklets.write(BAG, CHUNK, records)
    index = masklets.MaskletIndex.load(BAG, CHUNK)
    return FrameCache(BAG, CHUNK, index, neighborhood=1, max_frames=max_frames)


def test_frame_holds_the_pooled_depth_and_the_mask_raster():
    _depth(1)
    cache = _cache([_masklet("a", (1,)), _masklet("b", (2,))])
    frame = cache.get(_ref(1))
    assert frame.size == (6, 4)
    assert frame.labels[1, 1] == 0 and frame.labels[0, 0] == -1
    assert np.all(frame.occluder_m == 5.0)


def test_frames_without_masklets_or_depth_are_none():
    _depth(1)
    cache = _cache([_masklet("a", (1, 2))])
    assert cache.get(_ref(3)) is None  # no masklet in frame 3
    assert cache.get(_ref(2)) is None  # masklet, but no depth (failed fit)
    assert cache.n_no_depth == 1


def test_a_cached_frame_is_not_read_again(monkeypatch):
    _depth(1)
    cache = _cache([_masklet("a", (1,))])
    first = cache.get(_ref(1))
    monkeypatch.setattr(depth, "read_frame", lambda *a: pytest.fail("re-read"))
    assert cache.get(_ref(1)) is first


def test_depth_and_masks_at_different_resolutions_are_rejected():
    _depth(1, shape=(8, 12))
    cache = _cache([_masklet("a", (1,))])
    with pytest.raises(ArtifactFormatError, match="differ"):
        cache.get(_ref(1))
