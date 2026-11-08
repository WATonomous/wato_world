"""Tests for frames.py — per-frame depth + mask inputs, cached across sweeps."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.artifacts import ArtifactFormatError, depth, masklets
from wato_common.artifacts.frame_index import CameraFrame
from wato_semantic_lifting.frames import FrameCache, MaskClaim

BAG, CHUNK, CAM = "b", "0000", "CAM_FRONT"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _ref(seq: int) -> CameraFrame:
    return CameraFrame(CAM, seq, seq * 100, f"/img/{seq}.jpg", (seq,), np.eye(4))


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


def test_frame_holds_the_pooled_depth_and_independent_masks():
    _depth(1)
    cache = _cache([_masklet("a", (1,)), _masklet("b", (2,))])
    frame = cache.get(_ref(1))
    assert frame.size == (6, 4)
    (mask,) = frame.masks
    assert mask.label == 0
    assert mask.sample(np.array([1, 0]), np.array([1, 0])).tolist() == [True, False]
    assert np.all(frame.occluder_m == 5.0)


def test_overlapping_masks_are_cached_without_pixel_ownership():
    _depth(1)
    frame = _cache([_masklet("a", (1,)), _masklet("b", (1,))]).get(_ref(1))
    assert len(frame.masks) == 2
    assert all(mask.sample(np.array([1]), np.array([1]))[0] for mask in frame.masks)


def test_cached_masks_store_only_their_rectangle_without_dense_backing_arrays():
    shape = (100, 200)
    _depth(1, shape=shape)
    frame = _cache([_masklet("a", (1,), shape), _masklet("b", (1,), shape)]).get(
        _ref(1)
    )
    assert sum(mask.pixels.nbytes for mask in frame.masks) == 12
    assert all(mask.pixels.base is None for mask in frame.masks)
    assert frame.size == (200, 100)


def test_crop_sampling_preserves_holes_and_rejects_all_pixels_outside_the_roi():
    dense = np.zeros((20, 30), bool)
    dense[7:10, 12:16] = True
    dense[8, 13] = False
    mask = MaskClaim.from_mask(4, dense)
    ui = np.array([12, 13, 15, 11, 16, 12, 12, -1, 100])
    vi = np.array([7, 8, 9, 7, 7, 6, 10, 7, 7])
    assert mask.sample(ui, vi).tolist() == [
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    # Cropping owns storage: changing the decoded image cannot corrupt cache.
    dense[:] = False
    assert mask.sample(np.array([12]), np.array([7])).tolist() == [True]


def test_empty_mask_sampling_is_all_background():
    mask = MaskClaim.from_mask(0, np.zeros((10, 20), bool))
    assert mask.pixels.nbytes == 0
    assert mask.sample(np.array([0, 10]), np.array([0, 5])).tolist() == [False, False]


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
