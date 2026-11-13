"""debug-bag's camera layers: masks and depth drawn as images."""

import io

import numpy as np
import pytest
from PIL import Image

from wato_common.artifacts import depth, masklets
from wato_common.artifacts.masklets import MaskletRecord
from wato_ingest import debug_images as di

RED = (1.0, 0.0, 0.0)


def _decode(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)


def test_depth_is_red_near_blue_far_and_black_without_a_measurement():
    lut = di.turbo_lut()
    d = np.array([[1.0, 10.0, 100.0, 0.5, 1000.0, 0.0, np.nan]], np.float32)
    rgb = di.colour_depth(d)
    near, mid, far, nearer, farther, none, nan = (tuple(p) for p in rgb[0])
    assert near == tuple(lut[255]) and near[0] > near[2]  # red end
    assert far == tuple(lut[0]) and far[2] > far[0]  # blue end
    assert mid == tuple(lut[128])  # log scale: 10 m is mid-scale over 1-100 m
    assert nearer == near and farther == far  # saturated, not wrapped
    assert none == nan == (0, 0, 0)


def test_masks_blend_inside_and_are_outlined_at_their_edge():
    image = np.zeros((12, 12, 3), np.uint8)
    m = np.zeros((12, 12), bool)
    m[2:10, 2:10] = True
    out = di.tint(image, [(m, RED)])
    assert tuple(out[6, 6]) == (round(255 * di.MASK_ALPHA), 0, 0)
    assert tuple(out[2, 6]) == tuple(out[3, 6]) == (255, 0, 0)  # 2 px outline
    assert tuple(out[0, 0]) == (0, 0, 0)
    with pytest.raises(ValueError):
        di.tint(image, [(np.ones((4, 4), bool), RED)])


def test_overlapping_masks_both_show():
    image = np.zeros((20, 20, 3), np.uint8)
    a = np.zeros((20, 20), bool)
    b = np.zeros((20, 20), bool)
    a[2:14, 2:14] = True
    b[6:18, 6:18] = True
    out = di.tint(image, [(a, RED), (b, (0.0, 0.0, 1.0))])
    assert out[10, 10, 0] > 0 and out[10, 10, 2] > 0


def test_masklet_colour_is_stable_per_id():
    assert di.id_rgb("4a29b371-24a") == di.id_rgb("4a29b371-24a")
    assert di.id_rgb("4a29b371-24a") != di.id_rgb("223010e9-66a")


def _frame(tmp_path, records=(), seq=3):
    path = tmp_path / "img.png"
    Image.fromarray(np.full((240, 320, 3), 128, np.uint8)).save(path)
    return di.ImageFrame(
        bag_id="b",
        chunk_id="0000",
        cam_id="CAM_A",
        camera_seq=seq,
        image_path=str(path),
        header_ns=1,
        log_ns=2,
        frame_id="cam_a",
        masklets=tuple(records),
    )


def test_render_draws_each_layer_from_the_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    m = np.zeros((240, 320), bool)
    m[120:200, 40:200] = True
    masklets.write_mask("b", "0000", "m-1", 3, m)
    d = np.full((240, 320), 8.0, np.float16)
    depth.write_frame(
        "b", "0000", "CAM_A", 3,
        depth.DepthFrame(d, 1.0, 0.0, 100, 80, 0.3, depth.FitStatus.OK),
    )  # fmt: skip
    rec = MaskletRecord(
        masklet_id="m-1", cam_id="CAM_A", cls="car", score=0.9, frames_present=(3,)
    )
    out = di.render(_frame(tmp_path, [rec]), di.LAYERS)
    masks, coloured = _decode(out[di.MASKS]), _decode(out[di.DEPTH])
    assert masks.shape == coloured.shape == (240, 320, 3)
    assert abs(masks[160, 120] - 128).max() > 20  # tinted inside the mask
    assert abs(masks[230, 300] - 128).max() < 8  # the image elsewhere
    expected = di.turbo_lut()[int(round(float(di.depth_t(np.float32(8.0))) * 255))]
    assert abs(coloured[100, 160] - expected).max() < 12  # clear of the legend


def test_a_frame_without_depth_is_black(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    out = di.render(_frame(tmp_path), (di.DEPTH,))
    assert set(out) == {di.DEPTH}
    assert _decode(out[di.DEPTH])[30:, :].max() < 8  # below the title
