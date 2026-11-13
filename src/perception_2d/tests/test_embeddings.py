"""models/embeddings.py: FFA pooling and the chunk pass over saved masks."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PIL")

from wato_common.artifacts import masklets  # noqa: E402
from wato_common.artifacts.frame_index import CameraFrame  # noqa: E402
from wato_perception_2d.models import embeddings  # noqa: E402

BAG, CHUNK = "b", "0000"
GRID = embeddings.INPUT_SIZE // 14


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def test_pool_averages_only_the_tokens_under_the_mask():
    tokens = np.zeros((1, 4, 2), dtype=np.float32)
    tokens[0, :2] = [3.0, 4.0]  # foreground tokens
    tokens[0, 2:] = [-50.0, 9.0]  # background must not leak in
    grid = np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(embeddings.pool(tokens, grid), [[0.6, 0.8]])


def test_patch_mask_covers_the_masked_part_of_the_crop():
    crop = np.zeros((40, 80), dtype=bool)
    crop[:, :40] = True  # left half
    grid = embeddings.patch_mask(crop)
    assert grid.shape == (GRID, GRID)
    assert grid[:, : GRID // 2 - 1].all() and not grid[:, GRID // 2 + 1 :].any()


def test_mask_covering_no_token_keeps_the_whole_crop():
    crop = np.zeros((200, 200), dtype=bool)
    crop[100, 100] = True
    assert embeddings.patch_mask(crop).all()


def test_wide_crop_is_resized_whole_not_centre_cropped():
    """A centre crop of a 4:1 car keeps its middle quarter; the ends are what
    tell two cars apart."""
    crop = np.zeros((10, 40, 3), dtype=np.uint8)
    crop[:, :10] = 255  # the left end
    out = embeddings.model_input(crop)
    assert out.shape == (embeddings.INPUT_SIZE, embeddings.INPUT_SIZE, 3)
    assert out[:, :40].mean() > 200 and out[:, -40:].mean() < 50


def _frame(cam: str, seq: int, path: str) -> CameraFrame:
    return CameraFrame(
        cam_id=cam,
        camera_seq=seq,
        timestamp_ns=1000 + seq,
        image_path=path,
        sweep_ids=(),
        world_T_ego=None,
    )


def test_embed_chunk_embeds_each_masklet_on_every_kth_image(tmp_path, monkeypatch):
    from PIL import Image

    path = str(tmp_path / "img.png")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    records = [
        masklets.MaskletRecord("m0", "c", "car", 1.0, frames_present=(0, 1, 2)),
        masklets.MaskletRecord("m1", "c", "car", 1.0, frames_present=(2,)),
    ]
    for r in records:
        for seq in r.frames_present:
            masklets.write_mask(BAG, CHUNK, r.masklet_id, seq, mask)
    seen = []

    def fake(image, masks, model, device):
        seen.append(len(masks))
        return [np.array([1.0, 0.0], dtype=np.float32)] * len(masks)

    monkeypatch.setattr(embeddings, "extract_dino_features", fake)
    out = embeddings.embed_chunk(
        BAG,
        CHUNK,
        records,
        {"c": [_frame("c", s, path) for s in range(4)]},
        model_name="dinov2_vitl14",
        every_k_frames=2,
        device="cpu",
    )
    assert seen == [1, 2]  # images 0 and 2; image 2 holds both masklets
    assert out.masklet_ids.tolist() == ["m0", "m0", "m1"]
    assert out.camera_seq.tolist() == [0, 2, 2]
    assert out.camera_timestamp_ns.tolist() == [1000, 1002, 1002]
    assert (out.model, out.pooling) == ("dinov2_vitl14", "ffa")
    masklets.write_embeddings(BAG, CHUNK, out)
    assert len(masklets.read_embeddings(BAG, CHUNK)) == 3


def test_chunk_without_masklets_writes_an_empty_file():
    out = embeddings.embed_chunk(
        BAG, CHUNK, [], {}, model_name="m", every_k_frames=1, device="cpu"
    )
    masklets.write_embeddings(BAG, CHUNK, out)
    assert len(masklets.read_embeddings(BAG, CHUNK)) == 0
