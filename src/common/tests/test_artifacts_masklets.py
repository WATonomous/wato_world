"""wato_common.artifacts.masklets: what perception_2d writes, consumers read back."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PIL")  # stub images don't install Pillow

from wato_common.artifact_store import tracklets_2d_path  # noqa: E402
from wato_common.artifacts import ArtifactFormatError, masklets  # noqa: E402
from wato_common.io.parquet_io import read_rows, write_table  # noqa: E402
from wato_common.schemas import MASKLET_SCHEMA, MaskletRow  # noqa: E402

BAG, CHUNK = "b", "0000"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _mask(h=4, w=6, box=(1, 3, 2, 5)) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[box[0] : box[1], box[2] : box[3]] = True
    return m


def test_round_trip():
    car = masklets.MaskletRecord(
        masklet_id="m0",
        cam_id="CAM_FRONT",
        cls="vehicle",
        score=0.4,
        frames_present=(7, 8),
        raw_phrase="car",
        det_score=0.9,
        dino_feature_path=masklets.write_dino_feature(
            BAG, CHUNK, "m0", np.ones(4, dtype=np.float32)
        ),
    )
    ped = masklets.MaskletRecord(
        masklet_id="m1",
        cam_id="CAM_BACK",
        cls="pedestrian",
        score=0.7,
        frames_present=(8,),
        global_object_id="obj3",
    )
    masks = {("m0", 7): _mask(), ("m0", 8): _mask(box=(0, 2, 0, 2))}
    masks[("m1", 8)] = _mask(box=(2, 4, 3, 6))
    for (mid, seq), m in masks.items():
        masklets.write_mask(BAG, CHUNK, mid, seq, m)
    masklets.write(BAG, CHUNK, [car, ped])

    assert masklets.exists(BAG, CHUNK)
    assert masklets.read(BAG, CHUNK) == [car, ped]
    np.testing.assert_array_equal(np.load(car.dino_feature_path), np.ones(4))

    index = masklets.MaskletIndex.load(BAG, CHUNK)
    assert index.at("CAM_FRONT", 8) == [car]
    assert index.at("CAM_BACK", 8) == [ped]
    assert index.at("CAM_FRONT", 9) == []
    for (mid, seq), m in masks.items():
        (rec,) = [r for r in index.records if r.masklet_id == mid]
        np.testing.assert_array_equal(index.mask(rec, seq), m)
    assert car.instance_id == "m0" and ped.instance_id == "obj3"


def test_frames_present_is_stored_as_a_json_string():
    """The on-disk encoding semantic_lifting once failed to decode (read as [])."""
    masklets.write(
        BAG,
        CHUNK,
        [
            masklets.MaskletRecord(
                masklet_id="m0", cam_id="c", cls="vehicle", score=1.0,
                frames_present=(3, 4, 5),
            )
        ],
    )  # fmt: skip
    (row,) = read_rows(tracklets_2d_path(BAG, CHUNK))
    assert row["frames_present"] == "[3, 4, 5]"
    assert masklets.read(BAG, CHUNK)[0].frames_present == (3, 4, 5)


def test_empty_chunk():
    masklets.write(BAG, CHUNK, [])
    assert masklets.exists(BAG, CHUNK)
    assert masklets.MaskletIndex.load(BAG, CHUNK).at("c", 0) == []


def test_listed_frame_without_a_mask_raises():
    rec = masklets.MaskletRecord(
        masklet_id="m0", cam_id="c", cls="vehicle", score=1.0, frames_present=(1, 2)
    )
    masklets.write_mask(BAG, CHUNK, "m0", 1, _mask())
    masklets.write(BAG, CHUNK, [rec])
    index = masklets.MaskletIndex.load(BAG, CHUNK)
    index.mask(rec, 1)
    with pytest.raises(ArtifactFormatError, match="lists frame 2"):
        index.mask(rec, 2)


def test_row_without_det_score_votes_with_its_score():
    row = MaskletRow(
        masklet_id="m0",
        bag_id=BAG,
        chunk_id=CHUNK,
        cam_id="c",
        cls="vehicle",
        score=0.3,
        frames_present="[1]",
        mask_path="",
    ).model_dump()
    row["det_score"] = None
    write_table([row], MASKLET_SCHEMA, tracklets_2d_path(BAG, CHUNK))
    assert masklets.read(BAG, CHUNK)[0].det_score == 0.3
