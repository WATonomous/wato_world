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
    )
    ped = masklets.MaskletRecord(
        masklet_id="m1",
        cam_id="CAM_BACK",
        cls="pedestrian",
        score=0.7,
        frames_present=(8,),
    )
    masks = {("m0", 7): _mask(), ("m0", 8): _mask(box=(0, 2, 0, 2))}
    masks[("m1", 8)] = _mask(box=(2, 4, 3, 6))
    for (mid, seq), m in masks.items():
        masklets.write_mask(BAG, CHUNK, mid, seq, m)
    masklets.write(BAG, CHUNK, [car, ped])

    assert masklets.exists(BAG, CHUNK)
    assert masklets.read(BAG, CHUNK) == [car, ped]

    index = masklets.MaskletIndex.load(BAG, CHUNK)
    assert index.at("CAM_FRONT", 8) == [car]
    assert index.at("CAM_BACK", 8) == [ped]
    assert index.at("CAM_FRONT", 9) == []
    for (mid, seq), m in masks.items():
        (rec,) = [r for r in index.records if r.masklet_id == mid]
        np.testing.assert_array_equal(index.mask(rec, seq), m)


def _embeddings(**changes) -> masklets.MaskletEmbeddings:
    f = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.0, 1.0, 0.0]])
    fields = dict(
        model="dinov2_vitl14",
        pooling="ffa",
        masklet_ids=np.array(["m0", "m0", "m1"]),
        cam_ids=np.array(["CAM_FRONT", "CAM_FRONT", "CAM_BACK"]),
        camera_seq=np.array([7, 8, 8]),
        camera_timestamp_ns=np.array([700, 800, 805]),
        features=f,
    )
    return masklets.MaskletEmbeddings(**{**fields, **changes})


def test_embeddings_round_trip():
    assert not masklets.embeddings_exist(BAG, CHUNK)
    with pytest.raises(FileNotFoundError, match="embed"):
        masklets.read_embeddings(BAG, CHUNK)
    wrote = _embeddings()
    masklets.write_embeddings(BAG, CHUNK, wrote)

    assert masklets.embeddings_exist(BAG, CHUNK)
    read = masklets.read_embeddings(BAG, CHUNK)
    assert (read.model, read.pooling, len(read)) == ("dinov2_vitl14", "ffa", 3)
    assert read.features.dtype == np.float16
    assert read.camera_seq.dtype == np.int64
    for name in ("masklet_ids", "cam_ids", "camera_seq", "camera_timestamp_ns"):
        np.testing.assert_array_equal(getattr(read, name), getattr(wrote, name))
    np.testing.assert_allclose(read.features, wrote.features, atol=1e-3)


def test_chunk_with_nothing_embedded_round_trips():
    empty = np.zeros(0, dtype=np.int64)
    masklets.write_embeddings(
        BAG,
        CHUNK,
        _embeddings(
            masklet_ids=np.zeros(0, dtype=str),
            cam_ids=np.zeros(0, dtype=str),
            camera_seq=empty,
            camera_timestamp_ns=empty,
            features=np.zeros((0, 3)),
        ),
    )
    assert len(masklets.read_embeddings(BAG, CHUNK)) == 0


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"features": np.ones((3, 3))}, "unit L2 norm"),
        ({"camera_seq": np.array([7, 8])}, "camera_seq has 2 rows"),
        ({"pooling": ""}, "no model or pooling"),
    ],
)
def test_malformed_embeddings_are_not_written(changes, reason):
    with pytest.raises(ArtifactFormatError, match=reason):
        masklets.write_embeddings(BAG, CHUNK, _embeddings(**changes))
    assert not masklets.embeddings_exist(BAG, CHUNK)


def test_embeddings_file_missing_an_array_raises():
    path = masklets.write_embeddings(BAG, CHUNK, _embeddings())
    with np.load(path) as z:
        kept = {k: z[k] for k in z.files if k != "camera_timestamp_ns"}
    np.savez(path, **kept)
    with pytest.raises(ArtifactFormatError, match="camera_timestamp_ns"):
        masklets.read_embeddings(BAG, CHUNK)


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
    assert row["mask_semantics"] == "independent"
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


def test_row_without_a_tracker_backend_raises():
    """sam2 and dam4sam make different masks: a row naming neither is not
    read as either."""
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
    row["tracker_backend"] = None
    write_table([row], MASKLET_SCHEMA, tracklets_2d_path(BAG, CHUNK))
    with pytest.raises(ArtifactFormatError, match="tracker_backend"):
        masklets.read(BAG, CHUNK)


@pytest.mark.parametrize("semantics", [None, "owned"])
def test_masks_without_independent_claim_semantics_are_rejected(semantics):
    row = MaskletRow(
        masklet_id="m0",
        bag_id=BAG,
        chunk_id=CHUNK,
        cam_id="c",
        cls="car",
        score=0.9,
        frames_present="[1]",
        mask_path="",
    ).model_dump()
    row["mask_semantics"] = semantics
    write_table([row], MASKLET_SCHEMA, tracklets_2d_path(BAG, CHUNK))
    with pytest.raises(ArtifactFormatError, match="independent.*re-run perception_2d"):
        masklets.read(BAG, CHUNK)
