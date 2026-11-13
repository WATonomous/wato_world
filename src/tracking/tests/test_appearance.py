"""Appearance ReID: masklet descriptor banks, the bag's own threshold, and
where tracking uses them."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.artifacts import masklets
from wato_common.artifacts.masklets import MaskletEmbeddings
from wato_common.artifacts.proposals import Proposal
from wato_tracking import appearance
from wato_tracking.appearance import Appearance
from wato_tracking.config import ComponentConfig, RelinkConfig
from wato_tracking.relink import relink

from .test_relink import FRAMES, _car, _obs, _sizes


def _unit(deg: float) -> list[float]:
    return [np.cos(np.radians(deg)), np.sin(np.radians(deg))]


def _embeddings(rows) -> MaskletEmbeddings:
    """rows: (masklet, cam, camera_seq, direction in degrees)."""
    return MaskletEmbeddings(
        model="dinov2_vitl14",
        pooling="ffa",
        masklet_ids=np.array([r[0] for r in rows]),
        cam_ids=np.array([r[1] for r in rows]),
        camera_seq=np.array([r[2] for r in rows]),
        camera_timestamp_ns=np.array([1000 + r[2] for r in rows]),
        features=np.array([_unit(r[3]) for r in rows], dtype=np.float32),
    )


def _lifted(instance: str, sweep: int, masklet_ids, cls="car") -> Proposal:
    return Proposal(
        proposal_id=f"lifted-{instance}-{sweep:06d}",
        sweep_id=sweep,
        frame_id=sweep,
        sweep_ids=(sweep,),
        reference_timestamp_ns=sweep,
        cx=0.0,
        cy=0.0,
        cz=0.0,
        w=1.0,
        l=1.0,
        h=1.0,
        heading=0.0,
        cls=cls,
        score=1.0,
        provenance="lifted",
        supporting_masklet_ids=tuple(masklet_ids),
    )


def test_distance_is_the_closest_pair_of_descriptors():
    """One shared viewpoint is enough: a mean over each masklet's views
    would put these two 45 degrees apart."""
    app = Appearance(
        _embeddings([("a", "c", 0, 0), ("a", "c", 1, 90), ("b", "d", 0, 80)]), 100
    )
    assert app.distance("a", "b") == pytest.approx(1 - np.cos(np.radians(10)), abs=1e-6)
    assert app.distance("b", "a") == app.distance("a", "b")
    assert app.distance("a", "a") == 0.0
    assert app.distance("a", "never-embedded") is None
    assert app.between(["never-embedded", "a"], ["b"]) == app.distance("a", "b")
    assert app.between(["never-embedded"], ["b"]) is None


def test_a_bank_keeps_at_most_budget_descriptors_spread_over_the_masklet():
    rows = [("a", "c", k, k) for k in range(90)] + [("b", "c", 0, 89)]
    assert Appearance(_embeddings(rows), 100).distance("a", "b") == pytest.approx(
        0.0, abs=1e-6
    )
    # Two kept: the first and the last image, so the last view still matches.
    assert Appearance(_embeddings(rows), 2).distance("a", "b") == pytest.approx(
        0.0, abs=1e-6
    )
    assert Appearance(_embeddings(rows[:-1] + [("b", "c", 0, 45)]), 2).distance(
        "a", "b"
    ) == pytest.approx(1 - np.cos(np.radians(44)), abs=1e-3)


def test_member_cost_is_half_the_distance_of_a_pair_that_looks_alike():
    app = Appearance(
        _embeddings([("a", "c", 0, 0), ("b", "c", 0, 10), ("x", "c", 0, 90)]), 100
    )
    cost = app.member_cost(max_cosine_distance=0.1)
    assert cost(["a"], ["b"]) == pytest.approx(
        0.5 * (1 - np.cos(np.radians(10))), abs=1e-6
    )
    assert cost(["a"], ["x"]) is None
    assert cost(["a"], ["x", "b"]) is not None  # the track's closest masklet
    assert cost(["a"], ["never-embedded"]) is None


def test_pairs_are_labelled_by_lifting_and_by_sharing_an_image():
    app = Appearance(
        _embeddings(
            [
                ("front", "F", 0, 0),  # one car in two cameras
                ("left", "L", 0, 10),
                ("other", "F", 0, 60),  # another car beside it in F
                ("ped", "F", 0, 90),  # another class: not a pair
                ("unlifted", "F", 0, 5),  # no instance: not a pair
            ]
        ),
        100,
    )
    rows = [
        _lifted("i0", 0, ["front", "left"]),
        _lifted("i1", 0, ["other"]),
        _lifted("i2", 0, ["ped"], cls="pedestrian"),
    ]
    same, different = appearance.self_labelled_pairs(app, rows)
    assert same == [pytest.approx(app.distance("front", "left"))]
    # "left" is in another camera's image, so only front / other share one.
    assert different == [pytest.approx(app.distance("front", "other"))]


def test_threshold_is_where_both_kinds_are_wrong_equally_often():
    same = [0.02, 0.04, 0.06, 0.20]  # one same-object pair looks different
    different = [0.05, 0.30, 0.40, 0.50]  # one different pair looks alike
    threshold, rejected, accepted = appearance.equal_error_threshold(same, different)
    assert 0.05 < threshold <= 0.20
    assert rejected == accepted == 0.25


def test_calibration_needs_both_kinds_of_pair():
    app = Appearance(_embeddings([("a", "c", 0, 0), ("b", "d", 0, 10)]), 100)
    with pytest.raises(ValueError, match="1 same-object and 0 different-object"):
        appearance.calibrate([(app, [_lifted("i0", 0, ["a", "b"])])])


def _with_masklets(obs, masklet_id):
    import dataclasses

    return [
        dataclasses.replace(o, det=dataclasses.replace(o.det, masklets=(masklet_id,)))
        for o in obs
    ]


def test_relink_fuses_look_alike_tracks_whose_boxes_do_not_overlap():
    """The car comes back 1.5 m to the side (its other face): no overlap with
    its prediction, so only what it looks like joins the two pieces."""
    first = _with_masklets(_car(range(0, 10)), "before")
    second = _with_masklets(
        [_obs(k, 10.0 * 0.05 * k, cy=1.5, l=0.4) for k in range(16, 26)], "after"
    )
    alike = Appearance(
        _embeddings([("before", "c", 0, 0), ("after", "d", 0, 5)]), 100
    ).member_cost(0.1)
    unlike = Appearance(
        _embeddings([("before", "c", 0, 0), ("after", "d", 0, 60)]), 100
    ).member_cost(0.1)

    def run(member_cost):
        return relink([first, second], FRAMES, RelinkConfig(), 2.0, member_cost)

    assert _sizes(run(None)) == [10, 10]
    assert _sizes(run(unlike)) == [10, 10]
    assert _sizes(run(alike)) == [20]


def test_relink_never_fuses_look_alike_tracks_past_the_distance_gate():
    first = _with_masklets(_car(range(0, 10)), "before")
    second = _with_masklets(
        [_obs(k, 10.0 * 0.05 * k, cy=3.0) for k in range(16, 26)], "after"
    )
    alike = Appearance(
        _embeddings([("before", "c", 0, 0), ("after", "d", 0, 0)]), 100
    ).member_cost(0.1)
    assert _sizes(relink([first, second], FRAMES, RelinkConfig(), 2.0, alike)) == [
        10,
        10,
    ]


def test_enabled_without_descriptors_names_the_step_to_run(tmp_path, monkeypatch):
    from wato_tracking.pipeline import bag_appearance

    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    cfg = ComponentConfig()
    assert bag_appearance(cfg, "b", ["0000"]) == ({}, None)  # off by default
    cfg.appearance.enabled = True
    with pytest.raises(FileNotFoundError, match="embed"):
        bag_appearance(cfg, "b", ["0000"])


def test_a_fixed_threshold_skips_calibration(tmp_path, monkeypatch):
    from wato_tracking.pipeline import bag_appearance

    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    masklets.write_embeddings(
        "b", "0000", _embeddings([("a", "c", 0, 0), ("b", "c", 0, 10)])
    )
    cfg = ComponentConfig()
    cfg.appearance.enabled = True
    cfg.appearance.max_cosine_distance = 0.5
    costs, calibration = bag_appearance(cfg, "b", ["0000"])
    assert calibration is None
    assert costs["0000"](["a"], ["b"]) is not None
