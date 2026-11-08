"""Prompt-policy and public SAM2 API contracts without model weights."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from wato_perception_2d.config import MaskRefinementConfig
from wato_perception_2d.models.mask_refinement import (
    KeyframeMaskRefiner,
    interior_point,
    stability_scores,
)


class ImagePredictor:
    def __init__(self, model):
        self.model = model
        self.outputs = []
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def make_refiner(monkeypatch, **kwargs):
    package = ModuleType("sam2")
    module = ModuleType("sam2.sam2_image_predictor")
    module.SAM2ImagePredictor = ImagePredictor
    monkeypatch.setitem(sys.modules, "sam2", package)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", module)
    model = object()
    refiner = KeyframeMaskRefiner(model, MaskRefinementConfig(**kwargs))
    assert refiner.predictor.model is model  # weights shared, no second builder
    return refiner


def output(scores):
    logits = np.full((len(scores), 12, 16), -4, dtype=np.float32)
    logits[:, 2:10, 3:13] = 4
    lowres = np.stack(
        [np.full((256, 256), i + 2, dtype=np.float32) for i in range(len(scores))]
    )
    return logits, np.array(scores), lowres


def test_selected_low_resolution_logits_feed_the_refinement(monkeypatch):
    refiner = make_refiner(monkeypatch, multimask_output=True)
    refiner.predictor.outputs = [output([0.85, 0.9]), output([0.95])]
    result = refiner.propose((3, 2, 13, 10))
    assert result.mask.shape == (12, 16)
    assert result.used_interior_point
    assert result.predicted_iou == 0.95
    second = refiner.predictor.calls[1]
    assert second["mask_input"].shape == (1, 256, 256)
    assert np.all(second["mask_input"] == 3)  # candidate 1, not candidate 0
    assert second["multimask_output"] is False
    assert second["return_logits"] is True
    assert second["point_labels"].tolist() == [1]
    x, y = second["point_coords"][0].astype(int)
    assert result.mask[y, x]


def test_refinement_that_lowers_quality_keeps_original_mask(monkeypatch):
    refiner = make_refiner(monkeypatch)
    refiner.predictor.outputs = [output([0.9]), output([0.85])]
    result = refiner.propose((3, 2, 13, 10))
    assert result.predicted_iou == 0.9
    assert not result.used_interior_point


@pytest.mark.parametrize(
    "point,recurrence", [(False, False), (True, False), (False, True)]
)
def test_prompt_and_logit_refinement_can_be_ablated_separately(
    monkeypatch, point, recurrence
):
    refiner = make_refiner(
        monkeypatch, interior_point=point, mask_input_refinement=recurrence
    )
    refiner.predictor.outputs = [output([0.9]), output([0.95])]
    result = refiner.propose((3, 2, 13, 10))
    assert len(refiner.predictor.calls) == (2 if point or recurrence else 1)
    if point or recurrence:
        second = refiner.predictor.calls[1]
        assert (second["point_coords"] is not None) is point
        assert (second["mask_input"] is not None) is recurrence
    assert result.mask is not None


@pytest.mark.parametrize("failure", ["empty", "unstable", "low_quality", "nan"])
def test_untrusted_candidates_do_not_become_mask_seeds(monkeypatch, failure):
    refiner = make_refiner(monkeypatch)
    logits, scores, lowres = output([0.9])
    if failure == "empty":
        logits[:] = -4
    elif failure == "unstable":
        logits[logits > 0] = 0.1
    elif failure == "low_quality":
        scores[:] = 0.1
    else:
        logits[0, 0, 0] = np.nan
    refiner.predictor.outputs = [(logits, scores, lowres)]
    result = refiner.propose((3, 2, 13, 10))
    assert result.mask is None
    assert len(refiner.predictor.calls) == 1


def test_ego_overlap_is_not_promoted_to_a_trusted_mask(monkeypatch):
    refiner = make_refiner(monkeypatch)
    refiner.predictor.outputs = [output([0.9])]
    ego = np.zeros((12, 16), dtype=bool)
    ego[8:, :] = True
    assert refiner.propose((3, 2, 13, 10), ego).mask is None


def test_distance_point_avoids_hollow_box_center_and_stays_inside_prompt():
    mask = np.zeros((20, 30), dtype=bool)
    mask[:10, :10] = True  # touches two image boundaries
    mask[3:7, 3:7] = False  # bbox center is background
    mask[:, 15:] = True  # larger leaked region, outside the prompt
    point = interior_point(mask, np.array([0, 0, 10, 10]))
    x, y = point[0].astype(int)
    assert 0 <= x < 10 and 0 <= y < 10 and mask[y, x]
    assert interior_point(mask, np.array([11, 0, 14, 20])) is None


def test_stability_is_threshold_iou_with_empty_masks_scored_zero():
    logits = np.array([[[2, 0.5], [-0.5, -2]], [[-4, -4], [-4, -4]]])
    np.testing.assert_allclose(stability_scores(logits), [1 / 3, 0])


def test_stability_reduces_on_tensor_device():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[2, 0.5], [-0.5, -2]]])
    np.testing.assert_allclose(stability_scores(logits), [1 / 3])
