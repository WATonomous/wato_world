"""Tests for _sam3_runtime helpers that don't need the heavy GPU predictor.

_apply_long_video_memory_savers walks the predictor's module tree and flips the
per-frame VRAM offload / past-memory-trim switches wherever they exist — the
levers that keep long-video propagation from OOMing. We exercise it against
fakes so no `sam3` install / GPU is required.
"""

from __future__ import annotations

from wato_perception_2d.models._sam3_runtime import (
    _apply_inference_memory_knobs,
    _apply_long_video_memory_savers,
    _backfill_multistep_key,
)


class _Tracker:
    """Mimics VideoTrackingMultiplex: exposes the eval memory-saver flags."""

    def __init__(self, num_frames_to_correct_for_eval: int = 1) -> None:
        self.forward_backbone_per_frame_for_eval = False
        self.offload_output_to_cpu_for_eval = False
        self.trim_past_non_cond_mem_for_eval = False
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval


class _Plain:
    """A module without the flags (mimics the detector / encoder)."""


class _FakeModel:
    def __init__(self, mods: list) -> None:
        self._mods = mods

    def modules(self):
        return iter(self._mods)


class _FakePredictor:
    def __init__(self, model) -> None:
        self.model = model


def test_applies_all_savers_to_tracker_modules():
    trackers = [_Tracker(), _Tracker()]
    model = _FakeModel([_Plain(), trackers[0], _Plain(), trackers[1]])
    counts = _apply_long_video_memory_savers(
        _FakePredictor(model),
        offload_output_to_cpu=True,
        trim_past_non_cond_mem=True,
        forward_backbone_per_frame=True,
    )
    assert counts == {
        "forward_backbone_per_frame_for_eval": 2,
        "offload_output_to_cpu_for_eval": 2,
        "trim_past_non_cond_mem_for_eval": 2,
    }
    assert all(t.forward_backbone_per_frame_for_eval for t in trackers)
    assert all(t.offload_output_to_cpu_for_eval for t in trackers)
    assert all(t.trim_past_non_cond_mem_for_eval for t in trackers)


def test_trim_skipped_when_multi_frame_correction():
    """Upstream forbids trimming past memory when >1 frame is corrected; the
    helper must honour that invariant and leave trim off (the others still
    apply)."""
    tracker = _Tracker(num_frames_to_correct_for_eval=3)
    counts = _apply_long_video_memory_savers(
        _FakePredictor(_FakeModel([tracker])),
        offload_output_to_cpu=True,
        trim_past_non_cond_mem=True,
        forward_backbone_per_frame=True,
    )
    assert counts["trim_past_non_cond_mem_for_eval"] == 0
    assert tracker.trim_past_non_cond_mem_for_eval is False
    assert counts["offload_output_to_cpu_for_eval"] == 1
    assert tracker.offload_output_to_cpu_for_eval is True
    assert counts["forward_backbone_per_frame_for_eval"] == 1
    assert tracker.forward_backbone_per_frame_for_eval is True


def test_flags_respected_when_disabled():
    tracker = _Tracker()
    counts = _apply_long_video_memory_savers(
        _FakePredictor(_FakeModel([tracker])),
        offload_output_to_cpu=False,
        trim_past_non_cond_mem=False,
        forward_backbone_per_frame=False,
    )
    assert counts == {
        "forward_backbone_per_frame_for_eval": 0,
        "offload_output_to_cpu_for_eval": 0,
        "trim_past_non_cond_mem_for_eval": 0,
    }
    assert tracker.forward_backbone_per_frame_for_eval is False
    assert tracker.offload_output_to_cpu_for_eval is False
    assert tracker.trim_past_non_cond_mem_for_eval is False


def test_backfill_multistep_key_injects_only_missing():
    """The trim-KeyError patch must add 'multistep_point_inputs'=None to stored
    frame outputs that lack it, leave existing values untouched, and only walk
    the cond/non-cond buckets."""
    out_dict = {
        "cond_frame_outputs": {
            0: {"pred_masks": "m0"},  # missing -> injected
        },
        "non_cond_frame_outputs": {
            5: {"pred_masks": "m5"},  # missing -> injected
            6: {"multistep_point_inputs": ["kept"]},  # present -> untouched
        },
        "something_else": {7: {"pred_masks": "m7"}},  # ignored bucket
    }
    touched = _backfill_multistep_key(out_dict)
    assert touched == 2
    assert out_dict["cond_frame_outputs"][0]["multistep_point_inputs"] is None
    assert out_dict["non_cond_frame_outputs"][5]["multistep_point_inputs"] is None
    assert out_dict["non_cond_frame_outputs"][6]["multistep_point_inputs"] == ["kept"]
    assert "multistep_point_inputs" not in out_dict["something_else"][7]


def test_backfill_multistep_key_tolerates_non_dict():
    assert _backfill_multistep_key(None) == 0
    assert _backfill_multistep_key({"cond_frame_outputs": None}) == 0


class _KnobModel:
    """Mimics the assembled multiplex model: build-time memory knobs as attrs."""

    def __init__(self) -> None:
        self.image_size = 1008
        self.postprocess_batch_size = 16
        self.batched_grounding_batch_size = 16


def test_inference_memory_knobs_override_build_defaults():
    model = _KnobModel()
    _apply_inference_memory_knobs(
        _FakePredictor(model),
        image_size=768,
        postprocess_batch_size=1,
        batched_grounding_batch_size=1,
    )
    assert model.image_size == 768
    assert model.postprocess_batch_size == 1
    assert model.batched_grounding_batch_size == 1


def test_inference_memory_knobs_none_leaves_build_default():
    model = _KnobModel()
    _apply_inference_memory_knobs(
        _FakePredictor(model),
        image_size=None,  # keep 1008
        postprocess_batch_size=1,
        batched_grounding_batch_size=None,  # keep 16
    )
    assert model.image_size == 1008  # untouched
    assert model.postprocess_batch_size == 1  # overridden
    assert model.batched_grounding_batch_size == 16  # untouched


def test_inference_memory_knobs_missing_attr_is_noop():
    """A sam3 rev that renamed an attr must be a logged no-op, not an
    AttributeError that takes down the whole predictor."""

    class _Partial:
        batched_grounding_batch_size = 16  # only this one exists

    model = _Partial()
    # image_size / postprocess_batch_size absent → skipped, no crash, no attr created.
    _apply_inference_memory_knobs(
        _FakePredictor(model),
        image_size=768,
        postprocess_batch_size=1,
        batched_grounding_batch_size=2,
    )
    assert model.batched_grounding_batch_size == 2
    assert not hasattr(model, "image_size")
    assert not hasattr(model, "postprocess_batch_size")


def test_inference_memory_knobs_tolerates_no_model():
    class _NoModel:
        model = None

    # Must not raise.
    _apply_inference_memory_knobs(
        _NoModel(),
        image_size=768,
        postprocess_batch_size=1,
        batched_grounding_batch_size=1,
    )


def test_tolerates_model_without_modules():
    class _NoModules:
        model = object()  # no .modules()

    counts = _apply_long_video_memory_savers(
        _NoModules(),
        offload_output_to_cpu=True,
        trim_past_non_cond_mem=True,
        forward_backbone_per_frame=True,
    )
    assert counts == {
        "forward_backbone_per_frame_for_eval": 0,
        "offload_output_to_cpu_for_eval": 0,
        "trim_past_non_cond_mem_for_eval": 0,
    }
