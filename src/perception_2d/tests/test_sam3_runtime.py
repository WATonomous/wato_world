"""Tests for _sam3_runtime helpers that don't need the heavy GPU predictor.

_apply_long_video_memory_savers walks the predictor's module tree and flips the
per-frame VRAM offload / past-memory-trim switches wherever they exist — the
levers that keep long-video propagation from OOMing. We exercise it against
fakes so no `sam3` install / GPU is required.
"""

from __future__ import annotations

import sys
import types

import numpy as np

from wato_perception_2d.models import _sam3_runtime
from wato_perception_2d.models._sam3_runtime import (
    _NO_BUCKETS,
    _apply_inference_memory_knobs,
    _apply_long_video_memory_savers,
    _backfill_multistep_key,
    _patch_remove_objects_keyerror,
    _pop_trimmed_noncond,
    _restore_trimmed_noncond,
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


# ---------------------------------------------------------------------------
# Object-removal KeyError patch (trim-stub quarantine)
# ---------------------------------------------------------------------------


def _full_frame():
    """A full (untrimmed) non-cond output: carries packed maskmem + obj_ptr."""
    return {
        "maskmem_features": np.arange(2 * 3).reshape(2, 3).astype(np.float32),
        "obj_ptr": np.arange(2 * 4).reshape(2, 4).astype(np.float32),
    }


def _trim_stub():
    """A pointer-only trim stub: obj_ptr kept, maskmem dropped (the crash trigger)."""
    return {"obj_ptr": np.arange(2 * 4).reshape(2, 4).astype(np.float32)}


def test_pop_trimmed_noncond_only_lifts_stubs():
    od = {
        "cond_frame_outputs": {0: _full_frame()},
        "non_cond_frame_outputs": {3: _full_frame(), 7: _trim_stub(), 9: _trim_stub()},
    }
    popped = _pop_trimmed_noncond(od)
    assert set(popped) == {7, 9}
    # Full frames (and all cond frames) stay put.
    assert set(od["non_cond_frame_outputs"]) == {3}
    assert set(od["cond_frame_outputs"]) == {0}


def test_pop_trimmed_noncond_tolerates_bad_shape():
    assert _pop_trimmed_noncond(None) == {}
    assert _pop_trimmed_noncond({"non_cond_frame_outputs": None}) == {}


def test_restore_slices_obj_ptr_when_buckets_given():
    od = {"non_cond_frame_outputs": {}}
    popped = {7: _trim_stub()}
    n = _restore_trimmed_noncond(od, popped, [0])  # keep only bucket 0
    assert n == 1
    restored = od["non_cond_frame_outputs"][7]
    assert restored["obj_ptr"].shape == (1, 4)  # bucket dim sliced 2 -> 1
    assert "maskmem_features" not in restored  # still a trim stub


def test_restore_no_buckets_leaves_obj_ptr_untouched():
    """Early-return removal (no bucket call) → stub restored verbatim."""
    od = {"non_cond_frame_outputs": {}}
    popped = {7: _trim_stub()}
    n = _restore_trimmed_noncond(od, popped, _NO_BUCKETS)
    assert n == 0
    assert od["non_cond_frame_outputs"][7]["obj_ptr"].shape == (2, 4)  # unsliced


def test_restore_tolerates_stub_without_obj_ptr():
    od = {"non_cond_frame_outputs": {}}
    popped = {7: {"pred_masks": "x"}}  # no obj_ptr
    n = _restore_trimmed_noncond(od, popped, [0])
    assert n == 0
    assert od["non_cond_frame_outputs"][7] == {"pred_masks": "x"}


def _install_fake_sam3(monkeypatch):
    """Inject minimal fake sam3 multiplex modules whose remove_objects reproduces
    the upstream _slice_state KeyError on maskmem-less frames."""

    class MultiplexState:
        def remove_objects(self, object_indices, strict=True):
            return [0]  # keep bucket 0 only (the common, identity-ish case)

    class VideoTrackingMultiplexDemo:
        def remove_objects(
            self, inference_state, obj_ids, strict=False, need_output=False
        ):
            od = inference_state["output_dict"]
            remaining = inference_state["obj_ids"]
            if not remaining:  # mirrors upstream's len(new_obj_ids)==0 early return
                return remaining, []
            btk = inference_state["multiplex_state"].remove_objects([0], strict=True)

            def _slice_state(storage_key):
                for _fi, out in od[storage_key].items():
                    # KeyErrors on a trim stub exactly like the real _slice_state.
                    out["maskmem_features"] = out["maskmem_features"][btk]
                    out["obj_ptr"] = out["obj_ptr"][btk]

            _slice_state("cond_frame_outputs")
            _slice_state("non_cond_frame_outputs")
            return remaining, []

    mu = types.ModuleType("sam3.model.multiplex_utils")
    mu.MultiplexState = MultiplexState
    demo = types.ModuleType("sam3.model.video_tracking_multiplex_demo")
    demo.VideoTrackingMultiplexDemo = VideoTrackingMultiplexDemo
    pkg = types.ModuleType("sam3")
    model = types.ModuleType("sam3.model")
    pkg.model = model
    model.multiplex_utils = mu
    model.video_tracking_multiplex_demo = demo
    for name, mod in [
        ("sam3", pkg),
        ("sam3.model", model),
        ("sam3.model.multiplex_utils", mu),
        ("sam3.model.video_tracking_multiplex_demo", demo),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return MultiplexState, VideoTrackingMultiplexDemo


def _inference_state(MultiplexState, *, remaining):
    return {
        "obj_ids": list(remaining),
        "multiplex_state": MultiplexState(),
        "output_dict": {
            "cond_frame_outputs": {0: _full_frame()},
            "non_cond_frame_outputs": {3: _full_frame(), 7: _trim_stub()},
        },
    }


def test_patch_lets_removal_survive_trim_stub(monkeypatch):
    MultiplexState, Demo = _install_fake_sam3(monkeypatch)
    assert _patch_remove_objects_keyerror() is True

    state = _inference_state(MultiplexState, remaining=["a"])
    # Without the patch this raises KeyError('maskmem_features'); with it, no raise.
    Demo().remove_objects(state, ["b"])

    noncond = state["output_dict"]["non_cond_frame_outputs"]
    # Full frame got sliced to the surviving bucket.
    assert noncond[3]["maskmem_features"].shape == (1, 3)
    # Stub restored, still maskmem-less, obj_ptr sliced to match the kept bucket.
    assert "maskmem_features" not in noncond[7]
    assert noncond[7]["obj_ptr"].shape == (1, 4)


def test_patch_early_return_leaves_stub_unsliced(monkeypatch):
    MultiplexState, Demo = _install_fake_sam3(monkeypatch)
    assert _patch_remove_objects_keyerror() is True

    state = _inference_state(MultiplexState, remaining=[])  # all objects gone
    Demo().remove_objects(state, ["b"])

    noncond = state["output_dict"]["non_cond_frame_outputs"]
    # No bucket removal happened, so the stub comes back exactly as it went in.
    assert "maskmem_features" not in noncond[7]
    assert noncond[7]["obj_ptr"].shape == (2, 4)


def test_patch_is_idempotent(monkeypatch):
    _install_fake_sam3(monkeypatch)
    assert _patch_remove_objects_keyerror() is True
    assert _patch_remove_objects_keyerror() is True  # second call: no double-wrap
