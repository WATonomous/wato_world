"""Backend selection must not return a cached model with a different policy."""

import sys
import types

import pytest

from wato_perception_2d.models import _sam2_runtime as runtime


def test_backend_cache_separation_and_no_silent_fallback(monkeypatch):
    monkeypatch.setattr(runtime, "_cache", {})
    stock, dam = object(), object()
    sam_module = types.ModuleType("sam2.build_sam")
    sam_module.build_sam2_video_predictor = lambda *a, **kw: stock
    dam_module = types.ModuleType("wato_perception_2d.models.dam4sam")
    dam_module.build_dam4sam = lambda *a: dam
    monkeypatch.setitem(sys.modules, "sam2.build_sam", sam_module)
    monkeypatch.setitem(sys.modules, dam_module.__name__, dam_module)
    assert runtime.get_sam2_predictor("weights", device="cpu", backend="sam2") is stock
    assert runtime.get_sam2_predictor("weights", device="cpu", backend="dam4sam") is dam
    assert runtime.get_sam2_predictor("weights", device="cpu", backend="sam2") is stock

    def fail(*args):
        raise RuntimeError("bad checkpoint")

    dam_module.build_dam4sam = fail
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        runtime.get_sam2_predictor("missing", device="cpu", backend="dam4sam")
    with pytest.raises(ValueError, match="Unknown"):
        runtime.get_sam2_predictor("weights", backend="typo")


def _fake_sam2_misc(monkeypatch, fill_holes):
    misc = types.ModuleType("sam2.utils.misc")
    misc.fill_holes_in_mask_scores = fill_holes
    monkeypatch.setitem(sys.modules, misc.__name__, misc)


def test_hole_filling_probe_accepts_the_expected_fill(monkeypatch):
    pytest.importorskip("torch")

    def fill(scores, max_area):
        assert max_area == 8
        out = scores.clone()
        out[:, :, 2:4, 2:6] = 0.1  # the eight-pixel hole, not the nine
        return out

    _fake_sam2_misc(monkeypatch, fill)
    runtime._probe_hole_filling("cpu")


@pytest.mark.parametrize(
    "failure,error",
    [("missing", ImportError), ("unchanged", RuntimeError), ("warns", UserWarning)],
)
def test_hole_filling_probe_rejects_a_kernel_that_does_not_fill(
    monkeypatch, failure, error
):
    pytest.importorskip("torch")
    import warnings

    def fill(scores, max_area):
        if failure == "missing":
            raise ImportError("sam2._C is missing")
        if failure == "warns":  # SAM2's own fallback: warn, return unchanged
            warnings.warn("cannot import name '_C'")
        return scores

    _fake_sam2_misc(monkeypatch, fill)
    with pytest.raises(error):
        runtime._probe_hole_filling("cpu")


def test_cuda_check_wraps_a_failed_probe_and_skips_cpu(monkeypatch):
    def broken(device):
        raise ImportError("sam2._C is missing")

    monkeypatch.setattr(runtime, "_probe_hole_filling", broken)
    with pytest.raises(RuntimeError, match="Rebuild the perception_2d"):
        runtime.check_sam2_cuda_extension("cuda")
    runtime.check_sam2_cuda_extension("cpu")
