"""Exercise the actual author memory policy through the pipeline adapter.

Only the neural network is faked. These run in the perception image, with its
SAM2/OpenCV dependencies, without a GPU or checkpoint.
"""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("sam2")
pytest.importorskip("cv2")

from wato_perception_2d.models.dam4sam import DAM4SAMPredictor  # noqa: E402


class _Model:
    device = torch.device("cpu")
    image_size = 16
    num_maskmem = 7
    max_obj_ptrs_in_encoder = 16

    def __init__(self):
        self.calls = []
        self.score = 0.9

    def track_step(self, **kw):
        self.calls.append((kw["frame_idx"], kw["is_init_cond_frame"]))
        # score: the chosen mask's predicted IoU, or frame_idx → that IoU.
        score = self.score(kw["frame_idx"]) if callable(self.score) else self.score
        chosen = torch.full((1, 1, 16, 16), -10.0)
        chosen[:, :, 2:6, 2:6] = 10
        alternatives = chosen.repeat(1, 3, 1, 1)
        alternatives[:, 1, 10:14, 10:14] = 10
        return dict(
            pred_masks=chosen,
            multimasks_logits=alternatives,
            ious=torch.tensor([[score, min(score, 0.5), min(score, 0.4)]]),
            obj_ptr=torch.zeros(1, 4),
            object_score_logits=torch.ones(1, 1),
            maskmem_features=torch.zeros(1, 4, 2, 2),
            maskmem_pos_enc=None,
        )

    def _encode_new_memory(self, **kw):
        return torch.zeros(1, 4, 2, 2), [torch.zeros(1, 4, 2, 2)]


@pytest.fixture
def make_tracker(tmp_path, monkeypatch):
    """(predictor, state, model, feature calls) for an adapter memory policy;
    no arguments: the authors' (update_delta 5, no RAM gate)."""
    # CUDA extension is unrelated to these memory/lifecycle tests.
    monkeypatch.setattr(
        "wato_perception_2d._vendor.d4sm.tracking_wrapper_mot.fill_holes_in_mask_scores",
        lambda masks, area: masks,
    )
    for frame in range(45):
        Image.new("RGB", (16, 16)).save(tmp_path / f"{frame:06d}.jpg")

    def make(**policy):
        model = _Model()
        predictor = DAM4SAMPredictor(model, **policy)
        feature_calls = []

        def features(image):
            feature_calls.append(1)
            return [torch.zeros(1)], [], []

        monkeypatch.setattr(predictor._preprocessor, "_get_features", features)
        return predictor, predictor.init_state(tmp_path), model, feature_calls

    return make


@pytest.fixture
def tracker(make_tracker):
    return make_tracker()


def _seed(p, s, frame, oid):
    return p.add_new_points_or_box(s, frame, oid, np.array([2, 2, 6, 6]))


def test_late_birth_correction_and_keyframe_revisit(tracker):
    p, s, model, features = tracker
    _seed(p, s, 0, 10)
    list(p.propagate_in_video(s, 0, 2))
    original = s["objects"][10].context
    _seed(p, s, 2, 30)
    calls = len(model.calls)
    [(frame, ids, logits)] = list(p.propagate_in_video(s, 2, 0))
    assert frame == 2 and ids == [10, 30] and logits.shape == (2, 1, 16, 16)
    assert len(model.calls) == calls  # lookahead keyframe cannot update twice
    list(p.propagate_in_video(s, 2, 1))
    assert original.frame_index == 3
    assert s["objects"][30].context.frame_index == 1
    other = s["objects"][30].context
    _seed(p, s, 3, 10)
    assert s["objects"][10].context is not original
    assert s["objects"][10].context.frame_index == 0
    assert s["objects"][30].context is other and other.frame_index == 1
    assert len(features) == 4  # one backbone pass per image across all objects
    p.remove_object(s, 10)
    list(p.propagate_in_video(s, 3, 1))
    assert list(s["objects"]) == [30] and other.frame_index == 2
    p.reset_state(s)
    assert not s["objects"] and not s["shared"]
    _seed(p, s, 7, 90)
    assert s["objects"][90].context.frame_index == 0


def test_author_memory_selection_and_bounds(tracker):
    p, s, model, _ = tracker
    _seed(p, s, 0, 1)
    list(p.propagate_in_video(s, 0, 40))
    context = s["objects"][1].context
    memory = context.per_object_outputs_all[1]
    assert memory[0]["is_init"] and memory[0]["frame_idx"] == 0
    assert len(memory) <= 7
    drm = [m["frame_idx"] for m in memory if m["is_drm"]]
    # Promotions are delayed one frame; exact upstream >5-frame interval.
    assert drm == [26, 32, 38]
    assert memory[-1]["frame_idx"] == 40
    assert len(context.per_object_obj_ptr[1]) == 16
    assert context.per_object_obj_ptr[1][0]["is_init"]


@pytest.mark.parametrize("score,has_drm", [(0.8, False), (0.8001, True)])
def test_author_strict_quality_gate(tracker, score, has_drm):
    p, s, model, _ = tracker
    model.score = score
    _seed(p, s, 0, 1)
    list(p.propagate_in_video(s, 0, 8))
    memory = s["objects"][1].context.per_object_outputs_all[1]
    assert any(m["is_drm"] for m in memory) == has_drm


def test_mask_prompt_and_invalid_time_direction(tracker):
    p, s, _, _ = tracker
    p.add_new_mask(s, 4, 9, np.ones((16, 16), dtype=bool))
    list(p.propagate_in_video(s, 4, 1))
    with pytest.raises(ValueError, match="in order"):
        list(p.propagate_in_video(s, 4, 1))
    with pytest.raises(ValueError, match="forward"):
        list(p.propagate_in_video(s, 5, 1, reverse=True))


def test_bfloat16_iou_export(tracker, monkeypatch):
    p, s, model, _ = tracker
    step = model.track_step

    def bfloat_step(**kwargs):
        result = step(**kwargs)
        result["ious"] = result["ious"].to(torch.bfloat16)
        return result

    monkeypatch.setattr(model, "track_step", bfloat_step)
    _seed(p, s, 0, 1)
    list(p.propagate_in_video(s, 0, 8))
    assert p.memory_diagnostics(s, 1)["drm_frames"]


def _ram(p, s, oid=1):
    return p.memory_diagnostics(s, oid)["ram_frames"]


@pytest.mark.parametrize("delta,ram", [(5, [5, 10]), (2, [2, 4, 6, 8, 10])])
def test_update_delta_sets_the_ram_spacing(make_tracker, delta, ram):
    """Δ is the spacing of kept recent-appearance frames (plus the last one).
    Predicted IoU 0.6 stays below the distractor memory's 0.8 gate."""
    p, s, model, _ = make_tracker(update_delta=delta)
    model.score = 0.6
    _seed(p, s, 0, 1)
    list(p.propagate_in_video(s, 0, 10))
    assert _ram(p, s) == ram


@pytest.mark.parametrize("gate,ram", [(0.0, [2, 4, 6, 8, 10]), (0.5, [2, 6, 10])])
def test_samurai_gate_keeps_low_confidence_frames_out_of_ram(make_tracker, gate, ram):
    """Frames 4 and 8 would be kept RAM slots; with SAMURAI's gate their
    predicted IoU (0.3) leaves them only as the next image's last frame."""
    p, s, model, _ = make_tracker(update_delta=2, memory_min_predicted_iou=gate)
    model.score = lambda frame: 0.3 if frame in (4, 8) else 0.6
    _seed(p, s, 0, 1)
    seen = []
    for frame, _, _ in p.propagate_in_video(s, 0, 10):
        seen.append((frame, _ram(p, s)))
    assert _ram(p, s) == ram
    # The newest frame is always in memory, gated or not; a gated frame
    # serves the next image only (8 is gone once 9 is in).
    assert all(kept[-1] == frame for frame, kept in seen[1:])
    assert (8 in dict(seen)[9]) is (gate == 0)
    assert p.memory_diagnostics(s, 1)["gated_frames"] == (2 if gate else 0)
    assert p.frame_quality(s, 1) == pytest.approx(
        {"predicted_iou": 0.6, "object_score_logit": 1.0}
    )


def test_samurai_gate_leaves_object_pointers_and_drm_alone(make_tracker):
    p, s, model, _ = make_tracker(update_delta=2, memory_min_predicted_iou=0.5)
    model.score = 0.3
    _seed(p, s, 0, 1)
    list(p.propagate_in_video(s, 0, 20))
    assert _ram(p, s) == [20]  # only the newest frame
    assert len(s["objects"][1].context.per_object_obj_ptr[1]) == 16
