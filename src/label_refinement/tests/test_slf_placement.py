"""The poses SLF evaluates must be the poses written as refined boxes."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wato_common.artifacts.calibration import Calibration, CameraCalibration  # noqa: E402
from wato_common.artifacts.frame_index import CameraFrame, FrameIndex  # noqa: E402
from wato_common.artifacts.masklets import MaskletIndex, MaskletRecord, write_mask  # noqa: E402
from wato_common.artifacts.tracks import TrackFrame, TrackSummary  # noqa: E402
from wato_common.artifacts.refined_labels import PERCENTILE  # noqa: E402
from wato_label_refinement import pipeline, slf, views  # noqa: E402
from wato_label_refinement.config import ComponentConfig  # noqa: E402


@pytest.mark.parametrize("selected", [tuple(range(7)), (0, 3, 6)])
def test_accepted_slf_positions_match_evaluated_positions_for_the_whole_track(
    tmp_path, monkeypatch, selected
):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", tmp_path.as_uri())
    cfg = ComponentConfig()
    cfg.moving.heading_min_speed_mps = 100  # Isolate placement from heading changes.
    # Accept the unchanged initialization to exercise the geometry handoff,
    # independently of optimizer convergence or mask quality.
    cfg.slf.iterations = 0
    cfg.slf.min_silhouette_gain = 0
    cfg.slf.max_residual_silhouette = 1
    cfg.slf.crop_px = 12
    frames = [
        TrackFrame(
            "t",
            "0000",
            k,
            k * 100_000_000,
            f"p{k}",
            10 + 0.2 * k + (0.3 if k % 2 else -0.3),
            0,
            0.75,
            1.8,
            4.5,
            1.5,
            0,
            "car",
            1,
            True,
        )
        for k in range(7)
    ]
    summary = TrackSummary("t", 7, 0, 600_000_000, "moving", "car", 7, 0, 0, 1, 0.1, 30)
    counts = {("t", "0000", k): 30 for k in range(7)}
    ego = np.zeros((7, 2))
    grounds = {"0000": None}
    refined = pipeline.moving(cfg, frames, summary, counts, ego, grounds)
    phase3 = pipeline.Phase3(
        refined.boxes,
        refined.heading_directed,
        refined.residual_smoothness,
        PERCENTILE,
        refined.velocity,
        ego,
    )
    K = np.array([[80.0, 0, 80], [0, 80, 60], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    transform[2, 3] = 1.5
    camera = CameraCalibration("CAM", K, transform, width=160, height=120)
    record = MaskletRecord("m", "CAM", "car", 1, tuple(range(7)))
    images = [
        CameraFrame("CAM", k, f.reference_timestamp_ns, "unused", (k,), np.eye(4))
        for k, f in enumerate(frames)
    ]
    mask = np.zeros((120, 160), bool)
    mask[40:80, 50:110] = True
    for k in selected:
        write_mask("bag", "0000", "m", k, mask)
    index = MaskletIndex("bag", "0000", [record])
    chunk = views.Chunk(FrameIndex(images), index, {"m": record}, {})
    selected_views = [
        views.View(k, images[k], (record,), transform, 100) for k in selected
    ]
    problem = views.problem(
        "t",
        "car",
        frames,
        phase3.boxes,
        phase3.velocity,
        ego,
        True,
        selected_views,
        {},
        {"0000": chunk},
        Calibration({"CAM": camera}, {}),
        cfg.slf,
        "free",
    )
    # Frames without selected camera views still receive fitted output boxes.
    assert len(problem.heading) == len(frames)
    fit = slf.fit(problem, cfg.slf, torch.device("cpu"))
    model = slf._Batch([problem], cfg.slf, torch.device("cpu"))
    yaw = model.yaw_param(torch.device("cpu")).detach()
    evaluated = model.centres(
        torch.tensor(fit.size[None], dtype=torch.float32),
        torch.tensor(fit.offset[None], dtype=torch.float32),
        yaw,
    ).numpy()
    expected_initial = np.array([[b.cx, b.cy, b.cz] for b in phase3.boxes])
    np.testing.assert_allclose(evaluated, expected_initial, atol=2e-6)
    np.testing.assert_allclose(model.headings(yaw), fit.headings, atol=1e-12)
    np.testing.assert_allclose(
        fit.headings, [b.heading for b in phase3.boxes], atol=1e-6
    )
    output, outcome = pipeline.apply_slf(cfg, frames, summary, phase3, fit)
    assert outcome.accepted
    actual = np.array([[b.cx, b.cy, b.cz] for b in output.boxes])
    np.testing.assert_allclose(actual, evaluated, atol=2e-6)
    np.testing.assert_allclose(
        [b.heading for b in output.boxes], fit.headings, atol=1e-6
    )


def test_accepted_fit_writes_its_corrected_headings():
    """A directed heading keeps its front; an undirected one is wrapped to
    (-pi/2, pi/2] as Phase 3 writes it."""
    from wato_common.schemas import Box3D

    cfg = ComponentConfig()
    cfg.slf.min_silhouette_gain = 0
    frames = [
        TrackFrame(
            "t",
            "0000",
            k,
            k * 100_000_000,
            f"p{k}",
            0,
            0,
            0.75,
            1.8,
            4.5,
            1.5,
            0,
            "car",
            1,
            True,
        )
        for k in range(2)
    ]
    summary = TrackSummary("t", 2, 0, 100_000_000, "moving", "car", 2, 0, 0, 1, 0.1, 30)
    box = Box3D(cx=0, cy=0, cz=0.75, w=1.8, l=4.5, h=1.5, heading=3.1)
    phase3 = pipeline.Phase3(
        [box, box], [True, False], None, PERCENTILE, np.zeros((2, 2)), np.zeros((2, 2))
    )
    fit = slf.Fit(
        size=np.array([4.6, 1.9, 1.5]),
        offset=np.zeros(2),
        centres=np.array([[0.0, 0.0, 0.75], [0.0, 0.0, 0.75]]),
        headings=np.array([3.1 + 0.1, 3.1 + 0.1]),
        residual_silhouette=0.1,
        residual_lidar=0.01,
        start_residual_silhouette=0.3,
        start_residual_lidar=0.02,
    )
    output, outcome = pipeline.apply_slf(cfg, frames, summary, phase3, fit)
    assert outcome.accepted
    assert output.boxes[0].heading == pytest.approx(3.2 - 2 * np.pi)  # front kept
    assert output.boxes[1].heading == pytest.approx(3.2 - np.pi)  # mod pi
    bad = slf.Fit(**{**fit.__dict__, "headings": np.array([0.0])})
    with pytest.raises(ValueError, match="heading"):
        pipeline.apply_slf(cfg, frames, summary, phase3, bad)
