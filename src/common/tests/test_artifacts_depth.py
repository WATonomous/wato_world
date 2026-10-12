"""wato_common.artifacts.depth: what perception_2d writes, consumers read back."""

from __future__ import annotations

import numpy as np
import pytest

from wato_common.artifact_store import depth_2d_path, local_path
from wato_common.artifacts import ArtifactFormatError, depth

BAG, CHUNK, CAM = "b", "0000", "CAM_FRONT"


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _frame(status=depth.FitStatus.OK) -> depth.DepthFrame:
    return depth.DepthFrame(
        depth_m=np.linspace(1, 80, 12, dtype=np.float32)
        .reshape(3, 4)
        .astype(np.float16),
        disp_scale=1.5,
        disp_shift=-0.25,
        n_anchors=40,
        n_inliers=31,
        rmse_inliers_m=0.5,
        fit_status=status,
    )


def test_round_trip():
    written = _frame(depth.FitStatus.FALLBACK)
    depth.write_frame(BAG, CHUNK, CAM, 7, written)
    assert depth.exists(BAG, CHUNK, CAM, 7)
    got = depth.read_frame(BAG, CHUNK, CAM, 7)
    assert got.depth_m.dtype == np.float32
    np.testing.assert_array_equal(got.depth_m, written.depth_m.astype(np.float32))
    assert (got.disp_scale, got.disp_shift) == (1.5, -0.25)
    assert (got.n_anchors, got.n_inliers, got.rmse_inliers_m) == (40, 31, 0.5)
    assert got.fit_status is depth.FitStatus.FALLBACK


def test_a_failed_fit_is_never_written_and_reads_as_absent():
    with pytest.raises(ValueError, match="not written"):
        depth.write_frame(BAG, CHUNK, CAM, 8, _frame(depth.FitStatus.FAILED))
    assert not depth.exists(BAG, CHUNK, CAM, 8)
    assert depth.read_frame(BAG, CHUNK, CAM, 8) is None


def test_missing_diagnostics_are_rejected():
    path = local_path(depth_2d_path(BAG, CHUNK, CAM, 9))
    depth.write_frame(BAG, CHUNK, CAM, 9, _frame())  # creates the directory
    np.savez_compressed(path, depth_m=np.ones((2, 2), np.float16))
    with pytest.raises(ArtifactFormatError, match="fit_status"):
        depth.read_frame(BAG, CHUNK, CAM, 9)


def test_pre_disparity_fit_files_are_rejected():
    """Files from the old depth-affine fit (affine_a/b) hold wrongly-scaled
    depth; they must fail loudly rather than be read as metric."""
    path = local_path(depth_2d_path(BAG, CHUNK, CAM, 10))
    depth.write_frame(BAG, CHUNK, CAM, 10, _frame())
    np.savez_compressed(
        path,
        depth_m=np.ones((2, 2), np.float16),
        affine_a=np.float32(1.0),
        affine_b=np.float32(0.0),
        n_anchors=np.int32(1),
        n_inliers=np.int32(1),
        rmse_inliers_m=np.float32(0.1),
        fit_status=np.int32(0),
    )
    with pytest.raises(ArtifactFormatError, match="disp_scale"):
        depth.read_frame(BAG, CHUNK, CAM, 10)
