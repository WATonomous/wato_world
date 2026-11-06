"""GPU regression checks for the official CUDA cleanup; no model weights needed."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_hole_filling_area_boundary():
    from wato_perception_2d.models._sam2_runtime import check_sam2_cuda_extension

    check_sam2_cuda_extension("cuda")
