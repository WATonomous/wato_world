"""Deployment export keeps learned tensors and specializes the camera axis."""

from pathlib import Path

import pytest


def test_pinned_export_changes_only_camera_shapes(tmp_path):
    onnx = pytest.importorskip("onnx")
    from wato_detection_3d.export_cameras import RESHAPES, export_backbone

    source = (
        Path(__file__).resolve().parents[3]
        / "data/models/bevfusion/resnet50_trt11_int8/camera.backbone.onnx"
    )
    if not source.exists():
        pytest.skip("pinned deployment model is not installed")
    target = tmp_path / "eight.onnx"
    export_backbone(source, target, 8)
    before, after = onnx.load(source), onnx.load(target)
    before_tensors = {v.name: v.SerializeToString() for v in before.graph.initializer}
    after_tensors = {v.name: v.SerializeToString() for v in after.graph.initializer}
    assert before_tensors.keys() == after_tensors.keys()
    assert {
        key for key in before_tensors if before_tensors[key] != after_tensors[key]
    } == set(RESHAPES)
    assert [v.SerializeToString() for v in before.graph.node] == [
        v.SerializeToString() for v in after.graph.node
    ]
    assert all(
        v.type.tensor_type.shape.dim[1].dim_value == 8 for v in after.graph.input
    )
    assert all(
        v.type.tensor_type.shape.dim[0].dim_value == 8 for v in after.graph.output
    )
