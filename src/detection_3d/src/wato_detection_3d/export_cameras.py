"""Specialize the pinned per-camera ONNX backbone for a different rig size.

The camera encoder shares weights across views. Only input/output shapes and
five Reshape targets contain the exported six-view count. BEV pooling happens
in CUDA, outside this graph; downstream graphs are unchanged. Install
onnx==1.17.0 to run this offline tool (not needed at inference).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from wato_detection_3d.config import MODEL_FILES, load_config
from wato_detection_3d.engines import sha256_file, verify_model

SOURCE_SHA256 = "9d08cd562600fa73d1131d65b60c20645c6f2781338cd232185a1b235aae46da"
# Shape initializers in the pinned export, not weights or quantization scales.
RESHAPES = {
    "534": [6, 3, 256, 704],
    "1468": [6, 1, 256, 704],
    "1341": [6, 2048, 16, 44],
    "1470": [6, 256, 32, 88],
    "1556": [-1, 6, 256, 32, 88],
}


def export_backbone(source: Path, destination: Path, num_cameras: int) -> None:
    import onnx
    from onnx import numpy_helper

    if onnx.__version__ != "1.17.0":
        raise RuntimeError("use onnx==1.17.0 to reproduce the pinned export")
    if num_cameras < 1:
        raise ValueError("num_cameras must be positive")
    if sha256_file(str(source)) != SOURCE_SHA256:
        raise ValueError("camera export requires the pinned six-camera backbone")
    graph = onnx.load(str(source))
    initializers = {v.name: v for v in graph.graph.initializer}
    for name, expected in RESHAPES.items():
        value = numpy_helper.to_array(initializers[name]).copy()
        if value.tolist() != expected:
            raise ValueError(f"unexpected Reshape target {name}: {value}")
        value[value == 6] = num_cameras
        initializers[name].CopyFrom(numpy_helper.from_array(value, name=name))
    for value in graph.graph.input:
        value.type.tensor_type.shape.dim[1].dim_value = num_cameras
    for value in graph.graph.output:
        value.type.tensor_type.shape.dim[0].dim_value = num_cameras
    del graph.graph.value_info[:]
    graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(destination))


def export(config_path: str, output_dir: str, num_cameras: int) -> dict:
    cfg = load_config(config_path)
    source_hashes = verify_model(cfg.model)
    source, destination = Path(cfg.model.dir), Path(output_dir)
    if source.resolve() == destination.resolve():
        raise ValueError("export into a new directory to preserve the source model")
    destination.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        target = destination / name
        if name == "camera.backbone.onnx":
            export_backbone(source / name, target, num_cameras)
        else:
            shutil.copyfile(source / name, target)
    info = {
        "num_cameras": num_cameras,
        "source_sha256": source_hashes,
        "sha256": {name: sha256_file(str(destination / name)) for name in MODEL_FILES},
        "method": "shared camera encoder; specialize camera axis and Reshape targets",
    }
    (destination / "camera_export.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="verified six-camera source config"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-cameras", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.config, args.output_dir, args.num_cameras), indent=2))


if __name__ == "__main__":
    main()
