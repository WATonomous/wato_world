"""TensorRT engines for BEVFusion, built once from the model's ONNX files.

A TensorRT plan only runs on the TensorRT version and GPU that built it, so
plans are cached per (model files, TensorRT version, GPU, precision) under
model.engine_cache_dir and never shipped. The build mirrors the car's
BEVFusionCore::compileTrtModel (2 GB workspace, detailed profiling verbosity,
linear I/O formats, the custom LayerNorm plugin loaded first) with one
correction: on TensorRT < 11, whose networks aren't strongly typed, INT8 is
enabled only for the engines whose ONNX carries quantize/dequantize scales —
camera.backbone and fuser — as NVIDIA's tool/build_trt_engine.sh does. The car
enables it for all four, which TensorRT 10 rejects for camera.vtransform
("Calibration failure ... no scaling factors").
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from wato_detection_3d.config import MODEL_FILES, ModelConfig

log = logging.getLogger(__name__)

# The four engines, by the ONNX file each is built from (the car's).
PLANS = {
    "camera.backbone": "camera.backbone.onnx",
    "camera.vtransform": "camera.vtransform.onnx",
    "fuser": "fuser.onnx",
    "head.bbox": "head.bbox.onnx",
}
# The engines quantized in an int8 model (NVIDIA's build_trt_engine.sh).
INT8_PLANS = ("camera.backbone", "fuser")
LIDAR_BACKBONE = "lidar.backbone.xyz.onnx"
LAYERNORM_PLUGIN = "libcustom_layernorm.so"


@dataclass(frozen=True)
class Engines:
    directory: str
    camera_backbone: str
    camera_vtransform: str
    fuser: str
    head_bbox: str
    lidar_backbone: str  # ONNX, read by libspconv
    info: dict  # what built them: written to each manifest


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_model(cfg: ModelConfig) -> dict[str, str]:
    """Each model file's sha256; raises when one differs from its pin."""
    actual = {}
    for name in MODEL_FILES:
        path = os.path.join(cfg.dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing: install the car's BEVFusion model "
                "(src/detection_3d/README.md, 'Model files')"
            )
        actual[name] = sha256_file(path)
        if actual[name] != cfg.sha256[name]:
            raise ValueError(
                f"{path}: sha256 {actual[name]} is not the pinned "
                f"{cfg.sha256[name]}; a different model is a different labeler"
            )
    return actual


def gpu() -> dict[str, str]:
    """Name and compute capability of CUDA device 0, from the driver API."""
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0) != 0:
        raise RuntimeError("cuInit failed: no usable NVIDIA driver/GPU")
    dev = ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
        raise RuntimeError("cuDeviceGet(0) failed")
    name = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name, 256, dev)
    major, minor = ctypes.c_int(), ctypes.c_int()
    cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)  # COMPUTE_CAPABILITY_MAJOR
    cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)  # COMPUTE_CAPABILITY_MINOR
    return {
        "name": name.value.decode(),
        "compute_capability": f"{major.value}.{minor.value}",
    }


def load_plugin() -> None:
    """head.bbox needs the fork's LayerNorm plugin registered, to build and to
    deserialize (the car dlopens it before both)."""
    ctypes.CDLL(LAYERNORM_PLUGIN, mode=ctypes.RTLD_GLOBAL)


def _build_plan(trt, onnx_path: str, plan_path: str, int8: bool) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse {onnx_path}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2048 << 20)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if int(trt.__version__.split(".")[0]) < 11:
        config.set_flag(trt.BuilderFlag.FP16)
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
    linear = 1 << int(trt.TensorFormat.LINEAR)
    for i in range(network.num_inputs):
        network.get_input(i).allowed_formats = linear
    for i in range(network.num_outputs):
        network.get_output(i).allowed_formats = linear
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build {onnx_path}")
    tmp = plan_path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(plan)
    os.replace(tmp, plan_path)


def prepare(
    cfg: ModelConfig, *, file_hashes: Optional[dict[str, str]] = None
) -> Engines:
    """Verify the model files and return its engines, building any missing."""
    import tensorrt as trt

    hashes = file_hashes or verify_model(cfg)
    info = {
        "model_dir": cfg.dir,
        "model_files_sha256": hashes,
        "precision": cfg.precision,
        "int8_engines": list(INT8_PLANS) if cfg.precision == "int8" else [],
        "tensorrt": trt.__version__,
        "gpu": gpu(),
        "wato_cuda_bevfusion_commit": os.environ.get("WATO_CUDA_BEVFUSION_COMMIT", ""),
        "cudasm": os.environ.get("WATO_CUDASM", ""),
    }
    key_src = {
        k: info[k]
        for k in (
            "model_files_sha256",
            "precision",
            "int8_engines",
            "tensorrt",
            "gpu",
            "cudasm",
        )
    }
    key = hashlib.sha256(json.dumps(key_src, sort_keys=True).encode()).hexdigest()[:16]
    directory = os.path.join(
        cfg.engine_cache_dir, os.path.basename(cfg.dir.rstrip("/")), key
    )
    os.makedirs(directory, exist_ok=True)
    load_plugin()
    for name, onnx in PLANS.items():
        plan = os.path.join(directory, f"{name}.plan")
        if not os.path.exists(plan):
            log.info("building TensorRT engine %s (minutes, once per GPU)", plan)
            int8 = cfg.precision == "int8" and name in INT8_PLANS
            _build_plan(trt, os.path.join(cfg.dir, onnx), plan, int8)
    with open(os.path.join(directory, "build_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    plan = lambda n: os.path.join(directory, f"{n}.plan")  # noqa: E731
    return Engines(
        directory=directory,
        camera_backbone=plan("camera.backbone"),
        camera_vtransform=plan("camera.vtransform"),
        fuser=plan("fuser"),
        head_bbox=plan("head.bbox"),
        lidar_backbone=os.path.join(cfg.dir, LIDAR_BACKBONE),
        info=dict(info, engine_dir=directory),
    )
