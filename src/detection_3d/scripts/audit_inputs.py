#!/usr/bin/env python3
"""Audit real BEVFusion inputs and each camera's effect without replacing detections.

Run in the detection_3d image after ingest/lidar_preprocessing. The report
records exact camera IDs/times and contributing LiDAR sweeps. Optional image
ablations verify each view reaches inference; they do not measure accuracy.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from wato_common.artifacts import calibration_refinement
from wato_common.artifacts.frame_index import FrameIndex
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.pose_lookup import PoseLookup
from wato_detection_3d import engines, model, pipeline
from wato_detection_3d.config import load_config


class Capture:
    def __init__(self, detector):
        self.detector = detector
        self.inputs = None

    def infer(self, points, cameras):
        self.inputs = points, cameras
        return self.detector.infer(points, cameras)


class CameraEncoder:
    """Probe the exported TensorRT graph without stochastic LiDAR voxelization."""

    def __init__(self, path):
        import tensorrt as trt

        self.trt = trt
        self.cuda = ctypes.CDLL("libcudart.so.12")
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.buffers = {}
        self.arrays = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            array = np.empty(
                shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name))
            )
            pointer = ctypes.c_void_p()
            self.check(
                self.cuda.cudaMalloc(
                    ctypes.byref(pointer), ctypes.c_size_t(array.nbytes)
                )
            )
            self.context.set_tensor_address(name, pointer.value)
            self.buffers[name] = pointer
            self.arrays[name] = array

    @staticmethod
    def check(status):
        if status != 0:
            raise RuntimeError(f"CUDA status {status}")

    def infer(self, normalized):
        for name, array in self.arrays.items():
            if self.engine.get_tensor_mode(name) != self.trt.TensorIOMode.INPUT:
                continue
            array[:] = normalized if name == "img" else 0
            self.check(
                self.cuda.cudaMemcpy(
                    self.buffers[name],
                    ctypes.c_void_p(array.ctypes.data),
                    ctypes.c_size_t(array.nbytes),
                    1,
                )
            )
        if not self.context.execute_async_v3(0):
            raise RuntimeError("camera encoder inference failed")
        array = self.arrays["camera_feature"]
        self.check(
            self.cuda.cudaMemcpy(
                ctypes.c_void_p(array.ctypes.data),
                self.buffers["camera_feature"],
                ctypes.c_size_t(array.nbytes),
                2,
            )
        )
        return array.astype(np.float32)

    def close(self):
        for pointer in self.buffers.values():
            self.check(self.cuda.cudaFree(pointer))


def probe_encoder(path, images, camera_ids):
    # Same resize/crop geometry and channel normalization as the vendor.
    rgb = (
        np.stack(
            [
                np.asarray(
                    Image.fromarray(im)
                    .resize((768, 432), Image.BILINEAR)
                    .crop((32, 176, 736, 432))
                )
                for im in images
            ]
        ).astype(np.float32)
        / 255.0
    )
    normalized = (
        (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    ).transpose(0, 3, 1, 2)[None]
    encoder = CameraEncoder(path)
    try:
        baseline = encoder.infer(normalized)
        repeat = encoder.infer(normalized)
        noise = float(np.max(np.abs(baseline - repeat)))
        report = {
            "output_shape": list(baseline.shape),
            "repeat_max_feature_delta": noise,
            "camera_ablations": {},
        }
        if baseline.shape[0] != len(camera_ids) or not np.isfinite(baseline).all():
            raise AssertionError("invalid encoder output")
        for i, cam_id in enumerate(camera_ids):
            ablated = normalized.copy()
            ablated[:, i] = 0
            features = encoder.infer(ablated)
            delta = float(np.max(np.abs(features[i] - baseline[i])))
            other = np.arange(len(camera_ids)) != i
            other_delta = float(np.max(np.abs(features[other] - baseline[other])))
            report["camera_ablations"][cam_id] = {
                "max_feature_delta": delta,
                "other_views_max_delta": other_delta,
            }
            if delta <= max(1e-5, 5 * noise) or other_delta > max(1e-3, 5 * noise):
                raise AssertionError(
                    f"camera {cam_id} is not encoded independently: {report}"
                )
        return report
    finally:
        encoder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--chunk", default="0000")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--ablate-cameras", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepared = engines.prepare(cfg.model)
    detector = model.Bevfusion(prepared, cfg.model.precision, cfg.model.num_cameras)
    capture = Capture(detector)
    calib = calibration_refinement.resolve(
        args.bag, cfg.calibration_source, cfg.cameras
    ).calibration
    sweeps = ProcessedSweeps.load(args.bag, args.chunk)
    poses = PoseLookup.load(args.bag, args.chunk)
    images = pipeline._Images(FrameIndex.load(args.bag, args.chunk))
    world = pipeline._Sweeps(sweeps)
    history = {}
    for meta in sweeps.rows:
        history.setdefault(meta.lidar_id, []).append(
            (meta.reference_timestamp_ns, meta.sweep_id)
        )
    for values in history.values():
        values.sort()
    report = {
        "bag": args.bag,
        "chunk": args.chunk,
        "config": cfg.model_dump(),
        "engines": prepared.info,
        "frames": [],
    }
    for frame in sweeps.frames(cfg.frames.canonical_lidar):
        row, boxes = pipeline.run_frame(
            cfg,
            frame,
            detector=capture,
            calib=calib,
            poses=poses,
            images=images,
            sweeps=sweeps,
            world=world,
            history=history,
        )
        if row.status != "ok":
            continue
        lidar_ids = sorted({sweeps.get(sid).lidar_id for sid in row.input_sweep_ids})
        expected = cfg.points.lidars or [sweeps.get(frame.reference_sweep_id).lidar_id]
        if set(lidar_ids) != set(expected):
            raise AssertionError(f"unexpected LiDAR input: {lidar_ids}")
        points, cameras = capture.inputs
        if cameras.images.shape[0] != len(cfg.cameras):
            raise AssertionError("missing camera images")
        record = {
            "sweep_id": row.sweep_id,
            "n_points": row.n_points,
            "input_sweep_ids": row.input_sweep_ids,
            "lidar_ids": lidar_ids,
            "camera_ids": cfg.cameras,
            "camera_timestamps_ns": row.camera_timestamps_ns,
            "image_shape": list(cameras.images.shape),
            "n_boxes": len(boxes),
        }
        if args.ablate_cameras and not report["frames"]:
            record["camera_encoder_probe"] = probe_encoder(
                prepared.camera_backbone, cameras.images, cfg.cameras
            )
            baseline = np.sort(detector.infer(points, cameras)[:, model.SCORE])
            repeat = np.sort(detector.infer(points, cameras)[:, model.SCORE])
            noise = float(np.max(np.abs(baseline - repeat)))
            record["repeat_max_score_delta"] = noise
            record["camera_ablations"] = {}
            for i, cam_id in enumerate(cfg.cameras):
                ablated = cameras.images.copy()
                ablated[i] = 0
                scores = np.sort(
                    detector.infer(points, replace(cameras, images=ablated))[
                        :, model.SCORE
                    ]
                )
                delta = float(np.max(np.abs(scores - baseline)))
                record["camera_ablations"][cam_id] = {
                    "max_score_delta": delta,
                    "above_repeat_noise_threshold": delta > max(1e-5, 5 * noise),
                }
        report["frames"].append(record)
        print(json.dumps(record), flush=True)
        if len(report["frames"]) >= args.frames:
            break
    if len(report["frames"]) < args.frames:
        raise AssertionError("not enough complete frames to audit")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
