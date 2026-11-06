#!/usr/bin/env python3
"""Run a mask-only sample into a new artifact root; no labels are required.

Run inside the perception image. Original ingest/mask artifacts are read-only.
Uses configured detector and tracking rules, omitting depth and DINO embeddings
because neither changes generated masks. Statistics are diagnostics, not accuracy.
"""

from __future__ import annotations

import argparse
import inspect
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from wato_common.artifacts import masklets
from wato_common.artifacts.frame_index import FrameIndex
from wato_perception_2d import mask_quality
from wato_perception_2d.config import load_config
from wato_perception_2d.model_registry import revisions
from wato_perception_2d.models._sam2_runtime import (
    check_sam2_cuda_extension,
    get_sam2_predictor,
    propagation_provenance,
)
from wato_perception_2d.models.sam2_tracker import track_camera
from wato_perception_2d.models.yolo_detector import YoloDetector
from wato_perception_2d.pipeline import _ego_mask, _load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--chunk", required=True)
    parser.add_argument("--cam", required=True)
    parser.add_argument("--backend", choices=["sam2", "dam4sam"], required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument(
        "--output", type=Path, required=True, help="New, empty output root"
    )
    args = parser.parse_args()
    if args.start < 0 or args.frames < 1:
        parser.error("start must be nonnegative and frames must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be empty; existing artifacts are never overwritten")
    cfg = load_config(args.config)
    cfg.segmentation.backend = args.backend
    if cfg.discovery.backend != "fixed" or cfg.detection.backend != "yolo":
        parser.error("This mask diagnostic currently supports fixed taxonomy + YOLO")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_sam2_cuda_extension(device)
    frames = FrameIndex.load(args.bag, args.chunk).by_camera()[args.cam]
    frames = frames[args.start : args.start + args.frames]
    if not frames:
        parser.error("No frames in requested range")
    images = [_load_image(f.image_path) for f in frames]
    if any(image is None for image in images):
        raise RuntimeError("Sample contains unreadable images")
    yolo = cfg.detection.yolo
    detector = YoloDetector(
        yolo.checkpoint,
        class_map=yolo.class_map,
        conf=yolo.conf,
        iou=yolo.iou,
        imgsz=yolo.imgsz,
        device=device,
    )
    predictor = get_sam2_predictor(
        cfg.segmentation.checkpoint,
        cfg.segmentation.config,
        device,
        backend=args.backend,
    )
    if predictor is None:
        raise RuntimeError("Could not load predictor")
    kwargs = {
        k: v
        for k, v in cfg.tracking.model_dump().items()
        if k in inspect.signature(track_camera).parameters
    }
    ego = _ego_mask(cfg, args.cam, images[0].shape[1], images[0].shape[0])
    input_root = os.environ.get("ARTIFACT_ROOT_URI")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ARTIFACT_ROOT_URI"] = output.as_uri()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    diagnostics: list[dict] = []
    result = track_camera(
        predictor,
        detector,
        frames,
        images,
        [(c, c) for c in cfg.discovery.fixed_classes],
        bag_id=args.bag,
        chunk_id=args.chunk,
        cam_id=args.cam,
        tracker_backend=args.backend,
        device=device,
        dino_every_k=0,
        redetect_every_k=cfg.detection.redetect_every_k,
        ego_mask=ego,
        ego_max_fraction=cfg.ego_mask.max_fraction,
        refinement=cfg.segmentation.refinement,
        diagnostics=diagnostics,
        **kwargs,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    try:
        importlib.import_module("sam2._C")
        cuda_components = True
    except ImportError:
        cuda_components = False
    masklets.write(
        args.bag,
        args.chunk,
        [
            masklets.MaskletRecord(
                masklet_id=m.masklet_id,
                cam_id=m.cam_id,
                cls=m.cls,
                score=m.score,
                frames_present=tuple(m.frames_present),
                tracker_backend=m.tracker_backend,
            )
            for m in result
        ],
    )
    mask_quality.write(
        args.bag,
        args.chunk,
        args.cam,
        backend=args.backend,
        refinement=cfg.segmentation.refinement.model_dump(mode="json"),
        events=diagnostics,
    )
    events = [e for e in diagnostics if e["kind"] == "mask"]
    summary = dict(
        backend=args.backend,
        bag=args.bag,
        chunk=args.chunk,
        camera=args.cam,
        input_artifact_root=input_root,
        device=device,
        frames=len(frames),
        sample_duration_seconds=(frames[-1].timestamp_ns - frames[0].timestamp_ns)
        / 1e9,
        sam2_cuda_extension_available=cuda_components,
        camera_sequences=[f.camera_seq for f in frames],
        effective_config=cfg.model_dump(mode="json"),
        models=revisions(),
        mask_propagation=propagation_provenance(args.backend, device),
        seconds=seconds,
        peak_gpu_gib=(
            torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
        ),
        masklets=len(result),
        saved_masks=sum(len(m.frames_present) for m in result),
        mask_events=len(events),
        empty_masks=sum(e["raw_area"] == 0 for e in events),
        median_threshold_stability=(
            float(np.median([e["stability"] for e in events])) if events else None
        ),
        accuracy_measured=False,
        depth_and_embeddings_omitted=True,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"effective_config", "models", "camera_sequences"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
