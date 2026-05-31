#!/usr/bin/env python3
"""Fast standalone check for the SAM 3.1 concept tracker (skips the depth pass).

Run inside the perception_2d dev container:

    python3 /ws/src/perception_2d/scripts/smoke_track.py \
        NuScenes_v1_0_mini_scene_0061 0000 CAM_FRONT 6 2

Args: bag_id chunk_id cam_id n_frames n_concepts (all optional).
Prints masklet count + each mask's fill fraction so you can tell real masks
(small fill) from degenerate full-image masks (fill ~1.0) or empty output.
"""

from __future__ import annotations

import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from wato_perception_2d.config import load_config
from wato_perception_2d.io import load_frame_index
from wato_perception_2d.models._sam3_runtime import get_sam3_predictor
from wato_perception_2d.models.sam3_concept_tracker import track_camera_concepts
from wato_perception_2d.pipeline import _load_image


def main() -> int:
    bag = sys.argv[1] if len(sys.argv) > 1 else "NuScenes_v1_0_mini_scene_0061"
    chunk = sys.argv[2] if len(sys.argv) > 2 else "0000"
    cam = sys.argv[3] if len(sys.argv) > 3 else "CAM_FRONT"
    n_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    n_concepts = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    cfg = load_config("/ws/src/perception_2d/config/perception_2d.yaml")

    frames = [f for f in load_frame_index(bag, chunk) if f.cam_id == cam][:n_frames]
    pairs = [(f, _load_image(f.image_path)) for f in frames]
    pairs = [(f, im) for f, im in pairs if im is not None]
    frames = [f for f, _ in pairs]
    images = [im for _, im in pairs]
    if not frames:
        print(f"no frames for {bag}/{chunk}/{cam}")
        return 1
    print(f"{len(frames)} frames, image {images[0].shape}")

    predictor = get_sam3_predictor(version=cfg.segmentation.version, use_fa3=cfg.segmentation.use_fa3)
    print("predictor loaded:", predictor is not None)
    if predictor is None:
        return 1

    concepts = cfg.concept_prompts()[:n_concepts]
    print("concepts:", concepts)

    masklets = track_camera_concepts(
        predictor, frames, images, concepts,
        bag_id=bag, chunk_id=chunk, cam_id=cam,
        masks_2d_base_dir="/tmp/smoke_masks",
        dino_every_k=0,  # skip DINOv2 for speed
        device="cuda",
        output_prob_thresh=cfg.segmentation.output_prob_thresh,
    )

    print(f"=> {len(masklets)} masklets")
    from PIL import Image as PILImage
    for m in masklets[:20]:
        fill = ""
        if m.mask_paths:
            arr = np.array(PILImage.open(m.mask_paths[0]))
            fill = f" mask0={arr.shape} fill={(arr > 0).mean():.3f}"
        print(f"  cls={m.cls:<12} frames={len(m.frames_present):<3} score={m.score:.2f}{fill}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
