# perception_2d

2D foundation pass: detects objects in every camera frame, segments them
into per-instance masks, tracks them across frames, lifts per-frame depth
maps (monocular), and reconciles tracks across cameras.  Heaviest GPU
component in the pipeline.

## Pipeline steps

```
ingest        → frame_index.parquet, camera images, calibration.json
lidar_preprocessing → world/*.npz, dynamic_masks/*.npy
        │
        ▼
A. Detect           GroundingDINO + YOLO-World (DetectorEnsemble)
        │           ▸ open-vocabulary 2D boxes per camera frame
        ▼
B. Segment          SAM2 + LiDAR-projected point prompts (SAM4D-style)
        │           ▸ per-detection binary masks
        ▼
B.5. Depth          Depth Anything V2 + LiDAR-based scale/shift fit
        │           ▸ depth_2d/<cam>/<seq>.npy + depth_index.parquet
        ▼
C. Track            IoU-based per-camera tracker with DINOv2 ReID embeds
        │           ▸ per-camera masklets
        ▼
D. Cross-cam merge  Cluster masklets across cameras by world-frame centroid
        │           ▸ global_object_id assigned within radius
        ▼
E. Write            detections_2d.parquet · tracklets_2d.parquet ·
                    depth_index.parquet · masks_2d/<masklet>/*.png ·
                    depth_2d/<cam>/*.npy
```

## What's new vs. the earlier 2D-only flow

This component used to be a single-detector (GroundingDINO) + SAM2 stack.
Two upgrades landed in this iteration:

- **YOLO-World as a parallel detector branch.**  Wrapped in a
  `DetectorEnsemble` that runs both detectors per frame and merges by IoU
  within class.  Improves rare-class recall (construction workers, debris,
  uncommon vehicles) that GroundingDINO's text encoder occasionally misses.
  Default ensemble configuration in [config/pipeline.yaml](../../config/pipeline.yaml).
  Background: [docs/research/detector_ensemble_guidance.md](../../docs/research/detector_ensemble_guidance.md).
- **Depth Anything V2 metric depth (step B.5).**  Per camera frame, DA V2
  predicts relative depth, then a RANSAC + least-squares fit against
  projected LiDAR points recovers a (scale, shift) that converts to metric.
  Raw fp16 depth maps are persisted to `depth_2d/<cam>/<seq>.npy` and the
  (scale, shift) lookup lives in `depth_index.parquet`.  Downstream
  `proposal_generation` lifts these depth maps inside SAM2 masks into a
  dense pseudo-LiDAR cloud that strengthens Segment-Lift-Fit's surface fit.
  Background: [docs/research/depth_anything_guidance.md](../../docs/research/depth_anything_guidance.md).

## Inputs

| Input | Source |
|---|---|
| `frame_index.parquet` | ingest |
| `cam_*/<seq>.jpg` | ingest |
| `calibration.json` (intrinsics + extrinsics) | ingest |
| `lidar_proc/<sweep>_world.npz` | lidar_preprocessing |
| `lidar_proc/<sweep>_dynamic_mask.npy` | lidar_preprocessing |
| `config/pipeline.yaml` (perception_2d section) | this component |
| `$MODELS_ROOT/*` (DA V2, YOLO-World, SAM2, GroundingDINO weights) | `watod fetch-models` |

## Outputs (per chunk, under `raw/<bag>/chunks/<chunk>/`)

| Artifact | Schema / format |
|---|---|
| `detections_2d.parquet` | `MASKLET_SCHEMA` (also written as `tracklets_2d.parquet` for backward compat) |
| `tracklets_2d.parquet` | `MASKLET_SCHEMA` — same rows as above |
| `masks_2d/<masklet_id>/<camera_seq:06d>.png` | binary mask PNGs |
| `masks_2d/<masklet_id>/dino_feature.npy` | DINOv2 ReID embedding (per masklet) |
| `depth_2d/<cam_id>/<camera_seq:06d>.npy` | raw DA V2 relative depth, fp16 |
| `depth_index.parquet` | `DEPTH_INDEX_SCHEMA` — per-frame (scale, shift, residual_rmse, n_overlap_pts, scale_method) |

`depth_2d/` is consumed by `proposal_generation`'s pseudo-LiDAR lift step.
The actual metric conversion is `z_metric = scale * d_raw + shift` and lives
in the index parquet so downstream readers don't have to re-fit.

## Configuration

All knobs are in [config/pipeline.yaml](../../config/pipeline.yaml) under the
`perception_2d:` key.  Highlights:

| Key | Default | Description |
|---|---|---|
| `detectors[].name` | `grounding_dino`, `yolo_world` | Ordered list of detectors that run per frame.  Drop entries to disable. |
| `detector_ensemble_iou` | 0.6 | IoU threshold for merging detectors' boxes within the same class. |
| `sam2_checkpoint` | `sam2_hiera_large` | SAM2 model variant. |
| `use_lidar_prompts` | `true` | Project LiDAR dynamic points as SAM2 cross-modal prompts (SAM4D-style). |
| `depth_estimator.enabled` | `true` | Toggle Depth Anything V2 inference + persistence. |
| `depth_estimator.model` | `depth_anything_v2_large` | DA V2 variant.  Only `large` is supported in v1. |
| `depth_estimator.save_dtype` | `float16` | NPY dtype used to persist `depth_2d/*.npy`. |
| `depth_estimator.align_to_lidar` | `true` | Run the per-frame RANSAC fit against overlapping LiDAR. |
| `depth_estimator.min_overlap_pts` | 50 | Minimum projected LiDAR points required to trust the fit. |
| `depth_estimator.inlier_thresh_m` | 0.5 | RANSAC inlier distance (m). |
| `reid_features.model` | `dinov2_vitl14` | Per-detection ReID embedding model. |
| `cross_camera_match_radius_m` | 1.5 | World-frame distance threshold for cross-camera tracklet clustering. |

## Models

GPU weights live under `$MODELS_ROOT` (mounted as `/data/models` inside the
container).  Run once after a fresh clone:

```bash
./watod fetch-models                       # all registered weights
./watod fetch-models -c perception_2d      # only this component's weights
```

Currently registers:

- `depth_anything_v2/depth_anything_v2_vitl.pth` (~1.3 GB)
- `yolo_world/yolov8l-worldv2.pt` (~150 MB)

SAM2 + GroundingDINO weights are still pulled lazily by their respective
libraries at first use until a follow-up adds them to the fetch manifest.

## How to run

```bash
# Edit ACTIVE_MODULES in watod-config.sh, e.g. ACTIVE_MODULES="perception_2d:dev"
./watod build                         # pulls bases + builds perception_2d image
./watod run perception_2d --bag <bag_id>
./watod -t perception_2d_dev          # interactive shell with src/ bind-mounted
./watod test perception_2d
```

## Spot-checks after a run

```python
import numpy as np
import pyarrow.parquet as pq

# Per-camera depth at a specific frame.
depth = np.load("data/artifacts/raw/<bag>/chunks/<chunk>/depth_2d/<cam>/000042.npy")
print("depth shape:", depth.shape, "dtype:", depth.dtype)

# Depth index → (scale, shift) per frame.
idx = pq.read_table("data/artifacts/raw/<bag>/chunks/<chunk>/depth_index.parquet").to_pandas()
print(idx[["cam_id", "camera_seq", "scale", "shift", "scale_method"]].head())

# Recover metric depth at a pixel:
#   z_metric = scale * depth + shift
```

## Local development (no Docker)

Heavy GPU deps are listed under the `gpu` optional extra in
[src/perception_2d/pyproject.toml](pyproject.toml).  For a CPU dev env that
runs the unit tests:

```bash
PYTHONPATH=src/common/src:src/perception_2d/src \
    python3 -m pytest src/perception_2d/tests -q
```

The tests do not require torch / ultralytics / depth-anything-v2 to be
installed — adapters fall back to empty output when their model package is
unavailable, and the depth alignment helper is pure numpy.
