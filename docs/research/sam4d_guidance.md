# SAM4D Alignment Guidance

**Paper**: "SAM4D: Segment Anything in Camera and LiDAR Streams" (arxiv 2506.21547, ICCV 2025)

SAM4D is a promptable multi-modal segmentation model that jointly processes
camera and LiDAR streams with temporal consistency.  It is the closest research
analogue to our `perception_2d` component and informs the sensor-fusion design
throughout the pipeline.

---

## What SAM4D does

### Architecture

```
camera frames  ──► Hiera-S image encoder ──────────────────────────┐
                                                                    ▼
                   Unified Multi-modal Positional Encoding (UMPE) ──► shared 3D feature space
                                                                    ▲
LiDAR sweeps   ──► MinkUNet sparse conv encoder (stride 4) ────────┘
                        (voxel size 0.15 m, binary occupancy)

shared 3D features ──► Motion-aware Cross-modal Memory Attention (MCMA)
                              (SE(3) ego-motion to align past frames)
                         ──► shared mask decoder
                              ├─ 2D masks per camera
                              └─ 3D point masks per LiDAR sweep
```

### Key mechanisms

**UMPE (Unified Multi-modal Positional Encoding)**
Camera pixels are depth-lifted into 3D using the camera intrinsic matrix K and
the camera→LiDAR extrinsic transform.  Both the lifted image pseudo-points and
the real LiDAR points then receive the same sinusoidal + MLP positional
encoding in the shared 3D coordinate frame.  This is what allows a point prompt
in the camera image to directly retrieve LiDAR features and vice versa.

**MCMA (Motion-aware Cross-modal Memory Attention)**
At each timestep, the model maintains a memory bank of past features.  Before
attending to the memory, each stored feature's 3D position is transformed by
the SE(3) ego-motion `T_{t←t'}` (the transform from past frame t' to the
current frame t).  This keeps historical object features geometrically aligned
even as the vehicle moves, enabling long-range temporal tracking without
explicit object matching.

**Data engine**
SAM4D trains on a large pseudo-labeled dataset built by separating each scene
into a static background and multiple foreground objects, each tracked in its
own body coordinate frame.  The per-object tracking in body frame is what
produces temporally consistent 3D masks without human annotation.

---

## What we already have that aligns with SAM4D

| SAM4D component | Our equivalent | Status |
|-----------------|---------------|--------|
| LiDAR voxelization at 0.15 m | `voxel_size_m: 0.15` in `classify.py` | ✓ Done |
| Ego-motion per-point deskewing | `deskew.py` (`batch_interpolate_poses`) | ✓ Done |
| Static / dynamic separation | `classify.py` (voxel sweep-count threshold) | ✓ Done |
| Ground plane removal | `ground.py` (Patchwork++ + height grid) | ✓ Done |
| Camera intrinsics + extrinsics | `calibration.json` (from TF + camera_info) | ✓ Done |
| Multi-LiDAR coverage | 3 Velodynes vs. Waymo's 1 top LiDAR | ✓ Richer |
| Multi-camera coverage | 12 cameras vs. Waymo's 5 | ✓ Richer |

---

## Gaps and what to do

### 1. Binary voxel occupancy encoding

**What SAM4D does**: discards raw xyz coordinates and uses binary occupancy per
0.15 m voxel.  This improves cross-dataset generalization because the feature
is translation-invariant and intensity-agnostic.

**What we do**: store float64 xyz + float32 intensity in world-frame NPZ.

**What to add**: in `lidar_preprocessing`, optionally produce a per-chunk
`voxel_occupancy.npz` alongside the existing world NPZ files.  MinkUNet
consumes sparse (coords, features) pairs; the coords are the voxel integer
indices and the feature is a scalar `1.0` per occupied voxel.  This is a
two-liner on top of the existing `voxel_indices()` call in `classify.py`.

**File to touch**: `src/lidar_preprocessing/src/wato_lidar_preprocessing/classify.py`
(add a `save_voxel_occupancy` flag to `ComponentConfig` and write the occupancy
NPZ alongside `static_map.npz`).

### 2. Camera-LiDAR projection (depth lifting)

**What SAM4D does**: lifts each camera pixel to 3D via `depth * K_inv * [u,v,1]`
then applies `cam_T_lidar_inv` to get points in the LiDAR frame.  UMPE then
encodes both pixel pseudo-points and real LiDAR points the same way.

**What we have**: `calibration.json` has `ego_T_lidar` for each LiDAR and
`ego_T_cam` (implicitly via TF chain) for each camera, plus intrinsic matrices
K and distortion coefficients per camera.  Projection is straightforward.

**What to add** in `perception_2d`:
- A `projection.py` module that:
  - For each camera, builds the 3×4 projection matrix `P = K @ [R | t]`
    where `[R|t] = lidar_T_cam` (the camera-to-LiDAR transform).
  - Projects LiDAR points into image space to produce a sparse depth map.
  - Optionally lifts camera pixels at known depth back into 3D (pseudo-point cloud).
- This is the foundation for both cross-modal prompting and for SLF's mask
  alignment loss (proposal_generation needs to reproject fitted boxes back
  to image space to compute dice loss against SAM masks).

### 3. SAM2 2D mask generation + tracking

**What SAM4D uses**: SAM (the original or SAM2) for promptable segmentation;
then temporal propagation across frames.

**What is built** in `perception_2d` (see `perception_2d_v2.md`):
- Run GroundingDINO on keyframes to get class-labeled 2D bounding boxes
  (text prompts from the taxonomy: "car . truck . pedestrian . ...").
- Feed the boxes as prompts to SAM2's video predictor, which segments each box
  and propagates it across the frame stream into a tracked masklet — no separate
  DEVA pass; SAM2 does both segmentation and temporal association.
- For SAM4D-style cross-modal prompting (future): project LiDAR dynamic-mask
  points (from `dynamic_masks/*.npy`) into image space and use them as additional
  SAM2 point prompts.  This can recover objects that GroundingDINO missed.
- Output per-chunk: `detections_2d.parquet` + `tracklets_2d.parquet` + `masks_2d/`.

### 4. Temporal memory across chunks (MCMA)

**What SAM4D does**: each frame queries a rolling memory bank; memory feature
positions are transformed by the SE(3) ego-motion before attention so that
objects remain aligned despite vehicle motion.

**What we have**: `deskew.py` already stores all points in the SLAM world
frame, so there is no ego-motion drift issue within a chunk.  Across chunks
the world frame is consistent by construction.

**What to add**: DEVA already provides frame-level temporal consistency within
a video clip.  For cross-chunk consistency the ingest `chunks_index.parquet`
carries overlapping time windows (`t_overlap_start_ns`, `t_overlap_end_ns`).
`perception_2d` should process each chunk with a short look-back into the
previous chunk's overlap window so DEVA can bridge chunk boundaries.

### 5. Multi-LiDAR fusion

SAM4D assumes a single top-mounted LiDAR.  We have three Velodynes at different
positions.  To exploit this:

- In `lidar_preprocessing`, the world-frame NPZ files are already in a common
  coordinate frame (because deskew applies the per-LiDAR `ego_T_lidar`
  extrinsic before writing world xyz).  So the three LiDARs are already fused
  spatially — they can be treated as a single denser point cloud.
- For the voxel occupancy encoding, just concatenate all three LiDARs' points
  before voxelizing.  The chunk-level static/dynamic classification already
  does this implicitly since all sweeps feed into the same voxel key set.
- The main gap is **per-sweep synchronization**: the three LiDARs have
  independent trigger timing.  The current deskew uses header timestamps per
  sweep, which is correct.  When feeding a "current frame" to SAM4D-like
  models, define the canonical timestamp as the center LiDAR (`lidar_cc`)
  and treat NE/NW sweeps within ±25 ms as part of the same frame.

---

## Summary of actionable steps

1. `lidar_preprocessing`: add optional `voxel_occupancy.npz` export (sparse
   (N,3) int coords + ones) alongside existing static_map.npz.
2. `perception_2d` / `proposal_generation`: implement `projection.py` with
   `project_lidar_to_image()` and `lift_image_to_3d()` using calibration.json.
3. `perception_2d`: GroundingDINO → SAM2 video-tracker pipeline (done);
   add LiDAR-dynamic-point cross-modal prompting.
4. `perception_2d`: handle cross-chunk consistency via chunk overlap windows.
5. Multi-LiDAR: treat merged world-frame points (all three LiDARs) as the
   canonical dense point cloud for downstream perception steps.
