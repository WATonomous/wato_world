# Research Alignment: Pipeline Overview

This document maps the four research papers to the eight wato_world pipeline
components and describes how our sensor rig relates to each paper's assumptions.

---

## Sensor rig (recording_20260217_224728_1.mcap)

| Sensor | Topics | Rate |
|--------|--------|------|
| LiDAR center | `/lidar_cc/velodyne_points` | ~20 Hz |
| LiDAR NE | `/lidar_ne/velodyne_points` | ~20 Hz |
| LiDAR NW | `/lidar_nw/velodyne_points` | ~20 Hz |
| Camera lower (×4) | `/camera_lower_{ne,nw,se,sw}/image_rect_compressed` | ~12 Hz |
| Camera panoramic (×8) | `/camera_pano_{ee,ne,nn,nw,se,ss,sw,ww}/image_rect_compressed` | ~12 Hz |
| Pose | `/novatel/oem7/odom` (NovAtel GNSS/INS, not eidos SLAM) | ~100 Hz |
| Extrinsics | `/tf_static` | static |

Three LiDARs and twelve cameras give us considerably denser sensor coverage
than any of the four papers assume (Waymo uses 5 cameras + 1 top LiDAR;
KITTI uses 2 cameras + 1 LiDAR).  This is a significant advantage for
multi-view shape fitting and multi-LiDAR point density.

---

## Paper → component mapping

```
ingest             ← bags + calibration + poses + chunk index
lidar_preprocessing← SAM4D preprocessing, static/dynamic split, ground plane
perception_2d      ← SAM4D camera encoding + 2D masks (SAM2/DEVA) + YOLO-World + Depth Anything V2
proposal_generation← MS3D++ detector ensemble + Segment-Lift-and-Fit + DA pseudo-LiDAR + cross-modal uncertainty
tracking           ← Fusion4DAL/DetZero 4D bidirectional tracking + SAM4D temporal memory
label_refinement   ← 3DAL/DetZero body-frame aggregation + LabelFormer trajectory refinement
open_vocab_discovery← rare-class extension (not covered by these papers)
student_training   ← distillation from auto-labels (not covered by these papers)
```

### SAM4D (arxiv 2506.21547)
Primarily drives **`perception_2d`** and informs **`lidar_preprocessing`**.

SAM4D is a multi-modal foundation model that jointly segments camera and
LiDAR streams using:
- Unified Multi-modal Positional Encoding (UMPE): both modalities lifted into
  shared 3D space
- Motion-aware Cross-modal Memory Attention (MCMA): SE(3) ego-motion aligns
  historical feature memories to the current frame
- Promptable segmentation: a point/box/mask prompt in one modality propagates
  to the other

See [sam4d_guidance.md](sam4d_guidance.md).

### MS3D++ (arxiv 2308.05988)
Drives the LiDAR-detector ensemble inside **`proposal_generation`**.

MS3D++ combines architecturally diverse 3D detectors (CenterPoint + DSVT +
FSDv2 in our setup) via Kernel-Based Fusion (or simpler Weighted Box Fusion):
- Each detector runs over the per-frame dynamic LiDAR cloud
- Optional test-time augmentation (flips, rotations) per detector
- KBF/WBF clusters proposals by BEV proximity and fuses center/size/heading
- Output records `n_detectors_agreeing` and `ensemble_score_var` as uncertainty
  signals

See [detector_ensemble_guidance.md](detector_ensemble_guidance.md).

### Segment-Lift-and-Fit (SLF)
Drives the camera-side proposal path in **`proposal_generation`**.

SLF turns a 2D mask (from SAM) into a 3D bounding box via three stages:
1. Segment — SAM produces per-camera 2D masks from point/box prompts
2. Lift — each mask is represented in a PCA vehicle-shape latent space via SDF
3. Fit — Adam optimizer minimizes: dice loss (mask reprojection) + LiDAR
   surface loss + ground alignment loss + **DA pseudo-LiDAR Chamfer loss**

See [segment_lift_fit_guidance.md](segment_lift_fit_guidance.md).

### Depth Anything V2 (arxiv 2406.09414)
Drives the monocular-depth pseudo-LiDAR pathway in **`perception_2d`** (depth
inference + caching) and **`proposal_generation`** (pseudo-LiDAR lift + SLF
`L_depth` term).

DA V2 Large predicts per-pixel relative depth which we rescale to metric via
LiDAR overlap, then lift inside each SAM2 mask into a dense pseudo-LiDAR cloud
that densifies SLF's fitting evidence (especially valuable for distant
objects and vertical surfaces).

See [depth_anything_guidance.md](depth_anything_guidance.md).

### 3DAL + DetZero (arxiv 2103.05073, arxiv 2306.06023)
Drives the multi-frame aggregation step inside **`label_refinement`**, before
LabelFormer.

Once `tracking` has produced coarse tracks, all dynamic LiDAR points from all
frames of a track are transformed into the object's body frame and
accumulated. Pairwise ICP between consecutive frame slices removes residual
pose noise. The resulting dense per-track cloud feeds an AggregateEncoder
that runs alongside LabelFormer's per-frame encoder.

See [body_frame_aggregation_guidance.md](body_frame_aggregation_guidance.md).

### LabelFormer (arxiv 2311.01444)
Drives the model side of **`label_refinement`**.

LabelFormer refines noisy initial boxes at the trajectory level:
- Per-frame encoder: embed the LiDAR points inside each frame's box crop
- **Aggregate encoder** (this pipeline's addition): embed the body-frame
  aggregated cloud once per track
- Temporal self-attention: reason over all frames of a track simultaneously
- Decoder: output refined size (W, L, H) once per track and per-frame pose
  (x, y, z, θ)

See [labelformer_guidance.md](labelformer_guidance.md).

### Cross-modal uncertainty
Cross-cutting bookkeeping across **`proposal_generation`** (compute) and
**`label_refinement`** (consume).

Each proposal records per-stage diagnostic signals
(`n_detectors_agreeing`, `slf_dice_loss`, `slf_depth_chamfer`,
`lidar_density_in_box`, etc.) and a combined `uncertainty ∈ [0, 1]`.
`label_refinement` uses this as a soft weight in its pose and size heads.

See [cross_modal_uncertainty_guidance.md](cross_modal_uncertainty_guidance.md).

---

## Data flow

```
ingest
  └─ chunks_index.parquet
  └─ per-chunk/
       ├─ lidar_sweeps.parquet + sweeps/*.npz (raw PointCloud2)
       ├─ cameras.parquet + frames/*.jpg
       ├─ poses.parquet
       └─ calibration.json  (intrinsics, extrinsics, LiDAR frame IDs)

lidar_preprocessing   reads: sweeps, poses, calibration
  └─ per-chunk/
       ├─ lidar_proc_index.parquet   (per-sweep stats, world_path, dynamic_mask_path)
       ├─ world/*.npz                (deskewed world-frame xyz + intensity + ground_mask)
       ├─ dynamic_masks/*.npy        (per-sweep boolean dynamic mask)
       ├─ static_map.npz             (chunk static cloud + voxel keys)
       └─ ground.npz                 (height grid + surface normals)
  └─ global_static_map.npz          (bag-level downsampled static cloud)

perception_2d         reads: frames, lidar_proc_index, world/*.npz, calibration
  └─ per-chunk/
       ├─ detections_2d.parquet      (per-frame box + class + confidence; ensemble of GroundingDINO + YOLO-World)
       ├─ masks_2d/                  (per-detection SAM2 masks, camera-aligned)
       ├─ tracklets_2d.parquet       (DEVA temporal associations across frames)
       ├─ depth_2d/<cam>/<seq>.npy   (Depth Anything V2 metric depth per camera frame, fp16)
       └─ depth_index.parquet        (per-frame scale/shift from LiDAR alignment + diagnostics)

proposal_generation   reads: world/*.npz, dynamic_map.npz, masks_2d, detections_2d,
                              depth_2d/, depth_index, ground.npz, calibration,
                              MODELS_ROOT/shape_priors/*.npz, MODELS_ROOT/lidar_detectors/*.pth
  └─ per-chunk/
       ├─ proposals.parquet          (3D box proposals: center, size, heading, score, provenance,
       │                              n_detectors_agreeing, slf_*_loss, *_chamfer, lidar_density_in_box,
       │                              uncertainty)
       └─ proposal_masks/            (projected 2D mask used during SLF fitting)

tracking              reads: proposals, tracklets_2d, world/*.npz, dino_features
  └─ per-bag/
       ├─ tracks_forward.parquet     (forward-pass tracks before merge)
       ├─ tracks_backward.parquet    (backward-pass tracks before merge)
       └─ tracks.parquet             (final merged tracks: direction column tags forward/backward/merged;
                                      merged_from column lists predecessor track_ids)

label_refinement      reads: tracks, world/*.npz, dynamic_masks, proposals.parquet (for uncertainty)
  └─ per-bag/
       ├─ aggregated_tracks/<track>.npz   (body-frame aggregated cloud per track + pose_history)
       ├─ aggregated_tracks_index.parquet (n_frames, n_points_aggregated, aggregation_method per track)
       └─ refined_labels.parquet          (track_id, per-frame refined box + confidence)
```

---

## Implementation priority order

Now sequenced into eight phases per the approved accuracy-upgrade plan:

0. **Research alignment docs** (this file + 4 paper-specific guides) — design
   reference for everything below.

1. **Shared infrastructure** — schemas, artifact_store helpers, geometry
   `body_frame.py`, watod `MODELS_ROOT` + `fetch-models`, Docker dependency
   updates, compose volume mounts.

2. **`perception_2d` refactor** — already implemented; extend with:
   - YOLO-World detector branch alongside GroundingDINO (parallel detection,
     IoU merge)
   - Depth Anything V2 metric depth per camera frame, cached as fp16 NPY
   - Per-frame scale/shift fit against overlapping LiDAR

3. **Shape-prior build script** — one-time job that voxelizes ShapeNetCore
   vehicles into SDFs, runs PCA, saves `shape_prior_<class>.npz` to
   `MODELS_ROOT`. Bootstrap pedestrian/cyclist priors from public box
   statistics.

4. **`proposal_generation` implementation** — once perception_2d artifacts
   and shape priors exist:
   - MS3D++ detector ensemble (CenterPoint + DSVT + FSDv2 via OpenPCDet)
   - Pseudo-LiDAR lift from DA depth maps inside SAM2 masks
   - SLF Adam fitter with L_mask + L_lidar + L_ground + L_depth
   - Ensemble ↔ SLF fusion + cross-modal uncertainty bookkeeping

5. **`tracking` implementation** — once proposals exist:
   - 3D Kalman filter per object (constant velocity + per-class noise)
   - Hungarian association (3D IoU + DINOv2 ReID cosine + class penalty)
   - Forward and backward passes; merge by Hungarian at track endpoints
   - Output `tracks.parquet` with `direction` and `merged_from` provenance

6. **`label_refinement` implementation** — once tracking is done:
   - Per-track body-frame aggregation (3DAL/DetZero) with ICP correction
     between consecutive frame slices
   - Two-encoder LabelFormer (frame encoder + aggregate encoder) feeding a
     transformer; size head shared per track, pose head per frame
   - Output `refined_labels.parquet` (the final auto-labels)

7. **Top-level docs + diagram + CLAUDE.md + component_versions bumps** —
   reflect the new pipeline shape, model conventions, and gotchas.

8. **End-to-end verification** — fetch models, run all six stages on a real
   bag, spot-check artifacts, run baseline comparison (ensemble off vs.
   ensemble on).

---

## Where our rig differs from paper assumptions

| Paper | Paper sensor setup | Our rig |
|-------|--------------------|---------|
| SAM4D | 5 cameras, 1 top LiDAR (Waymo) | 12 cameras, 3 LiDARs |
| SLF | 2 cameras, 1 LiDAR (KITTI) | 12 cameras, 3 LiDARs |
| LabelFormer | 1 LiDAR (ONCE) | 3 LiDARs |
| Fusion4DAL | multi-modal (exact rig TBD) | 12 cameras, 3 LiDARs |

In every case we have more sensors.  This is mostly an advantage, but requires
deliberate multi-sensor fusion rather than the single-sensor assumptions baked
into these papers' implementations.
