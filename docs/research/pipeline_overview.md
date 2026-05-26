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
perception_2d      ← SAM4D camera encoding + 2D mask generation (SAM3/DEVA)
proposal_generation← Segment-Lift-and-Fit, Fusion4DAL LiDAR detector ensemble
tracking           ← Fusion4DAL 4D tracking + SAM4D temporal memory
label_refinement   ← LabelFormer trajectory refinement
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

### Segment-Lift-and-Fit (SLF)
Drives **`proposal_generation`**.

SLF turns a 2D mask (from SAM) into a 3D bounding box via three stages:
1. Segment — SAM produces per-camera 2D masks from point/box prompts
2. Lift — each mask is represented in a PCA vehicle-shape latent space via SDF
3. Fit — Adam optimizer minimizes: dice loss (mask reprojection) + LiDAR
   surface loss + ground alignment loss

### Fusion4DAL
Drives **`proposal_generation`** (LiDAR detector side) and **`tracking`**.

Fusion4DAL describes an offline pipeline that fuses multi-modal detectors
(camera + LiDAR) with 4D (spatial + temporal) aggregation for auto-labeling.
It is the architectural blueprint for how our detector ensemble results are
fused before being handed to tracking.

### LabelFormer (arxiv 2311.01444)
Drives **`label_refinement`**.

LabelFormer refines noisy initial boxes at the trajectory level:
- Per-frame encoder: embed the LiDAR points inside each frame's box crop
- Temporal self-attention: reason over all frames of a track simultaneously
- Decoder: output refined size (W, L, H) and per-frame pose (x, y, z, θ)

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
       ├─ detections_2d.parquet      (per-frame box + class + confidence)
       ├─ masks_2d/                  (per-detection SAM3 masks, camera-aligned)
       └─ tracklets_2d.parquet       (DEVA temporal associations across frames)

proposal_generation   reads: world/*.npz, masks_2d, detections_2d, ground.npz, calibration
  └─ per-chunk/
       ├─ proposals.parquet          (3D box proposals: center, size, heading, score, source)
       └─ proposal_masks/            (projected 2D mask used during SLF fitting)

tracking              reads: proposals, tracklets_2d, world/*.npz
  └─ per-bag/
       └─ tracks.parquet             (track_id, chunk_id, sweep_id, box params, class)

label_refinement      reads: tracks, world/*.npz, dynamic_masks
  └─ per-bag/
       └─ refined_labels.parquet     (track_id, per-frame refined box + confidence)
```

---

## Implementation priority order

1. **`perception_2d`** — unblocks everything downstream
   - SAM3 mask generation per camera (text/box prompts via GroundingDINO)
   - DEVA temporal propagation across frames
   - DINOv2 per-detection embedding for ReID downstream

2. **`proposal_generation`** — once 2D masks exist
   - LiDAR detector (CenterPoint or similar) on aggregated static/dynamic points
   - SLF: lift 2D masks into 3D using ground plane from `ground.npz`
   - Fuse LiDAR proposals + SLF proposals (NMS or learned fusion)

3. **`tracking`** — once proposals exist
   - 3D Kalman filter on proposals across chunks
   - Masklet association using DINOv2 embeddings from `perception_2d`
   - Output: full-bag `tracks.parquet`

4. **`label_refinement`** — once tracking is done
   - Crop per-track LiDAR points using dynamic masks + track boxes
   - Run LabelFormer trajectory-level self-attention
   - Output: `refined_labels.parquet` (the final auto-labels)

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
