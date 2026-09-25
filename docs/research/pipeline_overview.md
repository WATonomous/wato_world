# Research Alignment: Pipeline Overview

This document maps the research papers to the nine wato_world pipeline
components and describes how our sensor rig relates to each paper's assumptions.
Per-paper detail lives in the `*_guidance.md` / `*_design.md` files alongside.

---

## Sensor rig (recording_20260217_224728_1.mcap)

| Sensor | Topics | Rate |
|--------|--------|------|
| LiDAR center | `/lidar_cc/velodyne_points` | ~20 Hz |
| LiDAR NE | `/lidar_ne/velodyne_points` | ~20 Hz |
| LiDAR NW | `/lidar_nw/velodyne_points` | ~20 Hz |
| Camera lower (×4) | `/camera_lower_{ne,nw,se,sw}/image_rect_compressed` | ~12 Hz |
| Camera panoramic (×8) | `/camera_pano_{ee,ne,nn,nw,se,ss,sw,ww}/image_rect_compressed` | ~12 Hz |
| Pose | eidos `map` frame: `/world_modeling/liso/odometry` (per scan, scan-stamped; agrees with the INS to ~5 cm/s) or `/world_modeling/slam/odometry` (newest keyframe; 1 per 5 m on eidos main); `/novatel/oem7/odom` (INS, UTM, ~2° attitude bias vs LiDAR on ring_road_corrected) for bags without eidos. Ingest requires a dense, smooth stream (ingest README "Pose requirements", "How good are the poses") | ~13 Hz LISO (matched scans) / keyframe-rate SLAM / 50 Hz INS position |
| Extrinsics | `/tf_static` | static |

Three LiDARs and twelve cameras give us considerably denser sensor coverage
than any of the papers assume (Waymo uses 5 cameras + 1 top LiDAR;
KITTI uses 2 cameras + 1 LiDAR).  This is a significant advantage for
multi-view shape fitting and multi-LiDAR point density.

---

## Paper → component mapping

```
ingest             ← bags + calibration + poses + chunk index
lidar_preprocessing← SAM4D preprocessing, static/dynamic split (AW log-odds /
                      MF-MOS / union), ground plane, UniLiPs IWU (Step E),
                      Chen et al. MOS auto-labeling proposals (Step F)
perception_2d      ← GroundingDINO detector + SAM2 video tracker + DA-V2 depth + DINOv2 ReID
semantic_lifting   ← occlusion-aware 2D→3D label lifting (UniLiPs Eq.1)
proposal_generation← Segment-Lift-and-Fit, Fusion4DAL LiDAR detector ensemble
tracking           ← Fusion4DAL 4D tracking + SAM4D temporal memory
label_refinement   ← LabelFormer trajectory refinement
open_vocab_discovery← rare-class extension (not covered by these papers)
student_training   ← distillation from auto-labels (not covered by these papers)
```

### UniLiPs (arxiv 2601.05105)
Drives **`semantic_lifting`** (Eq. 1 occlusion-aware lifting; see
`semantic_lifting_design.md`) and **`lidar_preprocessing` Step E** (the
Iterative Weighted Update over the bag static map, whose evicted "floaters"
become moving-object proposals; see `lidar_mos_guidance.md`).

### Chen et al., offline LiDAR MOS auto-labeling (arxiv 2201.04501)
Drives **`lidar_preprocessing` Step F**: map-cleaning candidates → HDBSCAN →
Kalman/Hungarian tracking → "moved further than its own size". Implemented as a
recall-oriented proposal artifact with soft motion features rather than a hard
relabel; the box/Kalman/association code lives in `wato_common.tracking` for
the `tracking` component to reuse. See `lidar_mos_guidance.md`.

### MF-MOS (SCNU-RISLAB)
Learned range-image MOS, opt-in via `--seg mos|union`. The default `--seg aw`
runs no model inference at all.

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
  └─ bag_meta.json, calibration.json  (bag-level: intrinsics, extrinsics, LiDAR frame IDs)
  └─ chunks/index.parquet
  └─ chunks/<chunk>/
       ├─ lidar_sweeps.parquet + lidar/<lidar_id>/<sweep>.npz (raw PointCloud2)
       ├─ camera_frames.parquet + cam_<CAM>/<seq>.jpg
       ├─ poses.parquet              (pose samples + which stretches may be interpolated;
       │                              every stage looks poses up via wato_common.pose_lookup,
       │                              at its own data's timestamp)
       └─ frame_index.parquet        (the contract every downstream stage reads; its
                                      world_T_ego is the pose at the SWEEP's time)

lidar_preprocessing   reads: sweeps, poses, calibration
  └─ chunks/<chunk>/
       ├─ lidar_proc_index.parquet   (per-sweep stats, world_path, dynamic_mask_path, mf_mos_mask_path)
       ├─ lidar_proc_summary.parquet (chunk stats: counts, seg method, MF-MOS, union, proposals)
       ├─ lidar_proc/*_world.npz     (deskewed world-frame xyz + origin + ground_mask + intensity)
       ├─ lidar_proc/*_dynamic_mask.npy     (PRECISION: the --seg method's per-point verdict)
       ├─ lidar_proc/*_motion_proposals.npz (RECALL: per-point source_bits + cluster_id, Step F)
       ├─ lidar_proc/*_mf_mos_mask.npy      (MF-MOS moving mask; --seg mos|union only)
       ├─ static_map.npz             (chunk static cloud + static/dynamic/ambiguous voxel keys)
       ├─ dynamic_map.npz            (chunk dynamic cloud + per-point sweep_id)
       ├─ motion_clusters.parquet    (Step F clusters: box + motion_score, track_life, ...)
       ├─ voxel_occupancy.npz        (sparse int32 voxel coords for MinkUNet encoder; all sweeps)
       └─ ground.npz                 (height grid + surface normals + raw ground points)
  └─ global_static_map.npz          (bag-level downsampled static cloud)
  └─ global_ground.npz              (bag-level height grid spanning all chunks)
  └─ global_iwu.npz                 (Step E: per-map-point static probability, evicted floaters)

perception_2d         reads: frames, poses, lidar_proc_index, lidar_proc/*_world.npz, dynamic_mask, calibration
  └─ chunks/<chunk>/
       ├─ detections_2d.parquet      (per-masklet class + detector score)
       ├─ tracklets_2d.parquet       (SAM2 temporal associations across frames)
       ├─ masks_2d/<masklet>/<seq>.png (per-masklet per-frame SAM2 masks)
       └─ depth_2d/<cam>/<frame>.npz (Depth Anything V2 + LiDAR affine-scaled metric depth;
                                      anchors exclude dynamic_mask points; projected with
                                      the pose at each image's own timestamp)

semantic_lifting      reads: frame_index, poses, lidar_proc/*_world.npz, masks_2d, tracklets_2d, depth_2d, calibration
  └─ chunks/<chunk>/semantic_lifting/
       ├─ lifted_labels/<sweep_id>.npz (per-point instance_id, class, confidence)
       └─ lifted_stats.parquet         (per-sweep lifting statistics)

proposal_generation   reads: lidar_proc/*_world.npz, lifted_labels, motion_clusters, ground.npz, calibration
  └─ chunks/<chunk>/
       ├─ proposals.parquet          (3D box proposals: center, size, heading, score, provenance)
       └─ proposal_masks/            (projected 2D mask used during SLF fitting)

tracking              reads: proposals, tracklets_2d, lidar_proc/*_world.npz  (builds on wato_common.tracking)
  └─ per-bag/
       └─ tracks.parquet             (track_id, chunk_id, sweep_id, box params, class)

label_refinement      reads: tracks, lidar_proc/*_world.npz, motion_proposals / dynamic_mask
  └─ per-bag/
       └─ refined_labels.parquet     (track_id, per-frame refined box + confidence)
```

---

## Implementation priority order

Done: `ingest`; `lidar_preprocessing` (Steps A–F: deskew, `--seg aw|mos|union`,
ground, reduce, IWU, motion proposals); `perception_2d` (GroundingDINO / SAM2 /
DA-V2 / DINOv2); `semantic_lifting` core (UniLiPs Eq. 1 lifting + cross-camera
voting). Next:

1. **`proposal_generation`** — the next unblocker
   - Map `motion_clusters.parquet` rows onto `ProposalRow`
     (`provenance="lidar_mos"`), gating on the soft features
     (`motion_score`, `n_sources`, `frac_persistent`)
   - LiDAR detector (CenterPoint or similar) on aggregated static/dynamic points
   - SLF: lift 2D masks into 3D using ground plane from `ground.npz`
   - Fuse LiDAR, MOS and SLF proposals (NMS or learned fusion)

2. **`tracking`** — once proposals exist
   - 3D Kalman filter on proposals across chunks — build on
     `wato_common.tracking` (the model lidar_preprocessing's Step F uses)
   - Masklet association using DINOv2 embeddings from `perception_2d`
   - Output: full-bag `tracks.parquet`

3. **`label_refinement`** — once tracking is done
   - Crop per-track LiDAR points using track boxes (+ motion proposals /
     dynamic masks)
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
| Chen et al. (MOS labels) | 1 LiDAR: 64-beam (KITTI, Apollo) or Ouster (MulRan, IPB-Car) | 3 LiDARs: 32 + 2×16 beams |
| UniLiPs | 1 LiDAR (KITTI 64-beam, nuScenes 32-beam) + cameras | 12 cameras, 3 LiDARs |

In every case we have more sensors.  This is mostly an advantage, but requires
deliberate multi-sensor fusion rather than the single-sensor assumptions baked
into these papers' implementations.  The exception is ring density: our
32-/16-beam scanners are sparser than KITTI's 64, which is what forced IWU's
windowed seen-through test (a map point between two rings otherwise reads as
"seen through" — see `lidar_mos_guidance.md`).
