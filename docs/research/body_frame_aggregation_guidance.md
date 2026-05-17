# Body-Frame Multi-Frame Aggregation Alignment Guidance

**Papers**:
- "Offboard 3D Object Detection from Point Cloud Sequences" (Qi et al., Waymo,
  arXiv 2103.05073, 2021) — the original 3DAL paper
- "DetZero: Rethinking Offboard 3D Object Detection with Long-Term Sequential
  Point Clouds" (Ma et al., ICCV 2023, arXiv 2306.06023) — the current
  best-in-class offline auto-labeler

The single biggest accuracy lever available to an offline auto-labeling
pipeline is multi-frame aggregation of LiDAR returns in each object's body
frame. A pedestrian at 30 m typically gives 20–50 LiDAR returns per sweep.
Over a 4 s track at 10 Hz that's 800–2000 points to fit a box to, vs. 20–50
per-frame. The shape and size estimate stops being sparsity-limited and
becomes geometry-limited.

This document is the design blueprint for the aggregation step that lives in
`label_refinement`, immediately before the LabelFormer encoder.

---

## What 3DAL / DetZero do

```
   per-track input from `tracking`:
     frame   center (world)   size (initial)   heading
       0     (cx0,cy0,cz0)    (w0,l0,h0)        θ0
       1     (cx1,cy1,cz1)    (w1,l1,h1)        θ1
       ...
       T     (cxT,cyT,czT)    (wT,lT,hT)        θT
            │
   ┌────────┴────────────────────────────────────┐
   │ For each frame:                              │
   │   1. Crop enlarged box (margin 1.5×)         │
   │      from world/*.npz ∩ dynamic_masks/*.npy  │
   │   2. Transform crop world → body frame:      │
   │        R_body = R_z(-θ_frame)                 │
   │        t_body = -(cx, cy, cz)                 │
   │        pts_body = R_body @ (pts_world + t_body)│
   └─────────────────────┬────────────────────────┘
                         │
                         ▼
         per-frame body-frame clouds, all roughly aligned
                         │
                         ▼
   ┌─────────────────────────────────────────┐
   │  Concatenate across frames               │
   │  → (Σ N_t, 3) body-frame cloud           │
   └─────────────────────┬───────────────────┘
                         │
                         ▼
   ┌─────────────────────────────────────────┐
   │  ICP between consecutive frame slices    │   DetZero's "geometric refining"
   │  → small per-frame correction (R, t)     │   corrects residual pose noise
   │  → re-aggregate                          │   from the upstream tracker
   └─────────────────────┬───────────────────┘
                         │
                         ▼
   ┌─────────────────────────────────────────┐
   │  Voxel downsample (1 cm)                 │
   │  Statistical outlier removal             │
   └─────────────────────┬───────────────────┘
                         │
                         ▼
   aggregated body-frame cloud  → LabelFormer encoder input
```

**Why "body frame", not "world frame"**. A car moves several metres between
frames. In world frame, the returns scatter along the trajectory and look
like noise. In the car's own coordinate frame — with the car's heading
cancelled out and its centre at the origin — the returns from every frame
line up into a coherent dense surface of the car's actual shape.

**Why ICP correction matters**. The upstream tracker's per-frame heading and
centre are noisy (±0.1 m and ±2° is typical). Without correction, the
"aligned" body-frame points still have a per-frame offset, smearing the
surface. ICP between consecutive frame slices recovers tiny rigid
corrections that sharpen the aggregated surface considerably. DetZero
shows this is worth ~15% AP improvement over naive concat.

---

## What we already have that aggregation needs

| Aggregation requirement | Our artifact | Where |
|---|---|---|
| Per-sweep dynamic LiDAR in world frame | `world/*.npz` + `dynamic_masks/*.npy` | `lidar_preprocessing` |
| Upstream coarse track per object | `tracks.parquet` | `tracking` (after Phase 5) |
| Heading per frame (initial) | `tracks.parquet:heading` | `tracking` |
| Centre per frame (initial) | `tracks.parquet:(cx,cy,cz)` | `tracking` |
| World-frame consistency across chunks | `deskew.py` writes SLAM-world coords | `lidar_preprocessing` |
| Geometry helpers (world↔body) | `wato_common.geometry.body_frame` (after Phase 1) | `wato_common` |
| Voxel downsample helper | `_voxel_downsample` in `lidar_preprocessing/reduce.py` | reuse via `wato_common` |

The merged world-frame property from `deskew.py` is critical: because all
three LiDARs are deskewed into the same SLAM world frame, the per-frame
crops naturally contain returns from all three sensors. Aggregation gets
~3× the point count for free vs. a single-LiDAR rig.

---

## How our multi-sensor rig improves on the papers

**3 LiDARs vs. paper assumption of 1**

3DAL was designed for Waymo (1 top LiDAR). DetZero benchmarks on Waymo and
nuScenes. With 3 Velodynes at different mounting heights and positions:
- Vehicle sides and corners that the centre LiDAR sees obliquely are seen
  near-perpendicular by the NE or NW LiDAR → much better edge definition in
  the aggregated cloud
- Pedestrians and cyclists in centre-LiDAR blind spots (<10 m, low angles)
  are picked up by side LiDARs → more frames contribute non-empty crops
- More points per frame means ICP is more stable and the
  consecutive-frame correction step is more accurate

**Long observation windows**

The user's bags are 30–150 seconds. A vehicle followed for 60 s at 10 Hz =
600 frames. Aggregated cloud is potentially ~30k points. We can afford
heavier downstream processing (LabelFormer with bigger encoder, longer
attention window) precisely because the input is dense.

---

## Gaps and what to do

### 1. Body-frame helper utilities (Phase 1, shared)

**What to build** in `src/common/src/wato_common/geometry/body_frame.py`:

```python
import numpy as np

def heading_to_rotation(heading: float) -> np.ndarray:
    """Yaw-only 3x3 rotation around world z-axis."""
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def world_to_body(
    pts_world: np.ndarray,      # (N, 3)
    center: np.ndarray,         # (3,)
    heading: float,
) -> np.ndarray:                 # (N, 3) in body frame
    R = heading_to_rotation(heading)
    return (pts_world - center) @ R   # R.T applied via right-multiply

def body_to_world(pts_body, center, heading) -> np.ndarray:
    R = heading_to_rotation(heading)
    return pts_body @ R.T + center

def enlarged_box_indices(
    pts_world: np.ndarray,
    center: np.ndarray, size: np.ndarray, heading: float,
    margin: float = 1.5,
) -> np.ndarray:                 # bool mask (N,) of points inside enlarged box
    pts_body = world_to_body(pts_world, center, heading)
    enlarged = size * margin
    return (
        (np.abs(pts_body[:, 0]) < enlarged[0] / 2) &
        (np.abs(pts_body[:, 1]) < enlarged[1] / 2) &
        (np.abs(pts_body[:, 2]) < enlarged[2] / 2)
    )
```

### 2. Per-track aggregation function

**What to build** in `src/label_refinement/src/wato_label_refinement/aggregate.py`:

```python
@dataclass
class AggregatedTrack:
    track_id: str
    points_body: np.ndarray         # (N_agg, 3) float32, body frame
    pose_history: list[FramePose]   # one per frame in the track
    n_frames: int
    aggregation_method: str         # "naive" | "icp_corrected"

@dataclass
class FramePose:
    sweep_id: int
    center_world: np.ndarray        # (3,) refined after ICP if applied
    heading: float                  # refined after ICP if applied
    n_points: int

def aggregate_track(
    track_rows: list[TrackRow],
    world_npz_resolver: Callable[[str, int], str],  # (chunk_id, sweep_id) → URI
    dynamic_mask_resolver: Callable[[str, int], str],
    cfg: AggregateConfig,
) -> AggregatedTrack:
    # 1. Per-frame crop in body frame using enlarged_box_indices().
    # 2. Concatenate.
    # 3. If cfg.icp_correction: run pairwise ICP between consecutive frame slices,
    #    update pose_history with corrected (center_world, heading).
    # 4. Voxel downsample.
    # 5. Statistical outlier removal (k=10, std=2).
    # 6. Return AggregatedTrack.
```

Re-uses `world_to_body` and `enlarged_box_indices` from Phase 1. Re-uses
`_voxel_downsample` from `lidar_preprocessing/reduce.py` after that helper
is promoted into `wato_common`.

### 3. ICP correction step (DetZero's geometric refining)

**What to build**: pairwise ICP between consecutive frame slices in body
frame using Open3D's `PointToPointICP`:

```python
def icp_correct_frames(
    frame_slices: list[np.ndarray],          # per-frame body-frame points
    initial_poses: list[FramePose],
    max_iterations: int = 30,
    max_correspondence_distance: float = 0.3,
) -> tuple[list[np.ndarray], list[FramePose]]:
    """Iteratively refine per-frame poses so consecutive slices align."""
    # 1. Take first frame as reference.
    # 2. For each subsequent frame: ICP onto the running aggregated cloud.
    # 3. The ICP transform is a small rigid correction; apply it to that
    #    frame's slice AND record the corrected (center_world, heading) so
    #    the LabelFormer pose head learns relative to corrected initial pose.
    # 4. Return corrected slices + corrected pose history.
```

**Why running aggregate, not previous-frame**: an object's near surface may
not be visible in two consecutive frames (e.g. self-occlusion as it turns).
ICP onto the running aggregate is more robust.

**When to skip ICP**: if a track has fewer than `cfg.icp_min_frames` (default
3) frames or if the initial pose jitter is tiny (`pose_jitter_m < 0.05`),
ICP is unnecessary and wastes compute.

### 4. Outlier removal

**Why needed**: even after dynamic_mask filtering, occasional ground-plane
bleed-through or partial occlusion of nearby objects contaminates the crop.

**What to do**: statistical outlier removal — for each point, compute
distance to its k nearest neighbours in the aggregated cloud; reject points
whose mean kNN distance is > μ + 2σ where (μ, σ) are computed over all
points. Standard Open3D helper.

### 5. Output schema (AggregatedTrackRow)

Per-track, write to `aggregated_tracks/<track_id>.npz`:

```
points_body       (N, 3)  float32   body-frame xyz
pose_history      (T, 7)  float64   per-frame [cx, cy, cz, heading, sweep_id, chunk_id, n_points]
                                    (after ICP correction if applied)
n_frames          int
class             str
voxel_size_m      float
icp_corrected     bool
```

Index entry in `aggregated_tracks_index.parquet`:

```
track_id, bag_id, n_frames, n_points_aggregated, points_path,
aggregation_method, cls
```

### 6. Failure modes and fallbacks

- **Empty track** (no dynamic points inside any frame's enlarged box):
  emit `aggregated_track` with empty points and mark
  `aggregation_method = "empty"`. LabelFormer falls back to per-frame
  encoding for these tracks.
- **Single-frame track**: aggregation is trivially the single frame's crop.
  Still produces a valid AggregatedTrack.
- **Pose jitter too large** (`pose_jitter_m > 1.0`): the upstream tracker
  is likely confusing two objects. Mark
  `aggregation_method = "high_pose_jitter"` and reduce LabelFormer
  confidence on the output.

### 7. Parallelism

Each track is independent — `ProcessPoolExecutor` per track, mirroring the
chunk-parallel pattern in `lidar_preprocessing/pipeline.py`. The bottleneck
is loading world NPZs; cache them in the worker process if multiple tracks
share frames (most tracks do).

---

## How the aggregated cloud feeds LabelFormer

The aggregated body-frame cloud replaces the per-frame crop as the *shape*
evidence. LabelFormer's modified architecture:

```
   per-track aggregated cloud  → AggregateEncoder (PointNet-style, R^256)
                                                                  │
   per-frame crop (existing)   → FrameEncoder    (R^256, per frame)
                                                                  │
                                       concat per frame            │
                                              │                    │
                                              ▼                    │
                                  Transformer self-attention       │
                                  (over T frames)                  │
                                              │                    │
                                              ▼                    │
                                    ┌────────────────┐             │
                                    │ Size head      │  ← reads only AggregateEncoder
                                    │ (W, L, H)      │             │
                                    └────────────────┘             │
                                    ┌────────────────┐             │
                                    │ Pose head      │  ← reads FrameEncoder ⊕ AggregateEncoder per frame
                                    │ (Δx,Δy,Δz,Δθ)  │             │
                                    └────────────────┘             │
```

The size head gets exactly one input per track (the aggregate embedding),
matching the physics: a vehicle doesn't change size. The pose head gets
per-frame information so it can correct frame-specific pose drift.

This is a small modification to the LabelFormer described in
`labelformer_guidance.md` — wire two encoders in parallel rather than one.

---

## Summary of actionable steps

1. Add `body_frame.py` to `wato_common.geometry` with the four helpers above
   (Phase 1).
2. Promote `_voxel_downsample` from `lidar_preprocessing/reduce.py` into
   `wato_common.geometry` (or a new `wato_common.pointcloud_ops.py`) so it
   is shared cleanly.
3. Add `AggregatedTrackRow` + `AGGREGATED_TRACK_SCHEMA` to
   `wato_common.schemas`.
4. Add `aggregated_track_path()`, `aggregated_tracks_index_path()` to
   `wato_common.artifact_store`.
5. Build `label_refinement/aggregate.py` with `aggregate_track()` +
   `icp_correct_frames()`.
6. Wire `aggregate.py` into `label_refinement/pipeline.py` as a step
   *before* the LabelFormer model runs. Use `ProcessPoolExecutor` per track.
7. Modify `label_refinement/model.py` to add `AggregateEncoder` as a sibling
   of `FrameEncoder`; route encoder outputs through the transformer as
   described above.
8. Per-track unit test: synthetic vehicle moving in a straight line with
   known size → assert aggregated cloud reconstructs box dimensions within
   5 cm; assert ICP correction reduces pose noise.
