# LabelFormer Alignment Guidance

**Paper**: "LabelFormer: Object Trajectory Refinement for Offboard Perception"
(arxiv 2311.01444)

LabelFormer refines noisy 3D bounding box trajectories produced by an upstream
detector+tracker.  It is the direct design blueprint for the `label_refinement`
component.

---

## What LabelFormer does

Auto-labeling pipelines — including ours — produce initial 3D box proposals
that are noisy in two ways: (a) per-frame pose jitter from imperfect detection,
and (b) inconsistent size estimates across frames due to varying point density.
LabelFormer fixes both by reasoning about the full trajectory at once.

```
upstream tracker output:
  track_id  frame  box_center  box_size  heading  lidar_points_in_box
      42      0    [x,y,z]    [w,l,h]    θ        (N0, 3) float
      42      1    [x,y,z]    [w,l,h]    θ        (N1, 3) float
      42      2    [x,y,z]    [w,l,h]    θ        (N2, 3) float
      ...
          │
          ▼
  ┌──────────────┐
  │  Frame       │   encode each frame's LiDAR crop independently:
  │  Encoder     │   PointNet-style encoder or sparse voxel encoder
  │  (per frame) │   output: frame_feature ∈ R^d  per frame
  └──────┬───────┘
         │  sequence of frame features  [f0, f1, f2, ...]
         ▼
  ┌──────────────┐
  │  Temporal    │   Transformer self-attention over the full trajectory
  │  Self-       │   No causal mask — every frame can attend to every
  │  Attention   │   other frame (because this is offboard / offline)
  └──────┬───────┘
         │  refined sequence of features
         ▼
  ┌──────────────┐
  │  Decoder     │   per-track: output one (W, L, H) shared across frames
  │              │   per-frame: output (Δx, Δy, Δz, Δθ) pose corrections
  └──────┬───────┘
         │
         ▼
  refined trajectory:
    track_id  frame  refined_center  refined_size  refined_heading
```

Key insight: **size is shared across frames** (a car does not change size) but
**pose is per-frame** (it moves).  The self-attention lets the model leverage
frames with dense LiDAR coverage (object nearby) to infer size that is then
applied to frames where the object is far away and point-sparse.

---

## What we already have that LabelFormer needs

| LabelFormer input | Our artifact | Where |
|-------------------|-------------|-------|
| Initial box proposals | `proposals.parquet` | `proposal_generation` output |
| Per-frame LiDAR points in box | `world/*.npz` + `dynamic_masks/*.npy` | `lidar_preprocessing` |
| Track IDs + frame associations | `tracks.parquet` | `tracking` output |
| World-frame coordinate system | deskewed xyz in consistent world frame | `lidar_preprocessing` |

The world-frame coordinate system from `deskew.py` is especially important:
because all LiDAR points are already in a single consistent world frame,
object positions across frames are directly comparable without an additional
ego-motion compensation step inside LabelFormer.

---

## Gaps and what to do

### 1. Upstream tracking (prerequisite)

LabelFormer cannot run until `tracking` produces `tracks.parquet`.  The minimum
schema needed:

```
tracks.parquet:
  track_id     str    (unique per bag)
  chunk_id     str
  sweep_id     int
  cx, cy, cz   float  (initial box center, world frame)
  w, l, h      float  (initial box size)
  heading      float  (radians, world frame)
  class_label  str    ("vehicle" / "pedestrian" / "cyclist")
  score        float  (upstream detector confidence)
```

Add this schema to `src/common/src/wato_common/schemas.py`.

### 2. Per-frame LiDAR crop extraction

For each (track_id, sweep_id) row in `tracks.parquet`, crop the LiDAR points
that fall inside the initial bounding box.  This needs to happen in
`label_refinement` before the model runs.

```python
def crop_box(world_npz_path, dynamic_mask_path, cx, cy, cz, w, l, h, heading):
    data = np.load(world_npz_path)
    mask = np.load(dynamic_mask_path)        # (N,) bool — True = dynamic
    xyz = np.stack([data["x"], data["y"], data["z"]], 1)[mask]
    # rotate to box-local frame, keep points within ±(w/2, l/2, h/2)
    R = heading_to_rotation(heading)         # (3,3)
    local = (xyz - np.array([cx,cy,cz])) @ R
    inside = (
        (np.abs(local[:,0]) < w/2) &
        (np.abs(local[:,1]) < l/2) &
        (np.abs(local[:,2]) < h/2)
    )
    return local[inside]                     # (M, 3) in box-local frame
```

With 3 LiDARs, the merged world-frame points already contain all three sensors'
returns — no extra multi-LiDAR logic needed.

### 3. Frame encoder

LabelFormer uses a lightweight PointNet-style encoder (MLP + max-pool) to
encode the per-frame crop into a fixed-size feature vector.

- Input: (N, 3) float32 in box-local frame (already zero-centered at box)
- Architecture: 3 → 64 → 128 → 256 → max-pool → R^256
- The box-local frame ensures the encoder is translation-invariant and that
  the model learns shape/density features rather than absolute position.

Alternatively: use a small sparse 3D CNN (MinkowskiEngine) on a 32³ voxel grid
around the object — this may generalize better for pedestrians who have fewer
LiDAR returns.

### 4. Temporal self-attention (Transformer)

Standard Transformer encoder (no causal mask) over the sequence of frame
features.  Positional encoding should be the normalized frame index or the
elapsed time since track start.

- Sequence length: number of frames in the track (typically 20–100 for a
  vehicle tracked across a 156-second bag at 20 Hz).
- Heads: 4–8; layers: 2–4; d_model: 256.  LabelFormer is intentionally small.
- Offboard advantage: because we process a complete recorded bag, we always
  have the full trajectory before running refinement — no need for causal
  attention or online inference tricks.

### 5. Decoder

Two output heads, both linear:
- **Size head**: one (W, L, H) per track (shared across all frames).
  Apply `log` to make sizes positive and log-normally distributed.
- **Pose head**: (Δx, Δy, Δz, sin(Δθ), cos(Δθ)) per frame.
  Add to the initial box center and heading from `tracks.parquet`.

Output `refined_labels.parquet`:
```
track_id  sweep_id  cx cy cz  w l h  heading  confidence
```

### 6. Training strategy

LabelFormer is trained on datasets with ground-truth trajectories.  Two options:

**Option A — Pretrain on public data, run inference only**
- Download the LabelFormer checkpoint (ONCE or nuScenes variant if available).
- Adapt the crop-and-encode pipeline to match what the checkpoint expects.
- Skip training entirely.

**Option B — Train from scratch on pseudo-labels (bootstrapping)**
- Iteration 0: run the full pipeline with a simpler refinement (rule-based
  or ICP-based size estimation) to produce rough labels.
- Train LabelFormer on these rough labels.
- Iteration 1: re-run refinement with the trained model → better labels.
- Repeat until convergence (2–3 iterations is typically enough).
- This is the "bootstrap → learned" cycle described in the `label_refinement`
  CLI docstring.

**Recommended**: start with Option A for speed; fall back to B if the domain
gap (NovAtel-based poses vs. Waymo SLAM) causes accuracy problems.

### 7. Parallelism

Each track is independent of every other track — embarrassingly parallel.
Use `concurrent.futures.ProcessPoolExecutor` (same pattern as `lidar_preprocessing`)
to process multiple tracks simultaneously.  Batching tracks by length and
running the Transformer in batched PyTorch inference will be significantly
faster than one-track-at-a-time.

### 8. Multi-LiDAR benefit

Three LiDARs give denser point clouds per frame, which means:
- More frames will have enough points (>5) to produce a useful crop feature.
- The frame encoder sees more surface detail → better size estimation.
- Objects in the NE/NW LiDAR blind spots of the center LiDAR are still
  observed by one of the side LiDARs → fewer frames with empty crops.

No model change needed — the merged world-frame points already contain all
three sensors.

---

## Summary of actionable steps

1. Define `TRACKS_SCHEMA` and `REFINED_LABELS_SCHEMA` in
   `src/common/src/wato_common/schemas.py`.
2. Add `tracks_path()` and `refined_labels_path()` to
   `src/common/src/wato_common/artifact_store.py`.
3. Build `label_refinement/crop.py`: `crop_box()` function using world NPZ
   + dynamic mask + initial box parameters.
4. Build `label_refinement/model.py`: frame encoder + Transformer + dual
   decoder, in PyTorch.
5. Build `label_refinement/infer.py`: load checkpoint, batch tracks by length,
   run model, write `refined_labels.parquet`.
6. Wire into `label_refinement/pipeline.py` with parallel track processing.
7. (Optional) Build `label_refinement/train.py` for the bootstrapping loop.
