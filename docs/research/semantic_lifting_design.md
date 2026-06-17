# semantic_lifting — UniLiPs-style 2D→3D label lifting

**Paper reference**: UniLiPs (light.princeton.edu/unilips), Section 3.2
(occlusion-aware semantic lifting) and Section 3.3 (geometry-grounded fusion).
**Component**: NEW. Lives between `perception_2d` + `lidar_preprocessing` and
`proposal_generation` in the pipeline DAG.

The `semantic_lifting` component takes per-camera 2D instance masks and dense
metric depth from `perception_2d`, plus LiDAR sweeps from
`lidar_preprocessing`, and produces per-LiDAR-point label assignments. Each
point that falls inside an object mask in one or more cameras — and passes a
depth-based visibility test — inherits that mask's `(instance_id, class)`,
aggregated across cameras with a confidence count.

This component implements UniLiPs Equation 1 (the occlusion-aware visibility
test) and the multi-camera `(label, count)` accumulation scheme described in
Section 3.3. UniLiPs' downstream Algorithm 1 (probabilistic KD-tree
propagation) and `f_IWU` (iterative weighted update) are scoped as future
additions to this same component — see "Future extensions" below.

---

## Why this is a separate component

Could have lived in `perception_2d`, but doesn't, because:

1. **It joins two upstream sources** — perception_2d's masks/depth and
   lidar_preprocessing's static-map and dynamic-point artifacts. That's a
   natural DAG seam.
2. **It operates per-sweep, not per-camera-per-frame**. Perception_2d's unit
   of work is `(camera, frame)`. Semantic_lifting's unit is `(sweep)` — a
   single LiDAR sweep, projected into all cameras whose frames are
   temporally close.
3. **Its output is consumed by multiple downstream stages** — proposal_generation
   for SLF, tracking for cross-modal data-association cues, label_refinement
   for the final per-point labels. Producing it inside perception_2d would
   force every consumer to depend on perception_2d directly.
4. **It is the natural home for UniLiPs' probabilistic propagation and
   iterative weighted update** if those become needed. Both operate on the
   accumulated point cloud, not on per-frame data.

Could have lived in `proposal_generation`, but doesn't, because:

1. **It's needed by stages other than proposal_generation** (tracking,
   label_refinement, ovd).
2. **It's a clean perception primitive** — "which LiDAR points are inside
   which 2D object" — that doesn't depend on SLF's pose optimization at all.

---

## Architecture

```
                  ┌────────── perception_2d ──────────┐
                  │  masks_2d/    depth_2d/           │
                  │  detections_2d.parquet            │
                  │  tracklets_2d.parquet             │
                  └──────────────┬────────────────────┘
                                 │
                                 ▼
                ┌──────── lidar_preprocessing ────────┐
                │  world_T_ego per frame              │
                │  static map points                  │
                │  per-sweep dynamic points           │
                │  calibration.json                   │
                └──────────────┬────────────────────┘
                                 │
                                 ▼
                ┌──── semantic_lifting (THIS COMPONENT) ────┐
                │                                             │
                │  for each sweep:                            │
                │    for each LiDAR point:                    │
                │      for each camera at temporally          │
                │         nearest frame:                      │
                │        project point → (u, v, d_cam)        │
                │        visibility test:                     │
                │           d_cam ≤ depth_2d[u,v] + τ ?       │
                │        if visible AND inside mask:          │
                │          accumulate (instance_id, class,    │
                │            confidence)                      │
                │      reduce votes → final label             │
                │                                             │
                └──────────────┬────────────────────────────┘
                                 │
                                 ▼
                          lifted_labels/
                          <sweep_id>.npz
                                 │
                                 ▼
            ┌──── proposal_generation ────┐
            │  SLF reads lifted_labels   │
            │  for L_mask correspondence │
            │  and per-instance LiDAR    │
            │  subset selection          │
            └────────────────────────────┘
```

---

## Inputs

From `perception_2d/v2/<chunk_id>/`:
- `masks_2d/<cam>/<frame>/<det>.png` — binary instance masks
- `depth_2d/<cam>/<frame>.npz` — metric depth + confidence + coverage mask
- `detections_2d.parquet` — class, score, track_id, global_object_id per masklet
- `tracklets_2d.parquet` — temporal associations within camera streams

From `lidar_preprocessing/v1/<chunk_id>/`:
- `world/<frame>.npz` — `world_T_ego` per ego frame
- `static_map.npz` or per-sweep static points (depends on prior decision)
- `dynamic_points/<sweep>.npy` — moving-object LiDAR points per sweep
- `calibration.json` — `ego_T_lidar`, `ego_T_cam_*`, `K_*` per camera

From `ingest/v1/<chunk_id>/`:
- `frame_index.parquet` — sweep timestamps and nearest-frame mapping per camera

---

## Outputs

```
data/artifacts/raw/<bag_id>/semantic_lifting/v1/<chunk_id>/
├── lifted_labels/<sweep_id>.npz          (per-sweep, primary output)
├── lifted_stats.parquet                  (per-sweep diagnostics)
└── camera_assignments/<sweep_id>.npz     (debug: which camera contributed each label)
```

### `lifted_labels/<sweep_id>.npz` schema

| Field | Dtype | Shape | Description |
|---|---|---|---|
| `point_idx` | int32 | (N,) | Index into the sweep's point array |
| `class_id` | int32 | (N,) | Canonical class ID; 0 = background |
| `instance_id` | int64 | (N,) | global_object_id from cross-cam merge; -1 = background |
| `confidence` | float32 | (N,) | Normalized vote confidence ∈ [0, 1] |
| `n_supporting_cameras` | int8 | (N,) | How many cameras saw and labeled this point |
| `n_disagreeing_cameras` | int8 | (N,) | How many cameras saw it but assigned a different label |
| `visibility_min_depth_delta_m` | float16 | (N,) | Smallest d_da - d_cam across supporting cameras (debug) |

Only points inside at least one object mask are written with a non-background
label. Points outside all masks, or that failed the visibility test in every
camera, get `class_id = 0`, `instance_id = -1`. They are still written so
downstream stages know they were considered.

### `lifted_stats.parquet` schema

One row per sweep, for monitoring and debugging:

| Field | Type | Description |
|---|---|---|
| `sweep_id` | str | |
| `n_points_total` | int | Points in this sweep |
| `n_points_labeled` | int | Points with non-background label |
| `n_points_in_any_mask` | int | Points projecting into any mask in any camera |
| `n_points_failed_visibility` | int | Failed the depth occlusion test |
| `n_points_disagreement` | int | Cameras disagreed on label, resolved by majority |
| `mean_confidence_labeled` | float | Mean confidence over labeled points |
| `n_cameras_used` | int | Number of cameras actually queried (after frame matching) |

---

## Core algorithm

### Step 1: temporal matching

For each LiDAR sweep at time `t_sweep`, find the nearest frame from each
camera's stream. Reject cameras where `|t_sweep - t_frame| > max_offset_s`
(default 0.05 s = 50 ms — half a camera period at 12 Hz). Use
`frame_index.parquet` for this mapping.

### Step 2: ego-motion compensation (UniLiPs-borrowed)

Because cameras and LiDAR are not hardware-synchronized, the ego pose at
`t_sweep` differs slightly from the ego pose at the nearest camera frame
`t_frame`. For each candidate camera, compute the relative pose
`cam_T_lidar(t_sweep, t_frame)`:

```
cam_T_lidar = inv(world_T_ego(t_frame)) ∘ inv(ego_T_cam)
              ∘ world_T_ego(t_sweep) ∘ ego_T_lidar
```

This transforms a LiDAR point captured at `t_sweep` into the camera frame
captured at `t_frame`. The ~25 ms of ego motion is now compensated.

Note: this only compensates ego motion, not scene motion. Dynamic objects
will still project to slightly the wrong pixel by `Δt × v_object`. Mitigated
by handling dynamic points separately (see "Dynamic-point handling" below).

### Step 3: projection

Project the LiDAR point `p_lidar` into the camera using the intrinsic matrix
`K` and `cam_T_lidar`:

```
p_cam = cam_T_lidar @ p_lidar         # in camera frame
(u, v) = K @ p_cam / p_cam.z          # pixel coords
d_cam = p_cam.z                        # depth in camera frame
```

Reject the point for this camera if:
- `p_cam.z ≤ 0.5 m` (behind camera or too close)
- `(u, v)` outside image bounds
- `d_cam` outside `[depth_2d_min, depth_2d_max]` (sanity range)

### Step 4: visibility test (UniLiPs Eq. 1)

A LiDAR point at projected depth `d_cam` is considered visible at pixel
`(u, v)` if:

```
d_cam ≤ depth_2d[u, v] + τ_visibility
```

where `τ_visibility = 0.5 m` per UniLiPs. The intuition: if the LiDAR point's
depth is greater than what the dense depth map says is at that pixel, the
point is *behind* something visible to the camera and shouldn't inherit its
label.

UniLiPs uses a slightly stronger form involving the depth of pixel
neighborhoods to handle depth discontinuities at object edges:

```
d_cam ≤ min(D(N(u, v))) + τ_visibility
```

where `N(u, v)` is a small neighborhood (e.g., 3×3) around the pixel. Use
this form. Add a config option `visibility_neighborhood` (default 3).

### Step 5: mask lookup

For each visible (u, v), check which detection masks contain that pixel.
Multiple masks can overlap (a person inside a vehicle bounding box, etc.).
Order detections by `sam3_score` and take the *innermost* (smallest area)
mask containing the pixel. This handles the "person inside car" case
correctly — the person mask wins.

Record a vote:
```
vote = (instance_id, class_id, score = sam3_score × depth_confidence[u,v])
```

### Step 6: cross-camera vote reduction

For each LiDAR point that received votes from one or more cameras, reduce:

- If all votes agree on `instance_id`: assign that instance, confidence =
  mean of vote scores, `n_supporting_cameras` = count, `n_disagreeing` = 0.
- If votes disagree on `instance_id`: pick the instance with the highest
  cumulative score. Set `confidence = (winner_score - runner_up_score) /
  total_score`. Record `n_disagreeing_cameras` for diagnostics.
- If no votes (point projected outside all masks in every camera, or failed
  visibility in every camera): `class_id = 0`, `instance_id = -1`,
  confidence = 0.

### Step 7: write artifact

Pack into `lifted_labels/<sweep_id>.npz` with the schema above.

---

## Dynamic-point handling

Per `lidar_preprocessing/classify.py`, dynamic points (moving objects) are
already separated from static points in upstream artifacts.

**For dynamic points**, the projection and visibility test still work, but
the temporal offset issue is more severe: a moving car's LiDAR points at
`t_sweep` will project to *where the car was* at `t_sweep`, but the camera
captured the car at `t_frame`, where the car is now displaced by
`(t_sweep - t_frame) × v_car`.

UniLiPs handles this implicitly through its iterative weighted update
(`f_IWU`) — they let dynamic points be initially misclassified, then catch
them via map-inconsistency. We don't yet have f_IWU, so do this instead:

1. For dynamic points, additionally compensate for the object's velocity if
   it's known from the tracker output. Otherwise, accept the temporal slop.
2. Drop dynamic-point visibility-test pixels whose `depth_confidence` is
   low — these are usually at object edges, which is where the temporal
   error matters most.
3. Set `n_disagreeing_cameras ≥ 1` as a strong signal to drop the label
   (the label probably came from a stale pixel).

In v1, accept some noise on dynamic points and rely on SLF's L_lidar +
L_mask combination to resolve it during pose fitting. Plan to add `f_IWU`
later if dynamic-point labels prove unreliable enough to bottleneck label
quality.

---

## Future extensions (NOT in v1)

These are documented here so the architecture supports them later without
restructuring:

### Probabilistic label propagation (UniLiPs Algorithm 1)

After per-sweep lifting, accumulate labels across all sweeps in the chunk
into a single "labeled map" data structure (a point cloud with `(label,
count)` lists per point). Then diffuse labels through 3D space via a KD-tree
with Gaussian distance weighting:

```
for each point p_i in the accumulated map:
  neighbors = KDTree.query(p_i, radius=r)
  for each (label, count) in p_i.labels:
    for each p_j in neighbors:
      w_ij = exp(-||p_i - p_j||² / (2σ²))
      p_j.labels[label] += count × w_ij
```

UniLiPs uses `r = 0.2 m`, `σ = r / 2`. Output: every point has a smoothed
label distribution; assign argmax label as the final.

Add as `propagation.py`, called optionally after the per-sweep lifting.

### Iterative Weighted Update (UniLiPs `f_IWU`, Eqs. 3-4)

Removes "floaters" (moving objects mistakenly registered as static map
points) and detects moving objects from inconsistencies between scans and
the accumulated map. Update each map point's static probability across
sweeps based on:

- How often it has a nearest map-point hit
- A class-prior credibility factor
- A range-distance influence factor

Map points with static probability below threshold are evicted and
reclassified as moving. UniLiPs reports this is responsible for the bulk
of their 3D bounding-box quality (Table 8a ablation: mAP 31.0 → 11.7
without it).

Add as `iterative_update.py` if dynamic-point labels prove unreliable.

### Multi-sweep label aggregation

Currently each sweep produces an independent `lifted_labels/<sweep_id>.npz`.
A future addition is to aggregate labels across all sweeps in a chunk into
a single coherent map for downstream stages that operate on the accumulated
point cloud (label_refinement, ovd).

---

## Configuration schema

```yaml
# src/semantic_lifting/config/semantic_lifting.yaml
semantic_lifting:
  temporal:
    max_offset_s: 0.05              # reject cameras farther than this from sweep time
  visibility:
    tolerance_m: 0.5                # UniLiPs Eq. 1 τ
    neighborhood: 3                 # min over 3x3 patch in depth_2d
    require_depth_coverage: false   # if true, require lidar_coverage[u,v]=True
  projection:
    min_depth_m: 0.5
    max_depth_m: 250.0
  voting:
    overlap_resolution: innermost   # for nested masks, innermost wins
    confidence_min: 0.2             # drop labels below this
    disagreement_threshold: 0.4     # margin below which label is dropped
  dynamic_points:
    enabled: true
    drop_low_confidence_pixels: true
    confidence_threshold: 0.5
  outputs:
    save_camera_assignments: false  # debug artifact, expensive
    dtype: float16
  upstream_versions:
    perception_2d: v2
    lidar_preprocessing: v1
    ingest: v1
```

---

## File layout

```
src/semantic_lifting/
├── config/
│   └── semantic_lifting.yaml
└── src/wato_semantic_lifting/
    ├── __init__.py
    ├── cli.py                       Chunk-level CLI runner
    ├── config.py                    Pydantic config schemas
    ├── io.py                        Read perception_2d + lidar artifacts; write lifted_labels
    ├── pipeline.py                  Per-chunk orchestration
    ├── temporal_match.py            Sweep ↔ frame matching with ego-motion compensation
    ├── projection.py                LiDAR → camera projection (re-uses perception_2d's projection.py via wato_common)
    ├── visibility.py                Occlusion test (UniLiPs Eq. 1)
    ├── voting.py                    Cross-camera vote aggregation
    └── stats.py                     Per-sweep diagnostics
```

Shared projection logic: extract `project_lidar_to_image()` to
`src/common/src/wato_common/projection.py` so both `perception_2d` (for
depth-anchor pairs) and `semantic_lifting` (for label lifting) use the same
implementation. Single source of truth for the projection math.

---

## Edge cases and risks

- **Sweep with no temporally close frames**: e.g., a corrupted camera stream.
  Write empty `lifted_labels/<sweep_id>.npz` with all points = background.
  Log a warning.
- **Mask covering > 50% of image** (e.g., a building wall mistakenly tagged
  as "vehicle"): the visibility test still works, but the mask becomes
  essentially uninformative. Drop masks above an area threshold pre-lifting.
- **Depth fit failed** (`fit_status = 2` in `depth_2d` artifact): can't run
  the visibility test reliably. Either skip this camera entirely or fall
  back to LiDAR-only sparse-depth visibility (project sparse LiDAR back,
  use as a sparse occluder mask). Default: skip the camera, log it.
- **All cameras disagree on a point's label**: usually means the point is
  on an object boundary where masks differ. Drop the label rather than
  picking arbitrarily. Track in `lifted_stats` to monitor for systemic
  issues.
- **Performance**: a sweep has ~120k points × 12 cameras × per-point
  projection + lookup. The naive loop is O(N × C) per sweep. With ~100
  sweeps per chunk this is the dominant compute cost. Vectorize the
  projection in NumPy / PyTorch; do mask lookup with `cv2.remap` or a
  precomputed mask-ID raster per camera-frame.

---

## Testing strategy

**Unit tests:**

1. `projection.py` — round-trip a known 3D point through a known camera,
   verify pixel matches.
2. `visibility.py` — synthetic depth map with known occlusion, verify
   visibility test returns expected results.
3. `voting.py` — synthetic vote sets with known winners, verify reduction.
4. `temporal_match.py` — synthetic timestamps, verify correct frame
   matching and ego-motion math.

**Integration tests:**

1. Single sweep + single camera + 1 known object: verify points inside the
   object are correctly labeled.
2. Two cameras seeing the same object: verify vote agreement.
3. Two cameras assigning *different* classes: verify majority-vote
   resolution.
4. Occlusion case: an object behind another. Front object's mask in camera
   A, back object also has a mask in camera A but its LiDAR points are
   behind. Verify back-object LiDAR points get the back-object label, not
   the front-object's.

**End-to-end validation:**

- Visualize `lifted_labels` colored by class on the accumulated map for a
  full chunk. Should look like a semantic point cloud, with objects clearly
  delineated.
- For chunks where ground truth labels exist (e.g., manually annotated
  validation chunks), report per-class IoU.

---

## Open questions for implementation

- Should the visibility test use the full neighborhood `min(D(N(u,v)))` or
  just the central pixel? UniLiPs uses the neighborhood; this is more
  conservative but slower.
- Should `depth_confidence[u,v]` factor into the per-vote score, or only
  gate vote inclusion? Currently planned as a multiplicative factor on
  `sam3_score`.
- For dynamic points, should we offer a config to *skip* the
  ego-motion-only compensation and run with raw LiDAR coordinates? This
  would let us measure how much the compensation actually buys.
- Should `lifted_stats.parquet` include per-camera breakdowns (which camera
  contributed how many labels)? Useful for diagnosing under-performing
  cameras.

---

## Summary of actionable steps

1. Create `src/semantic_lifting/` directory structure mirroring
   `src/perception_2d/`.
2. Define `LiftedLabelRow` and `SemanticLiftingConfig` schemas in
   `src/common/src/wato_common/schemas.py`.
3. Add `semantic_lifting_dir()` and `lifted_labels_path()` to
   `src/common/src/wato_common/artifact_store.py`.
4. Promote `projection.py` from `perception_2d` to
   `src/common/src/wato_common/projection.py` for shared use.
5. Build `temporal_match.py` — sweep ↔ frame matching with ego-motion
   compensation.
6. Build `visibility.py` — UniLiPs Eq. 1 occlusion test, with
   neighborhood option.
7. Build `voting.py` — vote aggregation with overlap resolution and
   majority-vote disagreement handling.
8. Build `pipeline.py` orchestrator: loop over sweeps, project into
   matched cameras, run visibility + lookup + voting, write outputs.
9. Wire into `config/pipeline.yaml` with version `v1`.
10. Update `docs/research/segment_lift_fit_guidance.md` to describe SLF
    consuming the `lifted_labels` artifact (per-instance LiDAR subset
    selection, mask correspondence).
11. Update top-level README mermaid diagram to include the new component.
12. Add Dockerfile `docker/semantic_lifting.Dockerfile` — light on
    dependencies, mostly NumPy + Open3D for KDTree (preempting Algorithm 1
    addition later).
