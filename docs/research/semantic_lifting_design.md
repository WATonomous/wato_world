# semantic_lifting — UniLiPs-style 2D→3D label lifting

**Paper reference**: UniLiPs (light.princeton.edu/unilips), Section 3.2
(occlusion-aware semantic lifting) and Section 3.3 (geometry-grounded fusion).
**Component**: `src/semantic_lifting/`. The core algorithm (Steps 1–7 below)
is implemented. It lives between `perception_2d` + `lidar_preprocessing` and
`proposal_generation` in the pipeline DAG.

The `semantic_lifting` component takes per-camera 2D instance masks and dense
metric depth from `perception_2d`, plus LiDAR sweeps from
`lidar_preprocessing`, and produces per-LiDAR-point label assignments. Each
point that falls inside an object mask in one or more cameras — and passes a
depth-based visibility test — inherits that mask's `(instance_id, class)`,
aggregated across cameras with a confidence count.

This component implements UniLiPs Equation 1 (the occlusion-aware visibility
test) and the multi-camera `(label, count)` accumulation scheme described in
Section 3.3. UniLiPs' Algorithm 1 (probabilistic KD-tree propagation) is a
future addition here. `f_IWU` (iterative weighted update) was implemented
geometry-only in `lidar_preprocessing` Step E, because it operates on the bag
static map and feeds that component's motion proposals (see
`lidar_mos_guidance.md`). The one IWU term that needs this component is the
label consensus C(m) — see "Future extensions" below.

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
4. **It is the natural home for UniLiPs' probabilistic propagation** and for
   the accumulated `(label, count)` map that would feed IWU's consensus term
   back to `lidar_preprocessing`. Both operate on the accumulated point cloud,
   not on per-frame data.

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
                │  lidar_proc/*_world.npz (world xyz) │
                │  lidar_proc_index.parquet           │
                │  (+ ingest: calibration.json,       │
                │     frame_index.parquet)            │
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

All under `data/artifacts/raw/<bag_id>/`.

From perception_2d (`chunks/<chunk_id>/`):
- `masks_2d/<masklet_id>/<camera_seq:06d>.png` — binary instance masks
- `depth_2d/<cam>/<frame>.npz` — metric depth + confidence + coverage mask
- `tracklets_2d.parquet` — per-masklet class, score, frames present

From lidar_preprocessing (`chunks/<chunk_id>/`):
- `lidar_proc_index.parquet` — per-sweep `world_path` (and `dynamic_mask_path`,
  carried for the dynamic-point handling below, not yet read)
- `lidar_proc/<sweep_id:06d>_world.npz` — deskewed world-frame points

From ingest:
- `calibration.json` (bag level) — `ego_T_lidar`, `ego_T_cam_*`, `K_*` per camera
- `chunks/<chunk_id>/frame_index.parquet` — sweep timestamps and nearest-frame
  mapping per camera

---

## Outputs

```
data/artifacts/raw/<bag_id>/chunks/<chunk_id>/semantic_lifting/
├── lifted_labels/<sweep_id>.npz          (per-sweep, primary output)
└── lifted_stats.parquet                  (per-sweep diagnostics)
```

(A `camera_assignments/<sweep_id>.npz` debug artifact — which camera
contributed each label — was designed but is not written.)

### `lifted_labels/<sweep_id>.npz` schema

**As implemented**, the file is `lifted_labels/<sweep_id:06d>.npz` with
`point_indices`, `instance_ids` and `classes` (unicode strings; the instance id
is the masklet's `global_object_id`, else its `masklet_id`), `confidences`,
`n_supporting_cameras`, `n_disagreeing_cameras` — see
`wato_common.artifacts.lifted_labels`. The table below is the original design.

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
(default 0.05 s = 50 ms — half a camera period at 12 Hz). `t_sweep` is the
sweep's `reference_timestamp_ns` (lidar_preprocessing's index) and `t_frame`
the frame's `camera_timestamp_ns` (`frame_index.parquet`, one reference per
camera frame).

**As implemented**, lifting does no matching of its own: ingest already pairs
every sweep with each camera's nearest frame in `frame_index` (within its
`max_cam_offset_ms`, 60 ms on WATO, with a `camera_drop_reason` when it
doesn't), and `io.load_sweep_frames` takes that pairing as is. One owner for
the decision; a separate lifting window had drifted to 50 ms.

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

**As implemented**, the `world_T_ego(t_sweep) ∘ ego_T_lidar` half is already
done upstream: lidar_preprocessing writes every point in world coordinates,
each at its own measurement time. Lifting therefore only needs
`cam_T_world = inv(ego_T_cam) ∘ inv(world_T_ego(t_frame))`, with
`world_T_ego(t_frame)` looked up at `camera_timestamp_ns` through
`wato_common.pose_lookup` (`io.load_sweep_frames`). `frame_index.world_T_ego` is
the sweep's pose and is not used here: on the WATO rig it misplaced a point
30 m away by 17 cm median, up to 87 cm.

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

**As implemented**, D has two sources, and a point must pass both
(`visibility.py`; numbers in `docs/semantic_lifting_measurements.md`):

- perception_2d's depth map, min-pooled over the 7×7 patch, with `0` (fitted
  inverse depth ≤ 0) as +∞. Its single per-frame inverse-depth fit is off by a
  median 0.1 m at 5–10 m but 2 m at 20–30 m and 20 m at 30–50 m, so against it
  alone 39–62% of in-mask points beyond 10 m failed — including whole rows of
  parked cars. Only its surfaces nearer than `depth_map_max_range_m` (10 m)
  occlude.
- the sweep's own non-ground LiDAR returns, z-buffered into 8 px cells and
  min-pooled over 3×3 cells: `d ≤ z · 1.05 + 0.5 m`. Exact at any range, sparse
  (it misses most near occluders the map sees, hence both). This is the design's
  "LiDAR-only sparse-depth visibility" fallback, used everywhere.

In-mask points kept at 20–30 / 30–50 m: 37% / 25% → 84% / 65%.

### Step 5: mask lookup

For each visible (u, v), check which detection masks contain that pixel.
Multiple masks can overlap (a person inside a vehicle bounding box, etc.).
Order detections by `det_score` and take the *innermost* (smallest area)
mask containing the pixel. This handles the "person inside car" case
correctly — the person mask wins.

Record a vote:
```
vote = (instance_id, class_id, score = det_score × depth_confidence[u,v])
```

**As implemented**, `score = max(det_score, 1e-3)`: the depth artifact carries
no per-pixel confidence. Each frame's masks are painted once into a label
raster (largest first, so the innermost wins) and cached across the sweeps the
frame is paired with (`frames.py`).

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

**As implemented**, votes are reduced by **class** first, then instance:
masklets are per-camera and `global_object_id` is only assigned by `tracking`,
which runs after lifting, so voting on `instance_id` made two cameras on the
same car always disagree (on chunk 0000 of `may_30_ring_road_test_5_2`, every
one of 2,994 "disagreeing" points in 40 sweeps was a same-class pair). The
class with the highest summed score wins; within it, the highest-scoring
instance. `confidence = winner / (winner + runner-up class)`, 1.0 when
unanimous; `n_supporting_cameras` / `n_disagreeing_cameras` count cameras
voting the winning / another class. Unvoted points are not written.

**Cross-camera instances (as implemented, `instances.py`).** The design
assumed `global_object_id` from a cross-camera merge upstream; nothing assigns
it before lifting (`tracking` runs after), yet proposal_generation groups
points by instance to fit one box per object. So lifting merges masklets
itself, from geometry it already has: two same-class masklets from different
cameras that claim the same visible LiDAR points. Merge when, over the chunk,
they share ≥ `min_shared_points` (20) points that are ≥
`min_shared_fraction` (0.3) of the smaller one's claims where they overlap,
strongest pairs first, refusing a merge that would put two masklets one camera
saw in one frame — or two masklets whose claimed points are > `max_separation_m`
(5 m) apart (median over the sweeps both claim points in) — in one group.
`lifted_instances.parquet` maps each instance to its masklets and cameras.
Measurements: `docs/semantic_lifting_measurements.md`, "Instances".

### Step 7: write artifact

Pack into `lifted_labels/<sweep_id>.npz` with the schema above.

---

## Dynamic-point handling

lidar_preprocessing separates dynamic points upstream in two artifacts:
`lidar_proc/*_dynamic_mask.npy` (the `--seg` method's precision verdict) and
`lidar_proc/*_motion_proposals.npz` + `motion_clusters.parquet` (the
recall-oriented proposals, with per-cluster motion scores).

**For dynamic points**, the projection and visibility test still work, but
the temporal offset issue is more severe: a moving car's LiDAR points at
`t_sweep` will project to *where the car was* at `t_sweep`, but the camera
captured the car at `t_frame`, where the car is now displaced by
`(t_sweep - t_frame) × v_car`.

UniLiPs handles this implicitly through its iterative weighted update
(`f_IWU`) — they let dynamic points be initially misclassified, then catch
them via map-inconsistency. Our IWU lives in lidar_preprocessing (Step E) and
surfaces its floaters as `IWU_EVICTED` proposals, but it does not change the
labels lifted here. So for lifting itself, do this:

1. For dynamic points, additionally compensate for the object's velocity if
   it's known from the tracker output. Otherwise, accept the temporal slop.
2. Drop dynamic-point visibility-test pixels whose `depth_confidence` is
   low — these are usually at object edges, which is where the temporal
   error matters most.
3. Set `n_disagreeing_cameras ≥ 1` as a strong signal to drop the label
   (the label probably came from a stale pixel).

In v1, accept some noise on dynamic points and rely on SLF's L_lidar +
L_mask combination to resolve it during pose fitting. Measured on the WATO rig
(`docs/semantic_lifting_measurements.md`, "Dynamic points"): 12% of labeled
points are on moving clusters, displaced a median 0.1 m (p99 1.1 m) between
LiDAR and camera time; compensating with Step F track velocities is
directionally right but recovers ~5 points of in-mask rate on the fastest
movers only. It needs per-point times kept in the world NPZ and cluster
velocities exported by Step F (`tracking` runs after lifting, so it can't be
the source). Revisit when box fits on fast movers come out short or shifted. When the tracker's
velocities exist (item 1), `motion_clusters.parquet`'s per-cluster motion
features are the natural first signal for which points to compensate.

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

### Iterative Weighted Update (UniLiPs `f_IWU`, Eqs. 3-4) — consensus feedback

**The geometry is implemented** in `lidar_preprocessing` Step E (`iwu/`):
the update runs over the bag static map, evicted floaters become
`IWU_EVICTED` proposals, and the rule is adapted for sparse 32/16-beam
scanners. Details and measurements are in `lidar_mos_guidance.md`. UniLiPs
reports IWU carries the bulk of their 3D box quality (Table 8a ablation: mAP
31.0 → 11.7 without it).

**What remains for this component** is the label-consensus term
`C(m) = max_j n_mj / Σ_j n_mj`. It is 0 in Step E because lidar_preprocessing
has no semantics. After the multi-sweep aggregation below exists, export the
per-map-point `(label, count)` consensus and pass it to
`wato_lidar_preprocessing.iwu.update_with_sweep(..., consensus=...)`. That
makes both reinforcement (×(1+C)) and decay (×(1−C)) label-aware.

### Multi-sweep label aggregation

Currently each sweep produces an independent `lifted_labels/<sweep_id>.npz`.
A future addition is to aggregate labels across all sweeps in a chunk into
a single coherent map for downstream stages that operate on the accumulated
point cloud (label_refinement, ovd).

---

## Configuration schema

The shipped schema (authoritative: `src/semantic_lifting/config/semantic_lifting.yaml`
and its Pydantic model in `config.py`):

```yaml
visibility:
  tolerance_m: 0.5               # UniLiPs Eq. 1 τ
  neighborhood: 3                # depth map: min over the (2n+1)² patch
  depth_map_max_range_m: 10.0    # depth-map surfaces beyond this don't occlude
  lidar_cell_px: 8               # LiDAR occluder z-buffer cell
  lidar_neighborhood_cells: 1    # min over the (2n+1)² cells
  lidar_tolerance_per_m: 0.05    # occluder × (1 + this) + tolerance_m
instances:
  min_shared_points: 20
  min_shared_fraction: 0.3
  max_separation_m: 5.0
```

The design's `temporal.*` (ingest's `frame_index` pairing is used),
`voting.*` (innermost-wins and class-first voting are fixed; consumers filter on
`n_supporting_cameras`), `upstream_versions` (the manifest records inputs),
`projection.*`, `dynamic_points.*` and `outputs.*` keys are not part of it.

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
    ├── io.py                        Chunks; frame_index pairing with camera-time poses
    ├── pipeline.py                  Per-chunk orchestration
    ├── frames.py                    Per-frame pooled depth + mask raster, cached across sweeps
    ├── projection.py                cam_T_world, projection (wato_common.geometry), mask raster
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
- **Depth fit failed** (`FitStatus.FAILED`): can't run the visibility test
  reliably. Either skip this camera entirely or fall back to LiDAR-only
  sparse-depth visibility (project sparse LiDAR back, use as a sparse occluder
  mask). Implemented: skip the camera. perception_2d writes no `depth_2d`
  frame for a failed fit (`wato_common.artifacts.depth`), so the camera has no
  depth and is skipped; the planned `skip_on_fit_failure` switch was dropped
  because there is nothing on disk for it to act on.
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
4. `io.py` / `projection.py` — frame_index pairing, poses at camera time,
   cam_T_world.

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
  `det_score`.
- For dynamic points, should we offer a config to *skip* the
  ego-motion-only compensation and run with raw LiDAR coordinates? This
  would let us measure how much the compensation actually buys.
- Should `lifted_stats.parquet` include per-camera breakdowns (which camera
  contributed how many labels)? Useful for diagnosing under-performing
  cameras.

---

## Summary of actionable steps

Status: 1, 3–8 and 10–12 are done (4 landed as `wato_common/geometry/projection.py`). 2 is partial: `LiftedStatsRow` is in `wato_common.schemas`, but the lifted-labels NPZ has no row schema and the config model lives in the component's `config.py`. 9 is superseded.

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
9. ~~Wire into `config/pipeline.yaml`~~ — superseded: the component owns
   `src/semantic_lifting/config/semantic_lifting.yaml`.
10. Update `docs/research/segment_lift_fit_guidance.md` to describe SLF
    consuming the `lifted_labels` artifact (per-instance LiDAR subset
    selection, mask correspondence).
11. Update top-level README mermaid diagram to include the new component.
12. Add Dockerfile `docker/semantic_lifting.Dockerfile` — light on
    dependencies: NumPy/SciPy/Pillow/pyarrow, all from PyPI. ~~Open3D for
    KDTree~~ — dropped while nothing imports it; Algorithm 1 can use
    `scipy.spatial.cKDTree`, or re-add Open3D when it lands.
