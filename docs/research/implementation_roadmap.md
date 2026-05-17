# Accuracy-Upgrade Implementation Roadmap

This document is the in-repo continuation of the accuracy-upgrade work that
started with the perception_2d refactor.  It captures (a) what's already
landed, (b) what remains, in detailed enough form that any subsequent
contributor — including another agent session — can pick up where this one
left off without re-asking the same design questions.

The four research-paper-derived accuracy wins this initiative is built
around:

1. **Detector ensemble (MS3D++)** — multiple architecturally-diverse 3D
   detectors merged via Kernel-Based Fusion / Weighted Box Fusion.
2. **Segment-Lift-and-Fit (SLF)** — shape-aware 3D fitting from 2D SAM2
   masks via a PCA shape prior over SDFs.
3. **Depth Anything V2 pseudo-LiDAR** — densifies SLF's surface fit by
   lifting monocular metric depth into 3D world-frame points.
4. **Body-frame multi-frame aggregation (3DAL / DetZero)** — accumulates
   each track's LiDAR returns in its own coordinate frame before LabelFormer
   refines size and pose.

The matching paper-alignment docs live at
[docs/research/detector_ensemble_guidance.md](detector_ensemble_guidance.md),
[depth_anything_guidance.md](depth_anything_guidance.md),
[segment_lift_fit_guidance.md](segment_lift_fit_guidance.md),
[body_frame_aggregation_guidance.md](body_frame_aggregation_guidance.md), and
[cross_modal_uncertainty_guidance.md](cross_modal_uncertainty_guidance.md).
Cross-cutting overview: [pipeline_overview.md](pipeline_overview.md).

---

## Decisions captured (do not re-litigate without strong reason)

| Decision | Choice |
|---|---|
| Body-frame aggregation home | `label_refinement` (extends the existing stub) |
| Detector ensemble size | 3 LiDAR detectors: CenterPoint + DSVT + FSDv2 |
| Depth Anything caching | `perception_2d` writes `depth_2d/<cam>/*.npy` |
| SLF shape prior source | ShapeNet vehicles + KITTI/nuScenes bootstrap for ped/cyclist |
| Depth Anything variant | DA V2 Large (~335M params, ~1.3 GB checkpoint) |
| Checkpoint distribution | `MODELS_ROOT` volume mount + `watod fetch-models` |
| Cross-modal uncertainty scope | Diagnostic columns on `proposals.parquet` (no active gating in v1) |
| Tracking direction | Bidirectional (forward + backward + endpoint merge) |
| Open-vocab 2D detection | YOLO-World as parallel branch in `perception_2d` |
| Shape-prior build location | `src/proposal_generation/scripts/build_shape_prior.py` (standalone script) |
| 2D detector adapter source | DA V2 official GitHub repo; YOLO-World via ultralytics |
| Per-PR Dockerfile policy | Each component's Dockerfile is touched in its own PR — no pre-installing deps for stubs |

---

## Phase status

| Phase | Status | Land in |
|---|---|---|
| 0 — Research-alignment docs | ✅ Done | this PR |
| 1 — Shared infra (schemas, paths, geometry) | ✅ Done | this PR |
| 1 — Shared infra (watod fetch-models, perception_2d Dockerfile, MODELS_ROOT wiring) | ✅ Done | this PR |
| 2 — perception_2d refactor (YOLO-World + DA V2 + ensemble) | ✅ Done | this PR |
| 3 — Shape-prior build script | ⏳ Future | separate PR |
| 4 — proposal_generation implementation | ⏳ Future | separate PR (probably split into 2: detectors → SLF/fusion) |
| 5 — tracking implementation (bidirectional) | ⏳ Future | separate PR |
| 6 — label_refinement implementation | ⏳ Future | separate PR |
| 7 — Top-level docs + diagram + CLAUDE.md + component_versions bumps | ⏳ Future | folded into the last PR |
| 8 — End-to-end verification on a real bag | ⏳ Future | follows landing |

The remaining phases are detailed below.

---

## What's already landed (Phase 0-2 reference)

These changes are present at HEAD of this branch and shouldn't be touched
again by subsequent PRs (they form the assumed baseline):

### Phase 0 — research-alignment docs

- [docs/research/detector_ensemble_guidance.md](detector_ensemble_guidance.md)
- [docs/research/depth_anything_guidance.md](depth_anything_guidance.md)
- [docs/research/body_frame_aggregation_guidance.md](body_frame_aggregation_guidance.md)
- [docs/research/cross_modal_uncertainty_guidance.md](cross_modal_uncertainty_guidance.md)
- Updated [docs/research/pipeline_overview.md](pipeline_overview.md) with the new paper→component mapping and data flow.

### Phase 1 — shared infrastructure

- **Schemas** ([src/common/src/wato_common/schemas.py](../../src/common/src/wato_common/schemas.py)):
  - `ProposalRow` extended with `n_detectors_agreeing`, `ensemble_score_var`,
    `slf_dice_loss`, `slf_lidar_chamfer`, `slf_depth_chamfer`,
    `lidar_density_in_box`, `da_pixels_in_mask`, `uncertainty`.
  - New `DepthFrameRow` + `DEPTH_INDEX_SCHEMA` (perception_2d output index).
  - New `AggregatedTrackRow` + `AGGREGATED_TRACK_SCHEMA` (label_refinement
    body-frame aggregation index).
  - `TrackRow` extended with `direction` and `merged_from` (bidirectional tracking).
- **Artifact store** ([src/common/src/wato_common/artifact_store.py](../../src/common/src/wato_common/artifact_store.py)):
  - `depth_2d_dir`, `depth_2d_path`, `depth_index_path`
  - `proposal_diagnostics_path`, `pseudo_lidar_dir`
  - `tracks_forward_path`, `tracks_backward_path`
  - `aggregated_tracks_dir`, `aggregated_track_path`, `aggregated_tracks_index_path`
  - `detector_checkpoint_path`, `shape_prior_path`, `models_root`
- **Geometry** ([src/common/src/wato_common/geometry/body_frame.py](../../src/common/src/wato_common/geometry/body_frame.py)):
  - `heading_to_rotation`, `world_to_body`, `body_to_world`, `enlarged_box_indices`
  - Tests in [src/common/tests/test_body_frame.py](../../src/common/tests/test_body_frame.py).
- **watod fetch-models** ([watod_scripts/watod-fetch-models.sh](../../watod_scripts/watod-fetch-models.sh)) +
  dispatch in [watod](../../watod).  Downloads DA V2 Large + YOLO-World v2-L
  into `${WATO_WORLD_DIR}/data/models`.
- **perception_2d.Dockerfile** ([docker/perception_2d.Dockerfile](../../docker/perception_2d.Dockerfile)):
  uncommented `torch`/`torchvision`, added `depth-anything-v2` (git) +
  `ultralytics`.  SAM2 / GroundingDINO / DINOv2 lines remain commented.

### Phase 2 — perception_2d refactor

Files added:

- [src/perception_2d/src/wato_perception_2d/depth_anything.py](../../src/perception_2d/src/wato_perception_2d/depth_anything.py) — DA V2 estimator + `align_depth_to_lidar` helper.
- [src/perception_2d/src/wato_perception_2d/yolo_world.py](../../src/perception_2d/src/wato_perception_2d/yolo_world.py) — ultralytics-based adapter.

Files modified:

- [detector.py](../../src/perception_2d/src/wato_perception_2d/detector.py) — added `DetectorBase` Protocol + `DetectorEnsemble` wrapper.
- [pipeline.py](../../src/perception_2d/src/wato_perception_2d/pipeline.py) — `_build_detector` factory, new step B.5 depth computation, depth index parquet write.
- [config.py](../../src/perception_2d/src/wato_perception_2d/config.py) — new `DetectorEntry` and `DepthEstimatorConfig` sub-configs.
- [config/pipeline.yaml](../../config/pipeline.yaml) — populated `detectors` list and `depth_estimator` block.

Tests added:

- [test_depth_anything.py](../../src/perception_2d/tests/test_depth_anything.py) — 5 tests covering `align_depth_to_lidar`.
- [test_detector_ensemble.py](../../src/perception_2d/tests/test_detector_ensemble.py) — 8 tests covering IoU merging.
- [test_yolo_world.py](../../src/perception_2d/tests/test_yolo_world.py) — 2 tests for fallback behavior.

---

## Phase 3 — Shape-prior build script (SLF prerequisite)

### Goal
Produce `shape_prior_<class>.npz` under `$MODELS_ROOT/shape_priors/` for
vehicle / pedestrian / cyclist classes.  Each prior contains a mean SDF plus
a low-dimensional PCA basis that SLF's Adam fitter latches onto.

### Implementation sketch

- `src/proposal_generation/scripts/__init__.py` (empty marker — makes
  `python -m wato_proposal_generation.scripts.build_shape_prior` work).
- `src/proposal_generation/scripts/build_shape_prior.py`:
  ```python
  # CLI flags: --shapenet-root, --output, --classes, --pca-components, --voxel-grid
  # 1. For each vehicle mesh (ShapeNetCore "Car" synset id 02958343):
  #      - rescale to a canonical metric size
  #      - voxelize into a 64³ SDF using trimesh or PyTorch3D's `cubify`
  # 2. Stack all flattened SDFs; run sklearn.decomposition.PCA(n_components=20)
  # 3. Save shape_prior_vehicle.npz with mean_sdf, components, explained_var,
  #    class, voxel_size_m.
  # 4. For pedestrian / cyclist: sample (W,L,H) statistics from KITTI/nuScenes
  #    GT, parameterize as anisotropic ellipsoid + limb sub-blobs (k=5 PCA).
  ```
- Test in `src/proposal_generation/tests/test_build_shape_prior.py` — build a
  tiny prior from 3 synthetic boxes and assert reconstruction error is below
  threshold.

### Decisions for the implementer
- **Mesh library version**: ShapeNetCore v2.  License-gated — `watod
  fetch-models` does NOT auto-download (per Phase 1 rationale).  The build
  script's `--shapenet-root` flag points at a manually-downloaded copy.
- **Voxel grid resolution**: 64³ at 0.1 m voxel size = 6.4 m cube around
  vehicle origin.  Adequate for car-class size variation; cyclists / peds
  may need a smaller grid.
- **PCA components**: 20 for vehicle, 5 for ped/cyclist.  Empirically these
  capture > 95% variance for the respective classes.
- **Where to invoke it**: standalone, manual — `python -m
  wato_proposal_generation.scripts.build_shape_prior --shapenet-root
  /data/shapenet --output $MODELS_ROOT/shape_priors --classes vehicle
  pedestrian cyclist`.  Not part of `watod fetch-models` because of the
  license gate.

### Dependencies to add to `docker/proposal_generation.Dockerfile`
`trimesh`, `pyrender`, `scikit-learn`.

---

## Phase 4 — proposal_generation implementation

This is the biggest single phase; recommend splitting into two PRs:

- **Phase 4a — detector ensemble**: per-frame LiDAR detection + WBF fusion.
- **Phase 4b — SLF + pseudo-LiDAR + fusion + uncertainty**: shape-aware
  refinement using the Phase 3 shape priors and Phase 2 depth maps.

### Phase 4a — Detector ensemble

Files to add under `src/proposal_generation/src/wato_proposal_generation/detectors/`:

| File | Purpose |
|---|---|
| `__init__.py` | re-exports |
| `base.py` | `LidarDetector` `Protocol` + `DetectionBox` dataclass |
| `centerpoint.py` | OpenPCDet CenterPoint adapter |
| `dsvt.py` | OpenPCDet DSVT adapter |
| `fsdv2.py` | OpenPCDet FSDv2 adapter |
| `ensemble.py` | Weighted Box Fusion (KDE-clustering); emits `FusedBox` with `n_detectors_agreeing`, `ensemble_score_var` |
| `tta.py` | Optional test-time augmentation (flips + rotations) per detector |

Files to add at the package root:

| File | Purpose |
|---|---|
| `pipeline.py` (replace stub) | Chunk-parallel orchestration mirroring [src/lidar_preprocessing/src/wato_lidar_preprocessing/pipeline.py:151-240](../../src/lidar_preprocessing/src/wato_lidar_preprocessing/pipeline.py#L151-L240) |
| `config.py` (replace stub) | Full Pydantic schema for detector ensemble + SLF + fusion |
| `io.py` (extend stub) | Loaders for upstream artifacts (world NPZ, masks_2d, depth_index, etc.) |

Decisions for the implementer:

- **Adapter source**: prefer OpenPCDet (broader checkpoint catalog) over
  mmdet3d.  Decision deferred to implementation time — `base.py` should be
  a thin Protocol so the source is swappable later.
- **Class harmonization**: per
  [detector_ensemble_guidance.md §2](detector_ensemble_guidance.md), each
  adapter ships its own `CLASS_MAP` from native taxonomy to canonical
  `vehicle | pedestrian | cyclist`.
- **Heading convention**: every adapter outputs world-frame heading.  Add a
  per-adapter unit test that fabricates a constant-velocity synthetic object
  and asserts heading aligns with velocity direction.
- **Fusion algorithm**: start with Weighted Box Fusion (Solovyev et al.,
  2021).  Upgrade to KBF (paper exact) only if WBF profiling shows it
  matters.
- **Per-class confidence calibration**: ship identity calibration in v1.
  Calibration JSON lives at
  `$MODELS_ROOT/lidar_detectors/calibration_<name>.json` — TODO referenced
  in the guidance doc.

Dockerfile changes (this phase): add `mmdet3d` or OpenPCDet, plus their
build deps, to [docker/proposal_generation.Dockerfile](../../docker/proposal_generation.Dockerfile).

Tests:
- `tests/test_detector_ensemble.py` (NEW) — 3 mocked detectors, asserts
  `n_detectors_agreeing` count for known overlap patterns.
- `tests/test_pipeline.py` (NEW) — mirror
  [src/lidar_preprocessing/tests/test_pipeline.py](../../src/lidar_preprocessing/tests/test_pipeline.py).

### Phase 4b — SLF + pseudo-LiDAR + fusion + uncertainty

Files to add under `src/proposal_generation/src/wato_proposal_generation/slf/`:

| File | Purpose |
|---|---|
| `__init__.py` | re-exports |
| `shape_prior.py` | loads `shape_prior_*.npz` from `$MODELS_ROOT`; `expand_sdf(z) -> (64,64,64)` |
| `sdf_renderer.py` | PyTorch differentiable raycaster (torch.nn.Module) |
| `fitter.py` | Adam loop: L_mask + L_lidar + L_ground + L_depth |
| `multi_cam.py` | aggregate dice loss across visible cameras |

Files to add at the package root:

| File | Purpose |
|---|---|
| `pseudo_lidar.py` | Lifts DA depth + SAM2 masks → 3D world-frame points |
| `fusion.py` | MS3D++ style: matches ensemble ↔ SLF by BEV IoU; SLF shape + ensemble position |
| `uncertainty.py` | Computes the combined `uncertainty` score per
  [cross_modal_uncertainty_guidance.md §combination formula](cross_modal_uncertainty_guidance.md) |

Decisions:

- **PyTorch3D vs custom SDF kernel**: PyTorch3D's raycasters are tuned for
  triangle meshes, not SDF grids.  Implement `sdf_renderer.py` with plain
  torch operations (matmul + grid_sample) for v1.  Custom CUDA kernel only
  if profiling shows the Adam loop dominates wall time.
- **Per-class density priors**: vehicle 50, pedestrian 200, cyclist 100
  pts/m³ (from KITTI / nuScenes statistics).  Make configurable via
  `proposal_generation.yaml:uncertainty.class_density_priors` so the user
  can re-measure on their rig.
- **Loss weights** (start values):
  `λ_lidar=1.0, λ_ground=0.3, λ_depth=0.5` — DA pseudo-LiDAR is noisier
  than real LiDAR, so weighted lower.

Dockerfile changes (this phase): add `pytorch3d`, `trimesh` (already added
in Phase 3).

Tests:
- `tests/test_slf_fitter.py` — synthetic 2D mask + known SDF surface, fit
  converges to expected box within tolerance.
- `tests/test_pseudo_lidar.py` — synthetic depth + mask + scale, lifted
  points fall in expected box.
- `tests/test_fusion.py` — synthetic ensemble + SLF, assert match logic +
  provenance.
- `tests/test_uncertainty.py` — golden inputs, assert formula.

---

## Phase 5 — tracking implementation (bidirectional)

### Files

| File | Purpose |
|---|---|
| `src/tracking/src/wato_tracking/kalman_3d.py` | FilterPy 3D Kalman: `[cx, cy, cz, vx, vy, vz, w, l, h, heading, ω]` |
| `src/tracking/src/wato_tracking/association.py` | Hungarian matching with cost = 3D IoU + DINOv2 ReID cosine + class-mismatch penalty |
| `src/tracking/src/wato_tracking/bidirectional.py` | Forward + backward passes + endpoint merge |
| `src/tracking/src/wato_tracking/pipeline.py` (replace stub) | Bag-level orchestration |
| `src/tracking/src/wato_tracking/config.py` (replace stub) | Pydantic with Kalman noise, association weights, merge thresholds |
| `src/tracking/config/tracking.yaml` (new) | Per-class noise + association weights |

### Algorithm (bidirectional)

1. Load `proposals.parquet` for every chunk of the bag, sort by `sweep_id`.
2. **Forward pass**: instantiate `kalman_3d` per object; predict→associate→update.
   Write `tracks_forward.parquet` (every row tagged `direction="forward"`).
3. **Backward pass**: reverse the proposal list; run the same loop.  Write
   `tracks_backward.parquet` (`direction="backward"`).
4. **Endpoint merge**: at each forward-track's last frame, look for any
   backward-track whose first frame within ±N seconds matches by 3D IoU >
   threshold AND ReID cosine > threshold.  Hungarian-match them and emit
   merged rows (`direction="merged"`, `merged_from=[fwd_id, bwd_id]`).
5. Output `tracks.parquet` (bag-level) — union of {merged, forward-only,
   backward-only}.

### Decisions
- **Per-class process noise**: vehicle gets `σ_v=2 m/s`; pedestrian
  `σ_v=0.5 m/s`; cyclist `σ_v=1.5 m/s`.  Tunable via config.
- **Association cost weights**: IoU=1.0, ReID=1.0, class-mismatch=2.0.
  Hungarian library: `scipy.optimize.linear_sum_assignment`.
- **DINOv2 features**: read from `MaskletRow.dino_feature_path` written by
  perception_2d.  When a proposal is the union of multiple masklets
  (multi-camera), use the mean embedding.

Dockerfile (`tracking.Dockerfile`): add `filterpy`, `scipy`.

Tests:
- `tests/test_kalman_3d.py` — synthetic constant-velocity track.
- `tests/test_association.py` — synthetic predictions + observations,
  Hungarian matching.
- `tests/test_bidirectional.py` — chunked scene with one occlusion,
  forward+backward+merge recovers the full track.

---

## Phase 6 — label_refinement implementation

### Files

| File | Purpose |
|---|---|
| `src/label_refinement/src/wato_label_refinement/aggregate.py` | 3DAL/DetZero per-track body-frame aggregation with optional ICP correction |
| `src/label_refinement/src/wato_label_refinement/crop.py` | Per-frame crop helper (paper Eq. 2) |
| `src/label_refinement/src/wato_label_refinement/model.py` | FrameEncoder + AggregateEncoder + LabelFormer transformer |
| `src/label_refinement/src/wato_label_refinement/infer.py` | Load checkpoint, batch tracks by length, write `refined_labels.parquet` |
| `src/label_refinement/src/wato_label_refinement/pipeline.py` (replace stub) | Bag-level orchestration |
| `src/label_refinement/src/wato_label_refinement/config.py` (replace stub) | Pydantic for aggregation + model + inference |
| `src/label_refinement/config/label_refinement.yaml` (new) | Knobs |

### Algorithm

1. Load `tracks.parquet`.
2. **Per track (parallel via `ProcessPoolExecutor`)**: call
   `aggregate_track(...)` from
   [body_frame_aggregation_guidance.md](body_frame_aggregation_guidance.md):
   - Crop enlarged box (margin=1.5) per frame.
   - Transform to body frame via [world_to_body](../../src/common/src/wato_common/geometry/body_frame.py).
   - Concatenate, run optional pairwise ICP between consecutive frame
     slices (Open3D), voxel-downsample to 1 cm, statistical-outlier-remove.
   - Save `aggregated_tracks/<track>.npz`.
   - Write `aggregated_tracks_index.parquet`.
3. **Batched LabelFormer inference**: group tracks by length, run on GPU,
   write `refined_labels.parquet`.

### Model architecture

```
FrameEncoder       (PointNet on per-frame body-frame crop)  → R^256 per frame
AggregateEncoder   (PointNet on aggregated body-frame cloud) → R^256 per track
                                          │
                                          ▼
        per-frame: concat (frame_feat ⊕ aggregate_feat) → R^512
                                          │
                                          ▼
                  Transformer encoder (no causal mask) over T frames
                                          │
                                          ▼
                    Size head (reads aggregate only)      → (W, L, H) shared
                    Pose  head (reads concat per frame)   → (Δx, Δy, Δz, Δθ)
```

### Decisions

- **ICP correction**: enabled by default for tracks with ≥3 frames and
  pose_jitter > 0.05 m.  Open3D's `PointToPointICP` with max 30 iterations.
- **Fallback for empty/high-pose-jitter tracks**: emit
  `aggregated_track.npz` with `aggregation_method ∈ {"empty",
  "high_pose_jitter"}`.  LabelFormer falls back to per-frame encoder only
  for these.
- **Pretrained checkpoint vs train-from-scratch**: per
  [labelformer_guidance.md §training-strategy](labelformer_guidance.md),
  start with Option A (run inference using a public LabelFormer checkpoint
  if available, otherwise initialise from random and train on bootstrapped
  pseudo-labels).

Dockerfile (`label_refinement.Dockerfile`): add `Open3D`, plus optional
`MinkowskiEngine`/`torchsparse` for the sparse-CNN encoder variant.

Tests:
- `tests/test_aggregate.py` — synthetic track with known pose history,
  assert body-frame cloud is centered + ICP reduces residual error.
- `tests/test_model.py` — forward pass on synthetic input, assert shapes.
- `tests/test_crop.py` — synthetic box + points, expected subset.
- `tests/test_pipeline.py` — orchestration with mocked model.

---

## Phase 7 — Top-level docs + diagram + CLAUDE.md + versioning

### README.md (top-level)

- Mermaid diagram update: `proposal_gen` subgraph lists "MS3D++ detector
  ensemble · SLF · DA V2 pseudo-LiDAR · cross-modal fusion".  `label_ref`
  lists "body-frame aggregation · LabelFormer".  Add arrow
  `perception_2d -- "depth maps" --> proposal_gen`.
- Component table refresh.
- Add a "Models" subsection under Configuration explaining `MODELS_ROOT`
  and `watod fetch-models`.

### CLAUDE.md

Add sections:
- "Models and checkpoints" — `MODELS_ROOT` convention, `watod fetch-models` usage.
- "Shape priors" — how to rebuild via `build_shape_prior.py`; ShapeNet license note.
- "Detector ensemble gotchas" — heading drift across OpenPCDet models,
  class-name harmonization, per-class score calibration.
- "Pseudo-LiDAR caveats" — DA V2 needs scale alignment; what to do for
  cameras with little/no LiDAR overlap.
- "Body-frame aggregation troubleshooting" — coarse-track heading drift,
  why ICP correction matters, how to inspect `aggregated_tracks/*.npz`.
- Update "Build / test smoke check" with new module-specific pytest invocations.

### config/component_versions.yaml

Bump:
```
perception_2d:        v1 → v2   (adds depth_2d artifacts, YOLO-World detections)
proposal_generation:  v0 → v1   (first real implementation, new schema columns)
tracking:             v0 → v1   (first real implementation, bidirectional schema)
label_refinement:     v0 → v1   (first real implementation)
```

### config/pipeline.yaml

Per-component sub-sections matching each new Pydantic schema.

---

## Phase 8 — End-to-end verification

After Phases 3-7 land:

```bash
# Models (one-time after Phase 4's fetch-models entries land).
./watod fetch-models

# Build everything.
./watod build

# Build shape priors (manual, license-gated ShapeNet).
python3 -m wato_proposal_generation.scripts.build_shape_prior \
    --shapenet-root /data/shapenet --output $MODELS_ROOT/shape_priors \
    --classes vehicle pedestrian cyclist

# Run the full pipeline on the NuScenes mini bag (already exercised by
# ingest + lidar_preprocessing).
./watod run ingest --bag data/bags/NuScenes-v1.0-mini-scene-1100/
./watod run lidar_preprocessing --bag NuScenes_v1_0_mini_scene_1100
./watod run perception_2d --bag NuScenes_v1_0_mini_scene_1100
./watod run proposal_generation --bag NuScenes_v1_0_mini_scene_1100
./watod run tracking --bag NuScenes_v1_0_mini_scene_1100
./watod run label_refinement --bag NuScenes_v1_0_mini_scene_1100
```

Verification checklist:

- `depth_2d/<cam>/*.npy` exists for every camera frame; depth_index.parquet
  shows non-default scale/shift for cameras with LiDAR overlap.
- `proposals.parquet` has populated `n_detectors_agreeing`,
  `slf_dice_loss`, `slf_depth_chamfer`, `uncertainty` fields.
- `tracks.parquet`'s `direction` column shows
  `forward`/`backward`/`merged` representation; `merged_from` JSON
  well-formed where non-null.
- `aggregated_tracks/<track>.npz` exists for every track id;
  `aggregated_tracks_index.parquet` matches.
- `refined_labels.parquet` exists; spot-check that `w, l, h` are constant
  per `track_id` (size head shared across frames).

### Accuracy QA (baseline comparison)
- Run with `detector_ensemble: false` and `slf.enabled: false` →
  baseline `proposals.parquet`.
- Run with both enabled → upgraded `proposals.parquet`.
- Compare per-class recall + position variance + uncertainty distribution.
- Spot-check refined boxes vs LiDAR-frustum boxes on a handful of objects
  in `notebooks/` (rerun viewer).

---

## Risk register (carry forward)

1. **OpenPCDet vs mmdet3d**: pick at Phase 4a implementation time; `base.py`
   Protocol keeps the choice swappable.
2. **SLF SDF renderer perf**: PyTorch3D triangle raycasters don't fit SDFs.
   v1 uses plain torch (matmul + grid_sample).  Custom CUDA kernel only if
   profiling shows the Adam loop dominates.
3. **ShapeNet license**: research-use only.  `watod fetch-models` does NOT
   auto-download.  The shape-prior build script's `--shapenet-root` flag
   points at a manually-prepared copy.
4. **YOLO-World class taxonomy drift**: ultralytics' default class names
   don't match GroundingDINO's.  Harmonization via the existing
   `config/prompts.yaml` synonyms file.
5. **Bidirectional tracking memory**: doubles tracker memory.  If
   problematic on a multi-bag corpus, fall back to chunk-local forward +
   endpoint merge within `chunks_index.parquet:t_overlap_*_ns`.
6. **Depth cache disk usage**: ~30 GB / chunk for 12 cameras × 600 frames
   × 1920×1200 fp16.  Add `depth_2d_downsample_factor` config knob (default
   1) if disk-pressured.
7. **Tracker→aggregation handoff**: poor tracking → garbage aggregated
   clouds.  Gate aggregation on `n_frames ≥ 3` and pose-jitter threshold;
   fall back to per-frame LabelFormer for low-quality tracks.

---

## Out-of-scope (deferred or already documented)

- `open_vocab_discovery` — rare-class discovery branch.  Future iteration.
- `student_training` — distillation from auto-labels.  Future iteration.
- Marigold / UniDepth / DUSt3R — v1 is DA V2 Large only.
- 3D Gaussian Splatting / NeRF scene reconstruction.
- SAM4D-style cross-modal LiDAR mask propagation — partial (LiDAR-point
  prompts to SAM2 already implemented in `segmenter.py`).  Full cross-modal
  segmentation not in scope.
- ROS / runtime online integration — wato_world is offline batch only.
- Marigold-style diffusion depth — DA V2 is the v1 choice.

---

## Open decisions for next agent / contributor

These were *deferred* in this PR but will need a call when starting the
relevant phase:

| Decision point | Phase | Notes |
|---|---|---|
| OpenPCDet vs mmdet3d for detector adapters | 4a | Lean OpenPCDet (more checkpoints) — confirm at implementation time. |
| Per-detector calibration data source | 4a | Will probably need a labeled bag from the user's rig.  Until then, identity calibration is fine. |
| SLF: continuous SDF vs discretized voxel grid | 4b | Voxel grid (64³) is simpler and matches the shape-prior format. |
| LabelFormer: train from scratch vs pretrained transfer | 6 | Try pretrained transfer first (ONCE checkpoint if available); fall back to bootstrapping if domain gap shows. |
| Aggregation ICP library | 6 | Open3D is the obvious choice; torch-based ICP if Open3D install becomes a headache. |
