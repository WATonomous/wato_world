# MS3D++ Detector Ensemble Alignment Guidance

**Paper**: "MS3D++: Ensemble of Experts for Multi-Source Unsupervised Domain
Adaptation in 3D Object Detection" (Tsai et al., arXiv 2308.05988, 2024).
Earlier work: MS3D (CVPR 2024).

MS3D++ describes how to combine multiple pretrained 3D object detectors into
an ensemble whose merged proposals beat any single detector. The original
paper frames this as domain adaptation, but the same ensemble-fusion
machinery (Kernel-Based Box Voting + per-class confidence calibration +
test-time augmentation) is exactly what we need for the LiDAR side of
`proposal_generation`. It is the design blueprint for our detector ensemble.

---

## What MS3D++ does

```
  per-frame dynamic point cloud
            │
   ┌────────┼────────┐
   ▼        ▼        ▼
  ┌────┐  ┌────┐  ┌────┐
  │CP  │  │DSVT│  │FSDv2│   3 architecturally diverse detectors
  └─┬──┘  └─┬──┘  └─┬──┘    (each with optional TTA: flips, rotations)
    │       │       │
    ▼       ▼       ▼
   box_set  box_set  box_set
    │       │       │
    └───┬───┴───┬───┘
        ▼       ▼
  ┌────────────────────┐
  │ Kernel Box Fusion  │   1. cluster boxes by BEV proximity
  │     (KBF)          │   2. weight each box by per-class score
  └─────────┬──────────┘   3. KDE-weighted average for center
            │              4. circular mean for heading
            │              5. record n_detectors_agreeing
            ▼
   fused per-frame box set  → proposals.parquet
```

**Why three diverse detectors and not four copies of CenterPoint.** Ensemble
gains come from *architectural* diversity, not raw count. Center-based
(CenterPoint), pillar-transformer (DSVT) and fully-sparse (FSDv2) make
different errors:
- CenterPoint over-confidently localizes small/near objects; heading drifts on
  trucks.
- DSVT is strong at long range with dense point clouds, weak when sparse.
- FSDv2 handles very dense urban scenes and very sparse rural scenes well, but
  is noisier on heading for cyclists.

Adding TransFusion as a fourth is empirically only marginal (TransFusion is
architecturally similar to DSVT), so we stop at three.

---

## What we already have that MS3D++ needs

| MS3D++ requirement | Our artifact | Where |
|---|---|---|
| Per-sweep dynamic point cloud | `dynamic_map.npz` | `lidar_preprocessing` output |
| LiDAR frame_id grouping (multi-LiDAR sync) | `frame_id` column in `lidar_proc_index.parquet` | `lidar_preprocessing` |
| Per-sweep static cloud (background filtering) | `static_map.npz` | `lidar_preprocessing` |
| Ground plane (filter ground-clipping false boxes) | `ground.npz` height grid | `lidar_preprocessing` |
| Calibration (intrinsics + extrinsics) | `calibration.json` | `ingest` |
| Proposal schema with `n_detectors_agreeing` | `PROPOSAL_SCHEMA` (after Phase 1 extension) | `wato_common.schemas` |

The `lidar_preprocessing` dynamic/static split is exactly what MS3D++ assumes
each detector consumes: a cleaned, motion-compensated, deskewed point cloud
in a single coordinate frame.

---

## How our multi-sensor rig improves on the paper

**3 LiDARs vs. paper's 1 (typically Waymo top LiDAR or KITTI single Velodyne)**

The paper's "single dense point cloud" assumption is satisfied by our merged
world-frame points (all 3 Velodynes deskewed into the same SLAM frame). This
gives us:
- ~3× point density on near objects → CenterPoint's localization variance drops
- FSDv2's blind-spot brittleness disappears (NE/NW Velodynes cover what
  center LiDAR misses)
- Each detector's confidence on a given object becomes more reliable, so
  per-class score calibration is more meaningful

**12 cameras (for downstream SLF + uncertainty bookkeeping)**

The ensemble itself is LiDAR-only — cameras enter through SLF in a separate
stage. But the ensemble's `n_detectors_agreeing` becomes a strong prior for
SLF: low-agreement boxes get higher SLF weight (camera evidence breaks ties).

---

## Gaps and what to do

### 1. Detector adapter interface

**What MS3D++ assumes**: each detector exposes `predict(points) -> list[box]`
with a uniform box format `(cx, cy, cz, w, l, h, heading, score, class)`.

**What to build** in `src/proposal_generation/src/wato_proposal_generation/detectors/base.py`:

```python
from typing import Protocol

@dataclass
class DetectionBox:
    cx: float; cy: float; cz: float
    w: float; l: float; h: float
    heading: float          # radians, world frame
    score: float
    cls: str                # canonical: "vehicle" | "pedestrian" | "cyclist"
    detector_name: str      # provenance

class LidarDetector(Protocol):
    name: str
    def predict(self, points_world: np.ndarray, calib: dict) -> list[DetectionBox]: ...
```

Each adapter (CenterPoint, DSVT, FSDv2) is its own file under `detectors/`,
loads its checkpoint lazily from `MODELS_ROOT/lidar_detectors/<name>.pth`,
and converts its native output to `DetectionBox`. Use OpenPCDet's model zoo as
the checkpoint source (one set of pretrained weights for nuScenes; train on
our own data later if domain gap is a problem).

### 2. Class harmonization

**Problem**: every model ships its own taxonomy. nuScenes models have 10
classes including "barrier" and "construction_vehicle"; KITTI models have 3.
Our canonical set is `vehicle | pedestrian | cyclist`.

**What to build**: a per-adapter `CLASS_MAP` dict mapping native class →
canonical class (or `None` to drop). Apply inside the adapter so the
ensemble only ever sees canonical names.

```python
# detectors/centerpoint.py
CLASS_MAP = {
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "trailer": "vehicle", "construction_vehicle": "vehicle",
    "motorcycle": "vehicle",       # arguable; coarsen for now
    "pedestrian": "pedestrian",
    "bicycle": "cyclist",
    "barrier": None, "traffic_cone": None,
}
```

### 3. Heading-convention normalization

**Problem**: KITTI uses heading = angle from camera x-axis. OpenPCDet uses
ego-frame heading. Our pipeline is world-frame. Mixing conventions silently
flips boxes 180°.

**What to build**: each adapter converts to world-frame heading at output.
The conversion uses `world_T_ego` (from `frame_index.parquet`) and the
detector's documented convention. Add a one-line unit test per adapter that
detects a constant-velocity synthetic object and asserts heading aligns with
velocity direction.

### 4. Test-time augmentation (TTA)

**What MS3D++ does**: each detector is run on the original cloud plus
augmented versions (x-flip, y-flip, ±90° rotations, ±15° random rotation).
Boxes are inverse-transformed back to the original frame, scores averaged,
heading averaged via circular mean. Doubles or triples per-detector recall
on long-tail orientations.

**What to build** in `detectors/tta.py`:

```python
def run_with_tta(
    detector: LidarDetector,
    points_world: np.ndarray,
    calib: dict,
    augmentations: list[str] = ["original", "flip_x", "flip_y", "rot_90", "rot_180"],
) -> list[DetectionBox]:
    all_boxes = []
    for aug in augmentations:
        aug_pts, T_aug = _apply_aug(points_world, aug)
        boxes = detector.predict(aug_pts, calib)
        for b in boxes:
            all_boxes.append(_invert_aug(b, T_aug))
    return _merge_tta(all_boxes, score_avg="mean", heading_avg="circular")
```

Optional knob in config: `detectors.<name>.tta_augmentations: [...]`. Default
to `["original"]` for speed; opt into TTA for high-quality runs.

### 5. Kernel-Based Box Fusion (KBF)

**What MS3D++ does**: rather than greedy NMS, treats each detector's box as a
Gaussian kernel centered on its position with covariance scaled by its
inverse score. Clusters by KDE peaks. Each cluster becomes a fused box whose
center is the KDE peak, size is score-weighted mean, heading is circular
mean, score is sum-of-scores normalized by class.

**What to build** in `detectors/ensemble.py`:

```python
def kernel_box_fusion(
    boxes: list[DetectionBox],
    bev_bandwidth_m: float = 1.5,        # KDE bandwidth in BEV
    min_n_agree: int = 1,                # drop singletons with low score
    per_class_weights: dict[str, float] = {"vehicle": 1.0, "pedestrian": 0.7, "cyclist": 0.7},
) -> list[FusedBox]:
    """Cluster by BEV proximity, fuse each cluster."""
    # 1. Partition boxes by class.
    # 2. Per class: KDE clustering on (cx, cy) with given bandwidth.
    # 3. Per cluster: weighted average of center, size, heading; sum-of-scores.
    # 4. Output FusedBox with n_detectors_agreeing and ensemble_score_var populated.
```

Simpler fallback: Weighted Box Fusion (WBF) from the 2D detection literature
(Solovyev et al., 2021) — same idea, simpler implementation. Start with WBF;
upgrade to KBF only if profiling shows it matters.

### 6. Per-class confidence calibration

**Problem**: CenterPoint's "0.7 vehicle" and DSVT's "0.7 vehicle" don't mean
the same thing. Naively summing scores gives whichever detector is more
confident-by-default extra weight.

**What to build**: per-detector per-class isotonic regression calibration
based on a held-out validation set. Calibrated score = `P(true positive |
detector says s)`. Save to `MODELS_ROOT/lidar_detectors/calibration_<name>.json`
as a piecewise-linear lookup.

For v1, ship with identity calibration and add a TODO. Calibration can be
fit later from a single labeled bag using `student_training`'s
ground-truth comparison.

### 7. Output to proposals.parquet

Each fused box becomes a row with:

- `provenance = "lidar_detector"`
- `n_detectors_agreeing` = cluster size at fusion time
- `ensemble_score_var` = variance of detector scores in the cluster
- `score` = calibrated fused score
- `lidar_point_count` = number of points inside the box (queryable signal)

These fields are added to `PROPOSAL_SCHEMA` in Phase 1.

---

## Summary of actionable steps

1. Add `PROPOSAL_SCHEMA` extensions for `n_detectors_agreeing`,
   `ensemble_score_var`, and the SLF/uncertainty fields described in
   `cross_modal_uncertainty_guidance.md`.
2. Build `detectors/base.py` with the `LidarDetector` Protocol and
   `DetectionBox` dataclass.
3. Build `detectors/centerpoint.py`, `detectors/dsvt.py`, `detectors/fsdv2.py`
   as adapters over OpenPCDet checkpoints. Each enforces canonical class
   names and world-frame heading.
4. Build `detectors/tta.py` for optional test-time augmentation.
5. Build `detectors/ensemble.py` with WBF (start) / KBF (optimization)
   fusion that emits `FusedBox` entries with provenance.
6. Wire into `pipeline.py` per-LiDAR-frame loop: run each detector → optional
   TTA → ensemble fusion → write rows. Use `ProcessPoolExecutor` (matching
   `lidar_preprocessing/pipeline.py`) for chunk-level parallelism; detectors
   share GPU within a worker.
7. Per-detector unit test fabricating two synthetic boxes and asserting the
   adapter outputs match within tolerance.
8. Ensemble integration test with three mocked detectors and an asserted
   `n_detectors_agreeing` count for known overlap patterns.
