# lidar_preprocessing

lidar_preprocessing is the first compute-heavy stage of the auto-labeling
pipeline. It takes the raw per-sweep `.npz` files produced by ingest — each
one a snapshot of the world from a moving sensor at a single moment in time —
and turns them into a spatially coherent, semantically partitioned point cloud
that every downstream stage can query without needing to know about ego motion,
sensor geometry, or the temporal structure of LiDAR acquisition.

## Why this stage exists

Ingest decodes raw sensor data faithfully: each LiDAR sweep is stored exactly
as the sensor reported it, in the sensor's own coordinate frame, at the sensor's
own timestamp. That is the right thing for ingest to do — it is a faithful
archive of the recording. But raw sweeps are not usable by a 3D detector or a
tracker for three reasons:

1. **Motion distortion.** A rotating LiDAR takes ~100 ms to complete one
   revolution. During that time the vehicle moves. The first point in the sweep
   and the last are measured at different ego positions, so a stationary pole
   appears banana-shaped in sensor frame. This is called scan distortion or skew.

2. **Sensor-relative coordinates.** Each sweep lives in its own sensor frame,
   centred on the LiDAR at the moment of the sweep. Comparing two sweeps
   requires knowing the ego pose at each sweep's time and the sensor-to-ego
   extrinsic transform. Downstream components should not have to reason about
   this.

3. **No separation of foreground and background.** Moving objects (cars,
   cyclists, pedestrians) and static structure (buildings, road surface, kerbs)
   are mixed together. A 3D detector that receives 30 seconds of accumulated
   LiDAR without this separation sees every pedestrian as a 30-metre smear
   through the scene. A ground extractor that receives raw sweeps including
   moving vehicles can mistake a low-riding car for the ground plane.

lidar_preprocessing fixes all three. Its outputs are the stable spatial
foundation that stages 3–6 are designed to consume.

## Processing steps

```
ingest artifacts
    │
    ▼
A. deskew.py          per-sweep motion compensation + world-frame projection
                      (also runs Patchwork++ in sensor frame; ground mask
                      stored as a key inside each world NPZ)
    │
    ▼
B. classify.py        voxel-based static / dynamic decomposition
    │
    ▼
C. ground.py          aggregate per-sweep ground masks → height grid
    │
    ▼
D. reduce.py          [separate command] bag-level global static map
```

---

### Step A — Deskew and project (`deskew.py`)

**What it does.** For each LiDAR sweep, every point is transformed from the
sensor frame at the point's individual measurement timestamp into the SLAM world
frame. The output is a per-sweep world-frame `.npz` that downstream stages read
without any knowledge of ego motion or sensor geometry.

**Why per-point, not per-sweep.** The naive fix for motion distortion is to
transform all points using the ego pose at the sweep's header timestamp. This
corrects inter-sweep misalignment but leaves intra-sweep distortion intact:
points at the start and end of the revolution are still measured at different
ego positions. The correct fix is to assign each point its own timestamp
(`t_offset_us` from the raw NPZ, if the LiDAR provides it) and interpolate the
ego pose at that specific time. This is sometimes called "undistortion" or
"deskewing" and is standard practice in high-quality 3D object detection.

**What `batch_interpolate_poses` does.** Interpolating pose per-point naively
would require one binary search per point over the pose samples. For a 100k-point
sweep with ~100 pose samples, that is 100k Python-level binary searches. Instead,
`batch_interpolate_poses` (in `wato_common.geometry.interpolation`) does one
`np.searchsorted` call over all unique per-point timestamps at once, then
vectorises the SLERP across the entire sweep. This is O(N log M) with a small
constant and processes a full sweep in milliseconds.

**Fallback when `has_point_time` is False.** Not all LiDARs provide per-point
timestamps (NuScenes LiDAR does not, for example). When `has_point_time` is
False, all points in the sweep are projected using the sweep's header timestamp
— no deskewing, but the world-frame transform is still applied correctly. The
`deskewed` field in `lidar_proc_index.parquet` records which path was taken.

**Coordinate precision.** World-frame coordinates are stored as `float64`. At
1 km from the SLAM map origin, `float32` would introduce ~8 cm of quantisation
error. `float64` reduces that to ~0.1 mm at any realistic drive length.

**Calibration.** The `ego_T_lidar` extrinsic (sensor-to-ego rigid transform) is
read from `calibration.json` written by ingest. If that entry is null (ingest
could not resolve the `/tf_static` chain), deskew raises an error for that
lidar rather than silently applying the wrong transform.

**Outputs per sweep** (`lidar_proc/<sweep_id:06d>_world.npz`):

| Field | Dtype | Description |
|---|---|---|
| `x`, `y`, `z` | float64 | World-frame coordinates (SLAM map frame) |
| `intensity` | float32 | If present in raw sweep |
| `ring` | uint16 | If present in raw sweep |
| `sweep_id` | int32 scalar | For cross-referencing |
| `reference_timestamp_ns` | int64 scalar | Sweep header timestamp |

**Metadata per sweep** (`lidar_proc_index.parquet`, ProcessedSweepMeta):

| Field | Type | Description |
|---|---|---|
| `bag_id`, `chunk_id`, `sweep_id`, `lidar_id` | str/int | Identity |
| `reference_timestamp_ns` | int64 | Sweep timestamp (ns) |
| `world_path` | str | URI to world-frame NPZ |
| `dynamic_mask_path` | str | URI to per-point dynamic mask |
| `n_points_total`, `n_points_static`, `n_points_dynamic` | int32 | Point counts |
| `world_xmin`, `world_xmax`, `world_ymin`, `world_ymax`, `world_zmin`, `world_zmax` | float | Bounding box in world frame |
| `has_intensity`, `deskewed` | bool | Feature flags |
| `frame_id` | int64 (nullable) | Canonical-frame grouping per `frame_sync` config. When `canonical_lidar=null`, each lidar's sweeps are numbered sequentially. When set, non-canonical sweeps within `±tolerance_ms` inherit the canonical sweep's frame_id; orphans are null. Consumed by `perception_2d` for SAM4D-style multi-modal frame fusion. |

The `world_*min/max` columns are written by deskew and used by classify to
validate per-sweep coordinate ranges. Missing or zero values in these columns
cause classify to raise an error, preventing silent misclassification of
malformed data.

---

### Step B — Voxel classify (`classify.py`)

**What it does.** Treats the entire set of world-frame sweeps for a chunk as a
4D occupancy volume and classifies every point as belonging to the static
background or to a dynamic (moving) object. The output is a per-sweep boolean
mask (`True` = dynamic) and a per-chunk accumulated static cloud.

**The core idea.** If a voxel in world-space is occupied by a point cloud return
in nearly every sweep, the thing occupying it is not moving — it is a building,
a parked car, the road surface, a kerb. If a voxel is occupied in only a handful
of sweeps, the thing that was there was moving through the scene. The threshold is
configurable (`static_sweep_fraction`, `static_sweep_min`) but 30 % of sweeps is
a robust default: a vehicle at 30 km/h traverses a 0.15 m voxel in ~18 ms, which
at 10 Hz corresponds to less than 1 sweep.

**Two-pass memory management.** At 10 Hz over a 30-second chunk, a naive
approach that holds all 30M world-frame points in memory simultaneously would
require ~720 MB of float64 arrays. The implementation uses two passes:

- **Pass 1**: load each sweep's world-frame NPZ one at a time, discretise to voxel
  keys (packed int64), and record which sweep indices occupied each voxel. Only
  keys and sweep-index sets are kept in memory — the raw coordinates are not.
- **Pass 2**: load each sweep again, look up each point's voxel key in the static
  set, write the boolean mask, and append static-classified points to the
  accumulating static cloud.

**Voxel key encoding.** Each voxel `(vx, vy, vz)` is encoded into a single
`int64` as `vx << 40 | vy << 20 | vz` (20 bits per axis), supporting a
±157 km range per axis at 0.15 m resolution. This avoids Python dict overhead
on string or tuple keys and is directly compatible with `numpy.unique`. The
voxel-key packing logic is factored into a shared module `voxel.py` and
imported by both `classify` and `ground` to avoid layering violations.

**Loud failures on stale artifacts.** If the `static_map.npz` from an earlier
run contains missing `world_*min/max` bounding-box columns, classify now raises
a clear error with guidance to delete the stale artifact and re-run. This
prevents silent misclassification of sweeps that lack valid bounding-box metadata.

**Why not UniLiPs / ray casting.** The correct implementation of the monorepo's
online dynamic removal uses ray-casting through the voxel grid: for each sweep,
every ray from the sensor to a measured point marks all voxels along the ray as
free. Voxels that were previously marked occupied but are now marked free must
contain a dynamic object. This is more precise but requires storing and iterating
a full ray set per sweep — expensive to implement and difficult to validate
without ground-truth labels. The occupancy-consistency approach is equivalent for
the downstream goal: moving objects traverse a voxel in far fewer sweeps than
static structure, and the resulting static cloud is accurate enough for ground
extraction and proposal scoring.

**Outputs:**

| Artifact | Description |
|---|---|
| `lidar_proc/<sweep_id:06d>_dynamic_mask.npy` | `bool[N]`, True = dynamic point |
| `static_map.npz` | Accumulated static cloud: `xyz` (float64, M×3), `intensity`, `voxel_size`, `origin` |
| `dynamic_map.npz` | Accumulated dynamic cloud: `xyz` (float64, M×3), `sweep_id` (int32, M), `intensity` (when present). Symmetric to `static_map.npz` so downstream (`proposal_generation`'s LiDAR detector, SLF candidate seeding) loads one artifact per chunk instead of iterating every sweep. |
| `voxel_occupancy.npz` | Sparse int32 voxel coords (`coords`, `origin`, `voxel_size`) for SAM4D / MinkUNet encoders in `perception_2d`. Includes all occupied voxels — static and dynamic. Toggle via `save_voxel_occupancy` config (default: true). |
| `lidar_proc_index.parquet` | Updated with `n_points_static`, `n_points_dynamic`, `dynamic_mask_path` per sweep |

---

### Step C — Ground extraction (`ground.py`)

**What it does.** Aggregates the per-sweep ground masks that Step A wrote into
the world NPZs, then builds a 2D height grid and surface-normal grid over the
extent of the chunk.

**Where Patchwork++ actually runs.** Patchwork++ runs *per sweep* inside Step A
(`deskew.py`), on sensor-frame xyz, before the world-frame transform is
applied. Step C does not call Patchwork++ — it just unions the per-sweep ground
masks into a chunk-level ground cloud. This mirrors the monorepo's
`patchwork::GroundRemovalCore` which also operates on sensor-frame
PointCloud2 messages one at a time. Three reasons we kept the per-sweep
formulation rather than running Patchwork++ once on the accumulated static
cloud:

- *Patchwork++'s zone model is sensor-centric.* The Concentric Zone Model
  (CZM) divides the field of view into concentric annular zones around the
  sensor origin. The `sensor_height` parameter and the per-zone seed selection
  both assume the cloud is centred on the lidar at a known height above the
  ground plane. An accumulated static cloud is centred on the SLAM map origin
  (anywhere from a few metres to several kilometres from any individual sensor
  position), and points at very different ranges relative to the *current*
  ego pose end up in the wrong zones. The library works in sensor frame; we
  follow that.
- *Per-sweep parallelism.* Running Patchwork++ inside the deskew loop means
  the per-sweep work (load NPZ → run Patchwork → transform → save NPZ) is one
  pass over the data. Running it again in Step C on the static cloud would
  add a second I/O + compute pass over essentially the same points and force
  ground extraction to wait for the full chunk before starting.
- *Algorithm parity with the monorepo.* The runtime perception stack runs
  Patchwork++ per sweep too. Keeping the offline pipeline aligned makes it
  cheap to swap parameters between online and offline and to port the same
  failure-mode lessons in either direction.

The trade-off is real and worth naming: per-sweep means lower point density at
distance (~90 m), and we do see returns from dynamic vehicle undersides
contributing to the ground mask before classify strips them. We accept this
because the chunk-level aggregation in Step C averages over hundreds of
sweeps — a single sweep's noisy ground call does not survive into the height
grid.

**Dependency on Step B.** Ground extraction requires the `static_map.npz` from
Step B (classify) to exist. This file contains the voxel keys marking which
world-space voxels were classified as static. If `static_map.npz` is missing,
ground raises a clear `FileNotFoundError` directing you to re-run classify for
the chunk. This enforces the pipeline dependency without silent failures.

**Future improvement: ground-dynamic intersection.** If profiling later shows
the height grid is the limiting accuracy factor, the cleanest follow-up is to
*intersect* per-sweep ground masks with the static-voxel mask from classify,
dropping any "ground" point whose voxel ended up classified dynamic — see
*Possible follow-ups* below.

**Patchwork++.** The algorithm (from `url-kaist/patchwork-plusplus`) uses a
Concentric Zone Model (CZM): the sensor field of view is divided into concentric
annular zones, and within each zone an iterative plane-fitting with outlier
rejection estimates the local ground plane. This handles non-flat terrain (ramps,
banked roads, crowned lanes) far more robustly than a single global RANSAC fit
would.

The Python interface used here (`pypatchworkpp==1.0.4` from PyPI) wraps the same
C++ library at the same version tag as the monorepo's
`patchworkpp_vendor` package. The algorithm is identical; the only difference is
the interface layer (Python bindings via pybind11 instead of a ROS 2 lifecycle
node subscribing to PointCloud2 topics).

**Height grid.** The ground point cloud from Patchwork++ is rasterised into a 2D
grid at `ground_cell_size_m` resolution (default 0.5 m). Each cell stores the
median Z of all ground points within it. Empty cells are filled by
nearest-neighbour from populated cells (using `scipy.ndimage.distance_transform_edt`).
Surface normals are estimated per cell from the finite-difference gradient of the
height grid (`np.gradient`).

**Downstream use of the height grid.** The height grid is a queryable function
`z_ground(x, y)` for the whole chunk. Stage 4 (proposal_generation) uses it to:

- Constrain the bottom face of proposed 3D boxes to within `th_dist` of the
  ground surface (boxes that float or clip through the ground are penalised).
- Determine the height above ground when lifting 2D camera masks into 3D via
  Segment-Lift-Fit.

Stage 6 (label_refinement) uses `residual_lidar_fit` as a learned quality
signal: a well-fit box should have its bottom face near the height-grid surface.

**The non-ground static cloud.** Points classified as non-ground (buildings,
barriers, lane markings, signs) are definitively not detectable objects. Stage 4
can use them as hard negative space when scoring proposals.

**Outputs** (`ground.npz`):

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `height_grid` | H×W | float32 | Ground Z at each grid cell (world frame) |
| `normal_grid` | H×W×3 | float32 | Unit surface normals |
| `grid_origin` | (2,) | float64 | [x₀, y₀] lower-left cell, world frame |
| `cell_size` | scalar | float32 | Cell size in metres |
| `ground_xyz` | M×3 | float64 | Raw Patchwork++ ground points |

---

### Step D — Bag-level reduce (`reduce.py`)

**What it does.** After all chunks for a bag have been processed by steps A–C,
the `reduce` subcommand merges per-chunk artifacts into two bag-level outputs:

- `global_static_map.npz` — concatenates every chunk's `static_map.npz` and
  voxel-downsamples to ~30 cm resolution via numpy voxel snap (quantize to grid
  cells, keep unique points).
- `global_ground.npz` — concatenates every chunk's `ground_xyz` and rebuilds
  a single bag-level height grid + normal grid via the same `_build_height_grid`
  helper Step C uses. This solves SLF's `L_ground` chunk-boundary problem in
  `proposal_generation`: a box fit near a chunk seam can query `z_ground(x, y)`
  over the full bag without stitching multiple per-chunk grids.

**Why this is a separate command.** Steps A–C are chunk-parallel: different
chunks of the same bag can run on different machines simultaneously without
coordination. The global outputs require all chunks to be finished first.
Separating reduce into its own command makes the dependency explicit and avoids
blocking the chunk-level pipeline.

**Downstream use.** The global static map is useful for:

- **Relocalization checks in tracking (Stage 5).** If a tracked object's
  trajectory collapses onto a static infrastructure point across many frames,
  it is likely a false positive or a misclassified parked object.
- **Noise cancellation for distant objects.** Returns from far-away structures
  that appear in only some chunks (due to variable range) can be suppressed by
  checking against the global static map.
- **Visualization and dataset-level QA.** The global map is the closest thing
  this pipeline has to a 3D scene reconstruction.

The global ground grid feeds `proposal_generation` (SLF `L_ground` ground
alignment loss) and `label_refinement` (per-frame box-bottom validation).

**Graceful partial runs.** If some chunks have not yet been processed (their
`static_map.npz` / `ground.npz` is missing), the reduce step silently skips
them and processes whatever is available. This allows incremental reduces
during long pipeline runs.

**Outputs:**

| Artifact | Field | Dtype | Description |
|---|---|---|---|
| `raw/<bag_id>/global_static_map.npz` | `xyz` | float64, N×3 | Downsampled static world-frame points |
| `raw/<bag_id>/global_ground.npz` | `height_grid` | float32, H×W | Ground Z at each grid cell (full bag) |
| `raw/<bag_id>/global_ground.npz` | `normal_grid` | float32, H×W×3 | Unit surface normals |
| `raw/<bag_id>/global_ground.npz` | `grid_origin` | float64, (2,) | [x₀, y₀] lower-left cell, world frame |
| `raw/<bag_id>/global_ground.npz` | `cell_size` | float32 | Cell size in metres |
| `raw/<bag_id>/global_ground.npz` | `ground_xyz` | float64, M×3 | Concatenated per-chunk ground points |

---

## Inputs

| Input | Source | Notes |
|---|---|---|
| `chunks/index.parquet` | ingest | Chunk window timestamps; drives the main loop |
| `chunks/<chunk_id>/lidar_sweeps.parquet` | ingest | Per-sweep metadata: `lidar_path`, `header_timestamp_ns`, `has_point_time`, `valid` |
| `chunks/<chunk_id>/lidar/<sweep_id:06d>.npz` | ingest | Raw sensor-frame point cloud per sweep |
| `chunks/<chunk_id>/poses.parquet` | ingest | Sparse ego poses for interpolation |
| `calibration.json` | ingest | `ego_T_lidar` extrinsic per lidar ID |
| `config/lidar_preprocessing.yaml` | this component | Algorithm parameters |

## Outputs

All outputs are written under `data/artifacts/raw/<bag_id>/`.

| Artifact | Description |
|---|---|
| `chunks/<chunk_id>/lidar_proc/<sweep_id:06d>_world.npz` | Deskewed world-frame sweep |
| `chunks/<chunk_id>/lidar_proc/<sweep_id:06d>_dynamic_mask.npy` | Per-point dynamic boolean mask |
| `chunks/<chunk_id>/lidar_proc_index.parquet` | Per-sweep processing metadata (includes `frame_id` for cross-lidar grouping) |
| `chunks/<chunk_id>/lidar_proc_summary.parquet` | Chunk-level aggregation: point counts, sweep stats, cache budget |
| `chunks/<chunk_id>/static_map.npz` | Accumulated static cloud |
| `chunks/<chunk_id>/dynamic_map.npz` | Accumulated dynamic cloud + `sweep_id` per point (proposal_generation / SLF input) |
| `chunks/<chunk_id>/voxel_occupancy.npz` | Sparse int32 voxel coords (SAM4D / MinkUNet input; enabled by default) |
| `chunks/<chunk_id>/ground.npz` | Height grid, normal grid, ground points |
| `global_static_map.npz` | Bag-level downsampled static cloud (from `reduce`) |
| `global_ground.npz` | Bag-level height grid + normal grid (from `reduce`; spans all chunks so consumers near chunk boundaries don't have to stitch) |

**Chunk summary schema** (`lidar_proc_summary.parquet`):

| Field | Type | Description |
|---|---|---|
| `bag_id`, `chunk_id` | str | Identity |
| `n_sweeps_total`, `n_sweeps_valid`, `n_sweeps_invalid` | int32 | Sweep counts: total, passed validation, dropped |
| `n_points_total`, `n_points_static`, `n_points_dynamic`, `n_points_ground` | int32 | Aggregated point counts from all sweeps |
| `n_dropped_dynamic_ground` | int32 | Points dropped at ground-dynamic voxel intersection (future feature) |
| `cache_auto_disabled` | bool | Whether the cache was automatically disabled due to budget |
| `estimated_cache_bytes` | int64 | Estimated memory footprint if full caching was attempted |
| `ground_status` | str | Status of ground extraction: `"ok"`, `"skipped_no_ground_mask"`, `"empty"` |

## How to run

```bash
# Build the image (includes pypatchworkpp C++ build, ~3-5 min first time).
./watod build

# Process all chunks of a bag (steps A + B + C per chunk).
# Both the bag directory path and the normalized bag_id are accepted.
./watod run lidar_preprocessing --bag data/bags/NuScenes-v1.0-mini-scene-1100/
./watod run lidar_preprocessing --bag NuScenes_v1_0_mini_scene_1100   # equivalent

# Process a single chunk only.
./watod run lidar_preprocessing --bag data/bags/NuScenes-v1.0-mini-scene-1100/ --chunk 0000

# Re-process already-completed chunks (e.g. after a code change).
./watod run lidar_preprocessing --bag data/bags/NuScenes-v1.0-mini-scene-1100/ --force

# After all chunks are done, build the bag-level reductions (step D):
# global_static_map.npz + global_ground.npz.
# reduce is a separate subcommand — run it directly inside the dev container.
./watod -t lidar_preprocessing_dev   # open a shell in the dev container
python -m wato_lidar_preprocessing reduce --bag NuScenes_v1_0_mini_scene_1100

# Run tests.
./watod test lidar_preprocessing
```

**Local development** (no container required, no pypatchworkpp needed for most tests):

```bash
PYTHONPATH=src/common/src:src/lidar_preprocessing/src \
    python3 -m pytest src/lidar_preprocessing/tests src/common/tests -q
```

**Spot-check outputs after a run:**

```python
import numpy as np

# Check world-frame sweep — should be in absolute SLAM map coordinates.
d = np.load("data/artifacts/raw/<bag_id>/chunks/<chunk_id>/lidar_proc/000000_world.npz")
print("world-frame x range:", d['x'].min(), d['x'].max())

# Check static map — should be denser than any single sweep.
s = np.load("data/artifacts/raw/<bag_id>/chunks/<chunk_id>/static_map.npz")
print("static points:", s['xyz'].shape[0])

# Check dynamic map (proposal_generation input).  sweep_id is per-point.
dm = np.load("data/artifacts/raw/<bag_id>/chunks/<chunk_id>/dynamic_map.npz")
print("dynamic points:", dm['xyz'].shape[0], "across",
      len(np.unique(dm['sweep_id'])), "sweeps")

# Check voxel occupancy (SAM4D input).
vo = np.load("data/artifacts/raw/<bag_id>/chunks/<chunk_id>/voxel_occupancy.npz")
print("occupied voxels:", vo['coords'].shape[0])

# Check ground grid.
g = np.load("data/artifacts/raw/<bag_id>/chunks/<chunk_id>/ground.npz")
print("height grid shape:", g['height_grid'].shape)
print("grid origin:", g['grid_origin'])
print("ground point count:", g['ground_xyz'].shape[0])

# Check bag-level global ground (after `reduce`).  Spans all chunks.
gg = np.load("data/artifacts/raw/<bag_id>/global_ground.npz")
print("global ground grid:", gg['height_grid'].shape, "origin:", gg['grid_origin'])
```

## Configuration

### YAML parameters

All parameters live in [`config/lidar_preprocessing.yaml`](config/lidar_preprocessing.yaml).
The Pydantic schema is in [`src/wato_lidar_preprocessing/config.py`](src/wato_lidar_preprocessing/config.py).

| Parameter | Default | Description |
|---|---|---|
| `voxel_size_m` | 0.15 | Voxel side length for static/dynamic classification (m) |
| `static_sweep_fraction` | 0.30 | Fraction of chunk sweeps a voxel must be occupied in to be static |
| `static_sweep_min` | 5 | Minimum sweep count regardless of fraction |
| `global_map_voxel_size_m` | 0.30 | Voxel size for global static map downsampling (m) |
| `point_time_unit` | `"seconds"` | Unit of `t_offset_us` field: `"seconds"` \| `"microseconds"` \| `"nanoseconds"` |
| `save_voxel_occupancy` | `true` | Emit `voxel_occupancy.npz` for SAM4D / MinkUNet encoders |
| `patchwork.sensor_height` | 1.8 | LiDAR height above ground (m) |
| `patchwork.th_dist` | 0.15 | Ground inlier distance threshold (m) |
| `patchwork.max_range` | 90.0 | Maximum range considered for ground (m) |
| `frame_sync.canonical_lidar` | `null` | Canonical lidar for multi-lidar frame grouping. `null` → each sweep is its own frame (right for single-lidar bags). Set e.g. `"lidar_cc"` for the 3-Velodyne rig. |
| `frame_sync.tolerance_ms` | 25.0 | Non-canonical sweeps within ±this window of a canonical sweep inherit its `frame_id`. |

**`point_time_unit` note.** Ingest saves whatever per-point time field the LiDAR
provides (under the name `t_offset_us`) without unit conversion. Velodyne's `t`
field is in seconds. Other LiDARs may use microseconds or nanoseconds. Check your
hardware datasheet and set this parameter accordingly. If the unit is wrong,
deskewed points will be wildly displaced from their correct world-frame positions.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `WATO_LIDAR_CACHE_BYTES` | `4_000_000_000` | Budget for in-memory world-sweep caching (bytes). If a chunk's estimated cache size exceeds this, classify disables the cache and processes sweeps with two full loads instead of one. Set to `0` to always disable caching. |

The cache budget is estimated conservatively (100 KB per point × point count) and
logged in the chunk summary. If `cache_auto_disabled=True`, consider increasing
the budget on memory-rich machines or decreasing chunk size on memory-constrained
ones.

## Package layout

```text
src/lidar_preprocessing/
├── config/
│   └── lidar_preprocessing.yaml  # algorithm parameters
├── src/wato_lidar_preprocessing/
│   ├── cli.py        # Click CLI: `run` and `reduce` subcommands
│   ├── config.py     # Pydantic schema: ComponentConfig + PatchworkParams
│   ├── voxel.py      # shared voxel-key packing: pack_voxel_key(), voxel_indices()
│   ├── pipeline.py   # orchestration: calls deskew → classify → ground per chunk
│   ├── deskew.py     # Step A: per-point pose interpolation + world projection
│   ├── classify.py   # Step B: voxel occupancy table + static/dynamic masks
│   ├── ground.py     # Step C: per-sweep ground-mask aggregator + height grid
│   ├── reduce.py     # Step D: bag-level global static map merge (numpy-based downsample)
│   └── io.py         # reader helpers for downstream components
└── tests/
    ├── test_deskew.py    # parametrised extrinsic calibration (6 mounting positions)
    ├── test_classify.py  # synthetic static vs one-off sweeps; asserts masks
    ├── test_ground.py    # flat/tilted planes; height grid accuracy; Patchwork++ smoke test
    ├── test_pipeline.py  # chunk summary aggregation; cache budget auto-disable
    └── test_reduce.py    # two-chunk merge; downsampling; graceful missing-chunk handling
```

## Possible follow-ups

These are known trade-offs, not active bugs. Listed roughly in order of value
vs. complexity:

- **Intersect per-sweep ground masks with the static-voxel mask.** Today a
  point flagged as ground by Patchwork++ in a single sweep contributes to the
  chunk ground cloud regardless of whether classify later marks its voxel as
  dynamic. Filtering ground points whose voxel ended up dynamic would clean
  up vehicle-underside contamination without changing the per-sweep design.
  See Step C for the technical setup already in place.
- **Per-axis voxel range bookkeeping.** `AXIS_BITS = 20` in `voxel.py` caps
  per-axis index at ±157 km @ 0.15 m. This is fine for any realistic drive
  but could be unpacked into separate uint32 arrays if we ever need to support
  city-spanning voxel grids without re-keying.
- **Skip the second NPZ load when intensity is needed.** `cache_world_xyz`
  defaults to True so this is rarely hit, but the off-cache path could be
  unified to load once and slice.
- **Per-sweep ground count column in `lidar_proc_index.parquet`.** Today the
  ground mask lives only inside each world NPZ. Surfacing the count in the
  index would let downstream stages spot sweeps where Patchwork++ went
  pathological without opening every NPZ.
- **Configurable height-grid extent.** The grid currently sizes itself to the
  bbox of the chunk's ground points. Forcing a fixed metric extent (or
  chunk-overlap-aware extent) would make the grid mergeable across chunks.

## Dependencies

**Patchwork++:** `pypatchworkpp` is built from C++ source during the Docker image
build (via `uv pip install pypatchworkpp==1.0.4`). This requires `libeigen3-dev`
(added to the Dockerfile apt install) and `cmake` (present in the base image).
The build takes 3–5 minutes on first pull. The version `1.0.4` is pinned to match
the `patchworkpp_vendor` tag in `wato_monorepo`.

If `pypatchworkpp` is not installed (e.g. in a local dev environment without the
Docker image), steps A and B run normally. Step C skips with a warning logged
at `WARNING` level. Tests that exercise Patchwork++ are automatically skipped via
`pytest.importorskip`.

**Pure Python stack:** This component's voxel classification (`classify.py`),
ground aggregation (`ground.py`), and global static map reduction (`reduce.py`)
use only numpy, scipy, and PyArrow — no external C++ dependencies beyond
Patchwork++. The `reduce` step's voxel downsampling is implemented via numpy
voxel quantization, making the component runnable in pure-Python environments
if Patchwork++ is not needed.

## Testing

The test suite covers all four processing steps and includes:

- **`test_deskew.py`:** Synthetic sweeps with parametrised extrinsic calibrations
  (6 mounting positions: identity, pure translation, pure rotation, combined
  transform, roof mount, side mount). Each test case generates known world-frame
  points and verifies the coordinate transformation.

- **`test_classify.py`:** Static vs. dynamic classification logic on synthetic
  chunks. Tests voxel occupancy thresholds, intensity backfilling, empty-chunk
  handling, and schema validation.

- **`test_ground.py`:** Per-sweep ground aggregation, height-grid accuracy on
  flat and tilted planes, and integration with Patchwork++ (via pytest skip
  if not available).

- **`test_pipeline.py`:** End-to-end orchestration: chunk-level summary
  aggregation, cache auto-disable behavior with `WATO_LIDAR_CACHE_BYTES`,
  failure isolation, and parallel chunk processing.

Run tests locally without Docker:

```bash
PYTHONPATH=src/common/src:src/lidar_preprocessing/src \
    python3 -m pytest src/lidar_preprocessing/tests -v
```
