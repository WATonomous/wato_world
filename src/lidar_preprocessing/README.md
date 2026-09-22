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
A.   deskew/          per-sweep motion compensation + world-frame projection
                      (Patchwork++ runs here in sensor frame; ground mask
                      stored inside each world NPZ; sensor origin also stored)
    │
    ▼
A.5  mf_mos/          learned moving-object segmentation (optional; disabled
                      by default; produces per-sweep raw-frame boolean masks)
    │
    ▼
B.   classify/        voxel-based static / dynamic decomposition
                      (log-odds via Amanatides-Woo ray traversal; constants
                      derived from the sensor model; optional MF-MOS fusion)
    │
    ▼
C.   ground/          aggregate per-sweep ground masks → height grid
    │
    ▼
D.   reduce/          [separate command] bag-level global static map
```

---

### Step A — Deskew and project (`deskew/`)

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
| `origin` | float64 (3,) | Sensor position in world frame at sweep time — consumed by classify's AW ray traversal |
| `ground_mask` | bool (N,) | Per-point ground flag from Patchwork++ (sensor frame) |
| `intensity` | float32 | If present in raw sweep |
| `ring` | uint16 | If present in raw sweep |

**Metadata per sweep** (`lidar_proc_index.parquet`, ProcessedSweepMeta):

| Field | Type | Description |
|---|---|---|
| `bag_id`, `chunk_id`, `sweep_id`, `lidar_id` | str/int | Identity |
| `reference_timestamp_ns` | int64 | Sweep timestamp (ns) |
| `world_path` | str | URI to world-frame NPZ |
| `dynamic_mask_path` | str | URI to per-point dynamic mask |
| `mf_mos_mask_path` | str (nullable) | URI to raw-frame MF-MOS mask (null when MF-MOS disabled) |
| `n_points_total`, `n_points_static`, `n_points_dynamic` | int32 | Point counts |
| `world_xmin/xmax/ymin/ymax/zmin/zmax` | float | Bounding box in world frame |
| `has_intensity`, `deskewed` | bool | Feature flags |
| `frame_id` | int64 (nullable) | Canonical-frame grouping per `frame_sync` config. When `canonical_lidar=null`, each lidar's sweeps are numbered sequentially. When set, non-canonical sweeps within `±tolerance_ms` inherit the canonical sweep's frame_id. |

---

### Step A.5 — MF-MOS learned segmentation (`mf_mos/`)

**What it does.** An optional learned moving-object segmentation step that runs
between deskew and classify. Disabled by default (`mf_mos.enabled: false`).
When enabled, it runs the MF-MOS (Multi-Frame Moving Object Segmentation) model
on each sweep to produce a per-point boolean mask: `True` = the model thinks
this point belongs to a moving object.

**Algorithm.** MF-MOS projects each sweep's world-frame points into a range
image (spherical projection). To detect motion, it computes residual range
images: for each past-sweep offset (derived — see below), the current sweep's
range image minus the warp of the historical sweep into the
current viewpoint. A moving object leaves a nonzero residual after ego-motion
correction; a static wall does not. The multi-frame residual channels are
concatenated and fed to a lightweight encoder-decoder. Output logits above
`score_threshold` (default 0.5) are labeled moving.

**Key design points:**
- Runs in sensor frame on raw-length point arrays (before nonfinite filtering),
  so the mask can be aligned to raw NPZ arrays by downstream steps.
- Uses the stored `ego_T_lidar` extrinsic + SLAM poses to warp historical sweeps
  into the current viewpoint for residual computation.
- Skips any sweep that **deskew** already flagged invalid (`valid=False` in the
  proc index) — most commonly the start-of-bag window with no usable ego pose.
  deskew is the single source of truth for per-sweep pose validity (it honors
  ingest's `frame_index` `valid_pose`), so MF-MOS defers to it rather than
  re-checking, which also avoids a per-sweep pose-gap warning flood over that
  transient window. These sweeps get no mask; classify skips them regardless.
- Still guards its own residual pairs: skips a sweep if the pose gap to the
  required historical sweep exceeds `max_pose_gap_ms` — prevents bad residuals
  from large ego-motion jumps on sweeps deskew did accept.
- When `save_scores: true`, also writes a float32 `_mf_mos_score.npy` alongside
  each mask for threshold tuning.

**Fusion with classify.** The relationship between MF-MOS and the AW log-odds
classifier in Step B is controlled by `fusion_mode`:

| `fusion_mode` | Behaviour |
|---|---|
| `independent` | MF-MOS masks are written but Step B ignores them. Both signals available independently. |
| `union` | A point is dynamic if the AW classifier OR the sweep's MF-MOS voxel set flags it. A missing/empty MF-MOS mask leaves the AW verdict unchanged. |

There is deliberately no third "MF-MOS decides everything" mode. One existed
(`mfmos_only`) and was removed: it replaced the classifier's verdict outright,
so a sweep MF-MOS skipped — and it skips a lot, 190 of 656 sweeps in one
measured chunk and all 101 in another — contributed nothing at all to
`dynamic_map.npz`, with no error anywhere. Union can only ever add movers, so
a model that goes quiet degrades to the voxel classifier instead of silently
emptying the output. A chunk-level warning counts sweeps that had no usable
mask.

Fusion happens at **voxel level** within each sweep: the sweep's (spatially
denoised) MF-MOS point mask is lifted to the set of voxels containing at
least one flagged point, and every point of that sweep landing in those
voxels is fused per the table above. There is no chunk-wide vote tier —
per-sweep speckle is removed at mask-generation time by the 3D
connected-component denoise (`moving_cluster_voxel_m`,
`moving_min_cluster_pts`), and temporal confirmation of movers is the
downstream tracker's job. The Patchwork++ ground filter is re-applied after
fusion so a voxel-level MF-MOS flag can never drag co-voxel ground points
into `dynamic_map.npz`.

**Outputs per sweep:**

| Artifact | Description |
|---|---|
| `lidar_proc/<sweep_id:06d>_mf_mos_mask.npy` | `bool[N_raw]`, aligned to raw sweep NPZ length |
| `lidar_proc/<sweep_id:06d>_mf_mos_score.npy` | `float32[N_raw]`, logit scores (when `save_scores: true`) |

---

### Step B — Voxel classify (`classify/`)

**What it does.** Treats the entire set of world-frame sweeps for a chunk as a
4D occupancy volume and classifies every point as belonging to the static
background or to a dynamic (moving) object. The output is a per-sweep boolean
mask (`True` = dynamic) and per-chunk accumulated static/dynamic clouds.

#### Bayesian ray-casting with Amanatides-Woo traversal

For each sweep, the sensor origin and all endpoint points are passed to the
Amanatides-Woo 3D-DDA ray traversal kernel. The kernel marches each ray from
the sensor through the voxel grid, updating per-voxel log-odds accumulators:

- **Along the ray** (free-space voxels): `log_odds -= l_free` (evidence of
  absence — light passed through here to reach the measured surface). Carving
  stops a sensor-model-derived margin short of the endpoint, is down-weighted
  past the beam-footprint crossover `d* = voxel_size / divergence`, and skips
  occupied voxels the ray merely grazes (per-voxel surface-normal incidence
  gate).
- **At the endpoint** (occupied voxel): `log_odds += l_occ` (evidence of
  presence). Endpoint hits are registered for rays of ANY length — rays longer
  than the profile's `max_range_m` skip only the carve (a compute guard), so
  far returns still accumulate occupancy evidence instead of staying
  unobserved.
- Ground endpoint voxels: free-space carving runs along ground rays but
  `l_occ` is NOT added at the endpoint — this lets air voxels above the road
  accumulate free evidence while ground-surface voxels stay with
  `n_hits == 0` (classified not-dynamic, not polluting the static cloud).
  This is not a mode; the road is not a mover, and crediting every drive-over
  with occupancy evidence for a surface Patchwork++ has already claimed has
  no defensible alternative.

All log-odds constants (`l_occ`, `l_free`, clamp, decision thresholds, carve
margin, grazing gate, carve guard range) are derived from the datasheet sensor
profile selected by `sensor_model` — they are not YAML knobs. See
`sensor_model.py`, which also records why the four inverse-sensor-model
probabilities it does contain (`p_hit`, `p_miss`, `p_clamp`, `k_sigma`) are the
values they are, and what evidence would move them.

After Pass 1, `classify_from_log_odds` converts log-odds to occupancy
probabilities and partitions the observed voxels:

```
static_arr   = voxels where evidenced & has_hits & p_occ >= p_static_threshold
dynamic_arr  = voxels where evidenced & has_hits & p_occ <  p_dynamic_threshold
(free-only, under-evidenced-with-hits, and the ambiguous band in between are
 neither static nor dynamic)
```

A point is dynamic **if and only if its voxel key IS in `dynamic_arr`** —
the explicit carved-dynamic set. Voxels that were never observed (e.g.
zero-length rays) default to NOT dynamic: absence of evidence is not motion
evidence. Static-cloud membership uses `static_arr` only, so under-evidenced
and free-space voxels pollute neither `static_map.npz` nor `dynamic_map.npz`.

The Amanatides-Woo kernel is JIT-compiled by Numba for performance. The kernel
hard-fails at import time if Numba is absent — install `numba>=0.59` in the
container (already in the Dockerfile).

#### Two-pass memory management

At 10 Hz over a 30-second chunk, a naive approach holding all 30M world-frame
points in memory simultaneously would require ~720 MB of float64 arrays. The
classifier uses two passes:

- **Pass 1**: load each sweep's world-frame NPZ once and run the DDA
  accumulator (sub-pass 1a estimates per-voxel surface normals for the
  incidence gate; sub-pass 1b ray-casts the log-odds grid). Only voxel-key
  dicts and arrays are kept in memory; large coordinate arrays are cached only
  when `cache_world_xyz_in_memory: true` (default) and the estimated size is
  below `WATO_LIDAR_CACHE_BYTES`.
- **Pass 2**: apply the resulting `static_arr` / `dynamic_arr` (and the
  per-sweep MF-MOS voxel set when fusion is active) via searchsorted to each
  sweep, write the dynamic mask, and accumulate static/dynamic clouds.

**Voxel key encoding.** Each voxel `(vx, vy, vz)` is encoded into a single
`int64` as `vx << 40 | vy << 20 | vz` (20 bits per axis), supporting a ±524 km
range per axis at 0.15 m resolution. All sorted arrays support O(log K) lookup
via `np.searchsorted` — no Python dict overhead in Pass 2.

**Outputs:**

| Artifact | Description |
|---|---|
| `lidar_proc/<sweep_id:06d>_dynamic_mask.npy` | `bool[N]`, True = dynamic point |
| `static_map.npz` | Accumulated static cloud: `xyz` (float64, M×3), `intensity`, `voxel_size`, `origin`, `static_voxel_keys`, `dynamic_voxel_keys` (the carved-dynamic voxel set Step C intersects against) |
| `dynamic_map.npz` | Accumulated dynamic cloud: `xyz` (float64, M×3), `sweep_id` (int32, M), `intensity` (when present) |
| `voxel_occupancy.npz` | Sparse int32 voxel coords for SAM4D / MinkUNet (all sweeps aggregated). Toggle via `save_voxel_occupancy` (default: true). |
| `voxel_occupancy_frame_NNNN.npz` | Per-frame sparse voxel coords (what `perception_2d` feeds to MinkUNet). Written when `save_per_frame_voxel_occupancy: true`. |
| `voxel_diag.npz` | Per-voxel `log_odds` / `p_occ` / `n_obs` / `n_hits` / `classification` for every touched voxel, including carved ones. Toggle via `save_voxel_diagnostics`. |
| `lidar_proc_index.parquet` | Updated with `n_points_static`, `n_points_dynamic`, `dynamic_mask_path` per sweep |

---

### Step C — Ground extraction (`ground/`)

**What it does.** Aggregates the per-sweep ground masks that Step A wrote into
the world NPZs, then builds a 2D height grid and surface-normal grid over the
extent of the chunk. Also drops any "ground" point whose voxel Step B
classified DYNAMIC (`dynamic_voxel_keys` in `static_map.npz`). The check is
against the dynamic set, NOT the static set: ground endpoints are credited no
occupancy hit, so ground voxels can never be static and a
keep-if-static intersection would silently discard essentially all ground
points.

**Where Patchwork++ actually runs.** Patchwork++ runs *per sweep* inside Step A
(`deskew/`), on sensor-frame xyz, before the world-frame transform is applied.
Step C does not call Patchwork++ — it unions the per-sweep ground masks into a
chunk-level ground cloud. This mirrors the monorepo's
`patchwork::GroundRemovalCore` which also operates on sensor-frame PointCloud2
messages one at a time. Three reasons we kept the per-sweep formulation:

- *Patchwork++'s zone model is sensor-centric.* The Concentric Zone Model (CZM)
  divides the field of view into concentric annular zones around the sensor
  origin. An accumulated static cloud is centred on the SLAM map origin, and
  points at very different ranges relative to the current ego pose end up in the
  wrong zones. The library works in sensor frame; we follow that.
- *Per-sweep parallelism.* Running Patchwork++ inside the deskew loop means the
  per-sweep work is one pass over the data.
- *Algorithm parity with the monorepo.*

**Ground-dynamic intersection.** Points flagged ground by Patchwork++ whose
voxel was classified dynamic by Step B are dropped before the height grid is
built. This removes vehicle-underside contamination (low-riding cars that
triggered ground classification in early sweeps before classify strips them as
dynamic). The count of dropped points is reported in `n_dropped_dynamic_ground`
in the chunk summary.

**Height grid.** The ground point cloud from Patchwork++ is rasterised into a
2D grid at `ground_cell_size_m` resolution (default 0.25 m). Each cell stores
the median Z of all ground points within it. Empty cells are filled by
nearest-neighbour from populated cells (using
`scipy.ndimage.distance_transform_edt`). Surface normals are estimated per cell
from the finite-difference gradient of the height grid (`np.gradient`).

**Outputs** (`ground.npz`):

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `height_grid` | H×W | float32 | Ground Z at each grid cell (world frame) |
| `normal_grid` | H×W×3 | float32 | Unit surface normals |
| `grid_origin` | (2,) | float64 | [x₀, y₀] lower-left cell, world frame |
| `cell_size` | scalar | float32 | Cell size in metres |
| `ground_xyz` | M×3 | float64 | Raw Patchwork++ ground points (after dynamic filter) |

---

### Step D — Bag-level reduce (`reduce/`)

**What it does.** After all chunks for a bag have been processed by steps A–C,
the `reduce` subcommand merges per-chunk artifacts into two bag-level outputs:

- `global_static_map.npz` — concatenates every chunk's `static_map.npz` and
  voxel-downsamples to ~30 cm resolution via numpy voxel snap (quantize to grid
  cells, keep unique points).
- `global_ground.npz` — concatenates every chunk's `ground_xyz` and rebuilds
  a single bag-level height grid + normal grid. This solves SLF's `L_ground`
  chunk-boundary problem: a box fit near a chunk seam can query `z_ground(x, y)`
  over the full bag without stitching multiple per-chunk grids.

**Why this is a separate command.** Steps A–C are chunk-parallel: different
chunks of the same bag can run on different machines simultaneously. The global
outputs require all chunks to be finished first. Separating reduce makes the
dependency explicit.

**Graceful partial runs.** If some chunks have not yet been processed, the reduce
step silently skips them and processes whatever is available.

**Outputs:**

| Artifact | Field | Dtype | Description |
|---|---|---|---|
| `raw/<bag_id>/global_static_map.npz` | `xyz` | float64, N×3 | Downsampled static world-frame points |
| `raw/<bag_id>/global_ground.npz` | `height_grid` | float32, H×W | Ground Z (full bag) |
| `raw/<bag_id>/global_ground.npz` | `normal_grid` | float32, H×W×3 | Unit surface normals |
| `raw/<bag_id>/global_ground.npz` | `grid_origin` | float64, (2,) | [x₀, y₀] lower-left cell |
| `raw/<bag_id>/global_ground.npz` | `cell_size` | float32 | Cell size in metres |
| `raw/<bag_id>/global_ground.npz` | `ground_xyz` | float64, M×3 | Concatenated per-chunk ground points |

---

## Inputs

| Input | Source | Notes |
|---|---|---|
| `chunks/index.parquet` | ingest | Chunk window timestamps; drives the main loop |
| `chunks/<chunk_id>/lidar_sweeps.parquet` | ingest | Per-sweep metadata |
| `chunks/<chunk_id>/lidar/<sweep_id:06d>.npz` | ingest | Raw sensor-frame point cloud |
| `chunks/<chunk_id>/poses.parquet` | ingest | Sparse ego poses for interpolation |
| `calibration.json` | ingest | `ego_T_lidar` extrinsic per lidar ID |
| `config/lidar_preprocessing.yaml` | this component | Algorithm parameters |

## Outputs

All outputs are written under `data/artifacts/raw/<bag_id>/`.

| Artifact | Description |
|---|---|
| `chunks/<chunk_id>/lidar_proc/<sweep_id:06d>_world.npz` | Deskewed world-frame sweep (xyz, origin, ground_mask, intensity) |
| `chunks/<chunk_id>/lidar_proc/<sweep_id:06d>_dynamic_mask.npy` | Per-point dynamic boolean mask |
| `chunks/<chunk_id>/lidar_proc/<sweep_id:06d>_mf_mos_mask.npy` | MF-MOS moving mask, raw-frame aligned (when MF-MOS enabled) |
| `chunks/<chunk_id>/lidar_proc_index.parquet` | Per-sweep processing metadata |
| `chunks/<chunk_id>/lidar_proc_summary.parquet` | Chunk-level aggregation: point counts, MF-MOS stats, cache budget |
| `chunks/<chunk_id>/static_map.npz` | Accumulated static cloud + static/dynamic voxel-key sets |
| `chunks/<chunk_id>/dynamic_map.npz` | Accumulated dynamic cloud + `sweep_id` per point |
| `chunks/<chunk_id>/voxel_occupancy.npz` | Sparse int32 voxel coords, all sweeps aggregated |
| `chunks/<chunk_id>/voxel_occupancy_frame_NNNN.npz` | Per-frame sparse voxel coords (when `save_per_frame_voxel_occupancy: true`) |
| `chunks/<chunk_id>/voxel_diag.npz` | Per-voxel log-odds diagnostics incl. carved voxels (when `save_voxel_diagnostics: true`) |
| `chunks/<chunk_id>/ground.npz` | Height grid, normal grid, ground points |
| `global_static_map.npz` | Bag-level downsampled static cloud (from `reduce`) |
| `global_ground.npz` | Bag-level height grid + normal grid (from `reduce`) |
| `chunks/<chunk_id>/manifest_lidar_preprocessing.json` | Traceability record: image provenance, content-hashed ingest inputs, outputs, config hash |

**Chunk summary schema** (`lidar_proc_summary.parquet`):

| Field | Type | Description |
|---|---|---|
| `bag_id`, `chunk_id` | str | Identity |
| `n_sweeps_total`, `n_sweeps_valid`, `n_sweeps_invalid` | int64 | Sweep counts |
| `n_points_total`, `n_points_static`, `n_points_dynamic`, `n_points_ground` | int64 | Aggregated point counts |
| `n_dropped_dynamic_ground` | int64 | Ground points dropped at dynamic-voxel intersection |
| `cache_auto_disabled` | bool | Whether cache was auto-disabled due to memory budget |
| `estimated_cache_bytes` | int64 | Estimated memory if full caching was used |
| `ground_status` | str | `"ok"`, `"skipped_no_ground_mask"`, or `"empty"` |
| `mf_mos_n_processed` | int64 (nullable) | Sweeps processed by MF-MOS |
| `mf_mos_n_skipped` | int64 (nullable) | Sweeps MF-MOS **failed** on: deskew-invalid, pose gap, empty cloud, inference error |
| `mf_mos_n_unsupported` | int64 (nullable) | Sweeps from scanners below `MIN_BEAMS` (e.g. VLP-16) — skipped by design, not a failure |
| `mf_mos_n_points_moving` | int64 (nullable) | Total points labeled moving across all sweeps |

The three sweep counts add up to `n_sweeps_valid`. All `mf_mos_*` fields are
null when `mf_mos.enabled: false` — "didn't run" is distinct from "ran, found
nothing". On the WATO rig expect `mf_mos_n_unsupported` ≈ 2/3 of valid sweeps:
only the VLP-32C centre lidar is projectable.

## How to run

```bash
# Build the image (includes pypatchworkpp C++ build, ~3-5 min first time).
./watod build

# Process all chunks of a bag (steps A + A.5 + B + C per chunk).
# Automatically runs the bag-level reduce (step D) after all chunks finish.
./watod run lidar_preprocessing --bag data/bags/NuScenes-v1.0-mini-scene-1100/
./watod run lidar_preprocessing --bag NuScenes_v1_0_mini_scene_1100   # equivalent

# Process a single chunk only (auto-reduce is skipped on single-chunk runs).
./watod run lidar_preprocessing --bag data/bags/NuScenes-v1.0-mini-scene-1100/ --chunk 0000

# Re-process already-completed chunks (e.g. after a code change).
./watod run lidar_preprocessing --bag data/bags/NuScenes-v1.0-mini-scene-1100/ --force

# Disable auto-reduce when processing chunks across multiple machines.
./watod run lidar_preprocessing --bag <bag> --no-auto-reduce
./watod -t lidar_preprocessing_dev   # open a shell in the dev container
python -m wato_lidar_preprocessing reduce --bag NuScenes_v1_0_mini_scene_1100

# WATO 3-Velodyne rig bags: use the rig profile (per-corner lidars, velodyne
# sensor model, frame_sync).  Ingest the bag with ingest.wato.yaml first.
./watod run lidar_preprocessing --bag <wato_bag> \
    --config /ws/src/lidar_preprocessing/config/lidar_preprocessing.wato.yaml

# Run tests.
./watod test lidar_preprocessing
```

**Local development.** The full suite needs `numba` and `pypatchworkpp`, which
the image has. On a bare host without them, about 58 of 147 tests fail with
`ImportError` (classify, MF-MOS fusion, deskew/pipeline integration). Only
`test_ray_traversal.py` and the Patchwork++ smoke test in `test_ground.py`
skip cleanly. Run the whole suite in the container:

```bash
./watod test lidar_preprocessing
```

or on the host after `pip install 'numba>=0.59' pypatchworkpp==1.0.4` (the
latter needs `libeigen3-dev`):

```bash
PYTHONPATH=src/common/src:src/lidar_preprocessing/src \
    python3 -m pytest src/lidar_preprocessing/tests -q
```

**Spot-check outputs after a run:**

```python
import numpy as np

# World-frame sweep — in absolute SLAM map coordinates.
d = np.load("data/artifacts/raw/<bag_id>/chunks/<chunk_id>/lidar_proc/000000_world.npz")
print("world-frame x range:", d['x'].min(), d['x'].max())
print("sensor origin:", d['origin'])

# Static map — denser than any single sweep.
s = np.load("data/artifacts/raw/<bag_id>/chunks/<chunk_id>/static_map.npz")
print("static points:", s['xyz'].shape[0])

# Dynamic map (proposal_generation input).  sweep_id is per-point.
dm = np.load("data/artifacts/raw/<bag_id>/chunks/<chunk_id>/dynamic_map.npz")
print("dynamic points:", dm['xyz'].shape[0], "across",
      len(np.unique(dm['sweep_id'])), "sweeps")

# Ground grid.
g = np.load("data/artifacts/raw/<bag_id>/chunks/<chunk_id>/ground.npz")
print("height grid shape:", g['height_grid'].shape)
print("grid origin:", g['grid_origin'])

# Bag-level global ground (after `reduce`).  Spans all chunks.
gg = np.load("data/artifacts/raw/<bag_id>/global_ground.npz")
print("global ground grid:", gg['height_grid'].shape)
```

## Configuration

Two profiles, mirroring ingest's `ingest.yaml` / `ingest.wato.yaml` pattern:

- [`config/lidar_preprocessing.yaml`](config/lidar_preprocessing.yaml) —
  nuScenes bags (the default when `--config` is omitted).
- [`config/lidar_preprocessing.wato.yaml`](config/lidar_preprocessing.wato.yaml)
  — WATO 3-Velodyne rig (per-lidar scanner profiles, `frame_sync.canonical_lidar:
  lidar_cc`).

The Pydantic schema is in [`src/wato_lidar_preprocessing/config.py`](src/wato_lidar_preprocessing/config.py).

**What the config is allowed to say.** It names the scanners the bag was
recorded with and states the handful of choices that are genuinely ours. It
states no physics. Beam count, field of view, spin rate and direction, range
accuracy, beam divergence, usable range and intensity scale all come from the
profile table in `sensor_model.py`, as does every log-odds constant derived
from them. If a value in YAML looks like it belongs on a datasheet, it is in
the wrong file.

### Sensor model

| Parameter | Default | Description |
|---|---|---|
| `sensor_model.profile` | `"vlp32c"` | The scanner this bag was recorded with: `vlp32c`, `vlp16` or `hdl32e` (nuScenes LIDAR_TOP). Fixes `l_occ`, `l_free`, the log-odds clamp, the decision thresholds, the range-credibility crossover, the carve margin, the grazing gate, the carve guard range, the scan rate and direction, and the MF-MOS projection geometry. |
| `sensor_model.per_lidar` | `{}` | `{lidar_id: profile}` overrides for a mixed rig, e.g. `{lidar_cc: vlp32c, lidar_ne: vlp16, lidar_nw: vlp16}`. Deskew and MF-MOS use each sweep's own scanner; the chunk-level decision thresholds come from the default profile, which is safe because all profiles share them by construction. |

### Step B — Classification

| Parameter | Default | Description |
|---|---|---|
| `voxel_size_m` | 0.15 | Voxel side length for static/dynamic classification (m). A real trade-off with no datasheet answer: smaller is more faithful but more sensitive to pose drift and beam spacing. |
| `min_observations` | 3 | The only evidence gate: voxels with fewer ray traversals stay under-evidenced (neither static nor dynamic). "Was anything ever measured here?" needs no threshold and is not configurable. |

### Step A.5 — MF-MOS

| Parameter | Default | Description |
|---|---|---|
| `mf_mos.enabled` | `false` | Enable MF-MOS inference (requires CUDA + pretrained weights) |
| `mf_mos.checkpoint_path` | `/data/models/mf_mos/mf_mos_semantic_kitti.pt` | Path to pretrained model checkpoint |
| `mf_mos.arch_config` | `/data/models/mf_mos/arch_cfg.yaml` | MF-MOS architecture config |
| `mf_mos.data_config` | `/data/models/mf_mos/data_cfg.yaml` | MF-MOS data config (normalisation stats) |
| `mf_mos.device` | `"cuda"` | Inference device (`"cpu"` for smoke tests) |
| `mf_mos.score_threshold` | 0.5 | Logit threshold for binary moving label |
| `mf_mos.save_scores` | `false` | Also write float32 `_mf_mos_score.npy` per sweep |
| `mf_mos.fusion_mode` | `"independent"` | `"independent"` \| `"union"` — see the fusion table above |
| `mf_mos.max_pose_gap_ms` | 200.0 | Skip sweep if pose gap to required history exceeds this |
| `mf_mos.occlusion_range_tol_m` | 1.0 | Occlusion gate for unprojecting pixel labels back to points |
| `mf_mos.prime_window_from_prior_chunk` | `true` | Seed the residual window from the preceding chunk's sweeps |
| `mf_mos.moving_cluster_voxel_m` | 0.5 | 3D connected-component grid for per-sweep speckle removal |
| `mf_mos.moving_min_cluster_pts` | 8 | Drop moving clusters smaller than this |

**Not configurable, and why.** The spherical projection is geometry, so it is
read rather than chosen:

| Quantity | Source |
|---|---|
| Range-image rows | the lidar's profile `beams` |
| Vertical FoV bounds | the lidar's profile `fov_up_deg` / `fov_down_deg` |
| Intensity divisor | the lidar's profile `intensity_scale` |
| Range-image width (1024) | `RANGE_IMAGE_W` — a checkpoint-side sampling choice |
| Range window (2–50 m) | `TRAIN_MIN_RANGE_M` / `TRAIN_MAX_RANGE_M` — MF-MOS's own training preprocessing |
| Residual offsets | `residual_steps_for(sensor, n_scans)` — `n_input_scans` from the arch config, spaced by `round(sweep_rate / 10 Hz)` so each channel spans the wall-clock motion the KITTI-trained model expects. At 20 Hz that is `[2, 4, …, 16]`. |
| Which lidars run at all | `beams >= MIN_BEAMS` (32). This replaced `lidar_id_allowlist`: whether a scanner can be projected is a property of the scanner, not a list someone maintains. On the WATO rig it is what confines MF-MOS to the centre VLP-32C. |

### Other parameters

| Parameter | Default | Description |
|---|---|---|
| `global_map_voxel_size_m` | 0.30 | Voxel size for global static map downsampling (m). Doubles as the two-pass prior's KDTree match radius — reduce snaps points to voxel centres, so "within one map voxel" is what a match means. |
| `point_time_unit` | `"seconds"` | Unit of `t_offset_us` field: `"seconds"` \| `"microseconds"` \| `"nanoseconds"` |
| `synthesize_per_point_times` | `true` | Synthesize per-point timestamps from azimuth when the raw NPZ lacks them. The rotation period and direction come from the sensor profile. |
| `cache_world_xyz_in_memory` | `true` | Cache world-frame xyz in memory for Pass 2. Auto-disabled when estimated size exceeds `WATO_LIDAR_CACHE_BYTES`. |
| `save_voxel_occupancy` | `true` | Emit `voxel_occupancy.npz` (all sweeps aggregated — QA/visualization) |
| `save_voxel_diagnostics` | `false` | Emit `voxel_diag.npz` (per-voxel log_odds/n_obs/n_hits/classification incl. carved voxels — powers viz's p_occ mode and the debug scripts) |
| `save_per_frame_voxel_occupancy` | `false` | Emit one `voxel_occupancy_frame_NNNN.npz` per `frame_id` — what `perception_2d` feeds to SAM4D's MinkUNet encoder |
| `patchwork.sensor_height` | 1.8 | LiDAR height above ground (m) |
| `patchwork.th_dist` | 0.15 | Ground inlier distance threshold (m) |
| `patchwork.max_range` | 90.0 | Maximum range considered for ground (m) |
| `patchwork.ground_cell_size_m` | 0.25 | Height-grid cell resolution (m) |
| `frame_sync.canonical_lidar` | `null` | Canonical lidar for multi-lidar frame grouping (`null` = each sweep is its own frame; `lidar_cc` on the WATO rig) |
| `frame_sync.tolerance_ms` | 25.0 | Non-canonical sweeps within ±this window inherit the canonical sweep's `frame_id` |

**`point_time_unit` note.** Ingest saves whatever per-point time field the LiDAR
provides (under the name `t_offset_us`) without unit conversion. Velodyne's `t`
field is in seconds. Other LiDARs may use microseconds or nanoseconds. If the
unit is wrong, deskewed points will be wildly displaced from their correct
world-frame positions.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `WATO_LIDAR_CACHE_BYTES` | `4_000_000_000` | Budget for in-memory world-sweep caching (bytes). If a chunk's estimated size exceeds this, classify disables the cache and processes sweeps with two full disk reads instead of one. |

## Package layout

```text
src/lidar_preprocessing/
├── config/
│   ├── lidar_preprocessing.yaml      # nuScenes profile (default; Pydantic-validated)
│   └── lidar_preprocessing.wato.yaml # WATO 3-Velodyne rig profile
├── src/wato_lidar_preprocessing/
│   ├── cli.py                         # Click CLI: `run` and `reduce` subcommands
│   ├── config.py                      # Pydantic schema: ComponentConfig, MFMosParams, etc.
│   ├── sensor_model.py                # datasheet profiles → derived classifier constants
│   ├── pipeline.py                    # orchestration: deskew → mf_mos → classify → ground
│   ├── voxel.py                       # shared voxel-key packing: voxel_indices(), pack_voxel_key()
│   ├── io.py                          # reader helpers for downstream components
│   ├── viz.py                         # optional Open3D visualization helpers
│   ├── _inputs.py                     # shared I/O: load_pose_samples(), load_ego_T_lidar()
│   │                                  # (used by both deskew/ and mf_mos/)
│   │
│   ├── ray_traversal/                 # Amanatides-Woo 3D-DDA voxel traversal
│   │   ├── __init__.py                # public: make_log_odds_dicts, update_sweep_log_odds,
│   │   │                              #         extract_log_odds_arrays
│   │   ├── dispatch.py                # Numba/Python kernel selector; hard-fails if Numba absent
│   │   ├── _numba_kernel.py           # JIT-compiled AW traversal (primary path)
│   │   ├── _python_kernel.py          # pure-Python AW traversal (testing/fallback)
│   │   └── _keys.py                   # voxel key helpers shared by both kernels
│   │
│   ├── deskew/                        # Step A — motion compensation + world projection
│   │   ├── __init__.py                # public: process_chunk, DeskewResult
│   │   └── _core.py                   # implementation: per-point pose interpolation,
│   │                                  # Patchwork++ per sweep, world NPZ writer
│   │
│   ├── mf_mos/                        # Step A.5 — learned moving-object segmentation
│   │   ├── __init__.py                # public: process_chunk, MFMosResult
│   │   ├── _core.py                   # range projection, residual computation, mask writing
│   │   └── _runtime.py                # model loading, PyTorch inference (lazy import)
│   │
│   ├── classify/                      # Step B — voxel static/dynamic decomposition
│   │   ├── __init__.py                # public: process_chunk, ClassifyResult
│   │   ├── pipeline.py                # two-pass orchestration; MF-MOS fusion dispatch
│   │   ├── log_odds.py                # build_log_odds_grid (AW Pass 1 + normals + global prior),
│   │   │                              # classify_from_log_odds (static_arr / dynamic_arr)
│   │   ├── masking.py                 # apply_classification_to_sweep (Pass 2 per-sweep masks)
│   │   ├── global_map_prior.py        # bag-level KDTree prior for two-pass mode
│   │   ├── io_helpers.py              # load_world_full, load_mf_mos_world_mask, origin_from_index
│   │   └── occupancy_export.py        # write_chunk_voxel_occupancy, write_per_frame_voxel_occupancy,
│   │                                  # write_chunk_voxel_diagnostics
│   │
│   ├── ground/                        # Step C — ground mask aggregation + height grid
│   │   ├── __init__.py                # public: process_chunk, GroundResult
│   │   └── _core.py                   # ground-dynamic intersection, height grid builder
│   │
│   └── reduce/                        # Step D — bag-level global static map
│       ├── __init__.py                # public: reduce_static_map, reduce_ground_map
│       └── _core.py                   # voxel-snap downsample, global height grid
│
└── tests/
    ├── test_sensor_model.py           # profile sanity, derived constants, per-lidar resolution
    ├── test_deskew.py                 # per-point world projection, 6 extrinsic configurations
    ├── test_classify.py               # log-odds classification, dynamic-default regressions,
    │                                  # MF-MOS fusion contracts
    ├── test_mf_mos.py                 # range projection, residuals, fusion modes (Groups 1–7)
    ├── test_ray_traversal.py          # AW kernel parity (Numba vs Python), voxel traversal
    ├── test_ground.py                 # flat/tilted planes, height grid, dynamic intersection
    ├── test_global_map_prior.py       # two-pass IWU prior, range weighting
    ├── test_pipeline.py               # chunk summary, cache auto-disable, parallel workers
    └── test_reduce.py                 # two-chunk merge, downsampling, partial-run handling
```

## Testing

The test suite covers all processing steps with synthetic data. It needs no
GPU and no real bags, but it does need `numba` and `pypatchworkpp` (see *How to
run* above):

- **`test_deskew.py`:** Synthetic sweeps with parametrised extrinsic calibrations
  (6 mounting positions). Verifies per-point pose interpolation and world
  coordinate transforms.

- **`test_sensor_model.py`:** The property the config reduction rests on —
  picking a scanner is enough. Every profile states physical geometry, the
  derived log-odds constants stay ordered (`l_occ > l_free`, both inside the
  clamp), all profiles agree on the decision rule so a mixed rig judges every
  scanner alike, per-lidar profiles resolve and fall back, and the residual
  offsets scale with spin rate.

- **`test_classify.py`:** Log-odds classification on synthetic chunks. Covers
  static accumulation, AW ray-carving (free-space marking), free-only voxels,
  under-evidenced-with-hits, ground handling, hit-then-carved → dynamic, the
  static→dynamic leak regressions (never-observed voxels default not-dynamic;
  over-length rays still classify static; `dynamic_voxel_keys` export), and
  the MF-MOS fusion contracts — including that union keeps the classifier's
  verdict on empty/missing masks, and that a config still asking for
  `mfmos_only` fails loudly rather than being quietly ignored.

- **`test_mf_mos.py`:** Range image projection, residual computation,
  point-level mask recovery, per-sweep mask writing, the beam-count rule that
  decides which lidars MF-MOS runs on, and fusion mode contracts (Groups 1–7).

- **`test_ray_traversal.py`:** Amanatides-Woo kernel correctness — same-voxel
  edge case, Numba/Python bit-for-bit parity across random rays in all 8
  octants (including over-length rays), endpoint-hit registration for rays
  beyond the carve guard, and hard-fail behavior when Numba is unavailable.

- **`test_ground.py`:** Per-sweep ground aggregation, height-grid accuracy on
  flat and tilted planes, ground-dynamic intersection (drop-if-dynamic; ground
  points in non-static voxels must pass through), and Patchwork++ smoke test
  (auto-skipped if not installed).

- **`test_pipeline.py`:** End-to-end orchestration: chunk-level summary
  aggregation, cache auto-disable, failure isolation, parallel chunk processing.

## Dependencies

**Numba / LLVM (classify log-odds):** The Amanatides-Woo kernel in
`ray_traversal/` requires `numba>=0.59` (pulls `llvmlite` automatically). If
Numba is absent at import time, `classify` hard-fails with a clear error message
and a remediation hint. The Dockerfile installs Numba in a dedicated layer.

**Patchwork++:** `pypatchworkpp` is built from C++ source during the Docker image
build (`uv pip install pypatchworkpp==1.0.4`), requiring `libeigen3-dev` (pinned
to match `patchworkpp_vendor` in `wato_monorepo`). With the default
`require_patchwork: true`, deskew raises `ImportError` if it's absent. Set
`require_patchwork: false` to run without ground extraction, which lets ground
points pollute both maps. Only the direct Patchwork++ smoke test in
`test_ground.py` skips when it's missing; the deskew and pipeline tests fail.

**PyTorch (MF-MOS):** `torch>=2.7` is installed in the Dockerfile matched to
CUDA 12.8. When `mf_mos.enabled: false` (default), PyTorch is never imported at
runtime. The MF-MOS runtime (`mf_mos/_runtime.py`) uses a lazy import that
fails with a clear message if torch is absent but MF-MOS is enabled.

**Pure Python stack:** Everything else (ground aggregation, reduce) uses only
numpy, scipy, and PyArrow. Runnable in any Python 3.12+ environment without
Docker.

**Everything is version-locked.** The exact transitive dependency set lives in
[`docker/requirements/lidar_preprocessing.txt`](../../docker/requirements/lidar_preprocessing.txt)
and is installed with `--no-deps`, so the image contents are a pure function of
this repo. MF-MOS itself is pinned to an exact upstream commit
(`ARG MF_MOS_COMMIT`) rather than cloned from `main` — it decides which points
this component calls "moving", so an unpinned clone would silently change the
static/dynamic split between rebuilds. See the reproducibility section of the
root `CLAUDE.md` for how to regenerate a lock.

Note the lock currently captures `torch==2.11.0` plus the multi-GB
`cuda-toolkit` meta-wheel, diverging from `perception_2d`'s pinned 2.7.1. That
drift is documented in the lockfile header and is a known cleanup, not an
intentional choice.

## Possible follow-ups

- **Per-point ray origins for the DDA.** Deskew stores one mid-sweep `origin`
  per sweep; at 20 Hz with fast ego motion the true sensor position drifts up
  to ±(v·25 ms) within a sweep, slightly mis-tracing carve paths near the
  vehicle. Deskew already interpolates per-point poses, so exporting per-point
  (or per-time-bucket) origins and indexing them in the kernel is mechanical.
- **Validate MF-MOS on the WATO rig.** The released checkpoint is KITTI-trained
  (64-beam) and both configs keep `fusion_mode: independent` until mask quality
  on `lidar_cc` has been audited (`scripts/debug_mfmos_contribution.py`). A
  finetuned or re-projected model would justify `union`. Worth measuring
  first: MF-MOS currently skips a large fraction of sweeps (190/656 in one
  measured chunk, 101/101 in another), so its real contribution may be far
  smaller than "enabled" suggests.
- **Per-axis voxel range bookkeeping.** `AXIS_BITS = 20` in `voxel.py` caps
  per-axis index at ±524 km @ 0.15 m. Fine for any realistic drive but could be
  unpacked into separate uint32 arrays for city-spanning grids.
- **Configurable height-grid extent.** The grid currently sizes itself to the
  bbox of the chunk's ground points. A fixed metric extent (or chunk-overlap-aware
  extent) would make the grid mergeable across chunks without the bag-level
  reduce step.
- **Per-sweep ground count column in `lidar_proc_index.parquet`.** Today the
  ground mask lives only inside each world NPZ. Surfacing the count in the index
  would let downstream stages spot pathological sweeps without opening every NPZ.
