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
                      (log-odds via Amanatides-Woo ray traversal, or legacy
                      persistence counting; optionally fuses MF-MOS votes)
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
images: for each past-sweep offset in `residual_steps` (default `[1, 2, 4, 8]`),
the current sweep's range image minus the warp of the historical sweep into the
current viewpoint. A moving object leaves a nonzero residual after ego-motion
correction; a static wall does not. The multi-frame residual channels are
concatenated and fed to a lightweight encoder-decoder. Output logits above
`score_threshold` (default 0.5) are labeled moving.

**Key design points:**
- Runs in sensor frame on raw-length point arrays (before nonfinite filtering),
  so the mask can be aligned to raw NPZ arrays by downstream steps.
- Uses the stored `ego_T_lidar` extrinsic + SLAM poses to warp historical sweeps
  into the current viewpoint for residual computation.
- Skips a sweep if the pose gap to the required historical sweep exceeds
  `max_pose_gap_ms` — prevents bad residuals from large ego-motion jumps.
- When `save_scores: true`, also writes a float32 `_mf_mos_score.npy` alongside
  each mask for threshold tuning.

**Fusion with classify.** The relationship between MF-MOS and the AW log-odds
classifier in Step B is controlled by `fusion_mode`:

| `fusion_mode` | Behaviour |
|---|---|
| `independent` | MF-MOS masks are written but Step B ignores them. Both signals available independently. |
| `union` | A voxel is dynamic if AW log-odds OR MF-MOS votes it dynamic. |
| `mfmos_only` | Dynamic mask is derived from MF-MOS votes only; AW log-odds is used only for the static cloud. |

Fusion happens at **voxel level**, not per-point. For the log-odds path, MF-MOS
votes are accumulated during Pass 1 of classify alongside the AW log-odds:

```
For each sweep in Pass 1:
  1. AW ray traversal updates log_odds / n_obs / n_hits dicts (as usual)
  2. Load MF-MOS mask for this sweep (aligned to world-frame length)
  3. For each unique endpoint voxel this sweep:
       n_sweep_hits[voxel] += 1
  4. For each unique endpoint voxel labeled moving this sweep:
       mf_mos_votes[voxel] += 1

After Pass 1:
  vote_fraction = mf_mos_votes / n_sweep_hits
  mf_mos_dynamic_arr = voxels where votes >= min_mf_mos_votes
                       AND vote_fraction >= mf_mos_vote_fraction_threshold
```

This means a single noisy sweep cannot force a voxel dynamic — cross-sweep
agreement is required, exactly like the AW occupancy evidence. Votes are counted
once per SWEEP (not per point) to prevent high-density voxels from inflating
their vote fraction. For the persistence path, fusion falls back to per-sweep
binary OR (since there are no DDA-derived unique_keys to anchor chunk-level votes).

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

**Two classification methods.** `classification_method` in the config chooses
between them:

#### `log_odds` (default) — Bayesian ray-casting with Amanatides-Woo traversal

The primary method. For each sweep, the sensor origin and all endpoint points
are passed to the Amanatides-Woo 3D-DDA ray traversal kernel. The kernel
marches each ray from the sensor through the voxel grid, updating per-voxel
log-odds accumulators:

- **Along the ray** (free-space voxels): `log_odds -= l_free` (evidence of
  absence — light passed through here to reach the measured surface)
- **At the endpoint** (occupied voxel): `log_odds += l_occ` (evidence of
  presence)
- Ground endpoint voxels: with `ground_endpoint_strategy: skip_endpoint` (default),
  free-space carving runs along ground rays but `l_occ` is NOT added at the
  endpoint — this lets air voxels above the road accumulate free evidence while
  ground-surface voxels stay with `n_hits == 0` (classified not-dynamic, not
  polluting the static cloud)

After Pass 1, `classify_from_log_odds` converts log-odds to occupancy
probabilities and applies three gates:

```
static_arr     = voxels where evidenced & has_hits & p_occ >= p_static_threshold
free_only_arr  = voxels with n_hits == 0 (only ever traversed, never hit)
under_arr      = voxels that are evidenced=False but have hits (benefit of doubt)

not_dynamic_arr = union(static_arr, free_only_arr, under_arr)
```

A point is dynamic if and only if its voxel key is NOT in `not_dynamic_arr`.
The separation of `static_arr` (used for the static cloud) from `not_dynamic_arr`
(used for the dynamic mask) prevents under-evidenced voxels and free-space
ground voxels from polluting `static_map.npz`.

The Amanatides-Woo kernel is JIT-compiled by Numba for performance. The kernel
hard-fails at import time if Numba is absent — install `numba>=0.59` in the
container (already in the Dockerfile) or fall back to `classification_method:
persistence`.

#### `persistence` (fallback) — sweep-count threshold

The legacy method. For each voxel, counts the number of sweeps that placed an
endpoint in it. A voxel is static if the count exceeds
`max(static_sweep_min, static_sweep_fraction × n_sweeps)`. No ray traversal;
no Numba required. Useful for A/B comparison and environments without CUDA.

#### Two-pass memory management

At 10 Hz over a 30-second chunk, a naive approach holding all 30M world-frame
points in memory simultaneously would require ~720 MB of float64 arrays. Both
methods use two passes:

- **Pass 1**: load each sweep's world-frame NPZ once, run the appropriate
  accumulator (DDA or sweep-count), optionally accumulate MF-MOS votes. Only
  voxel-key dicts and arrays are kept in memory; large coordinate arrays are
  cached only when `cache_world_xyz_in_memory: true` (default) and the estimated
  size is below `WATO_LIDAR_CACHE_BYTES`.
- **Pass 2**: apply the resulting `static_arr` / `not_dynamic_arr` /
  `mf_mos_dynamic_arr` via searchsorted to each sweep, write the dynamic mask,
  and accumulate static/dynamic clouds.

**Voxel key encoding.** Each voxel `(vx, vy, vz)` is encoded into a single
`int64` as `vx << 40 | vy << 20 | vz` (20 bits per axis), supporting a ±524 km
range per axis at 0.15 m resolution. All sorted arrays support O(log K) lookup
via `np.searchsorted` — no Python dict overhead in Pass 2.

**Outputs:**

| Artifact | Description |
|---|---|
| `lidar_proc/<sweep_id:06d>_dynamic_mask.npy` | `bool[N]`, True = dynamic point |
| `static_map.npz` | Accumulated static cloud: `xyz` (float64, M×3), `intensity`, `voxel_size`, `origin`, `static_voxel_keys` |
| `dynamic_map.npz` | Accumulated dynamic cloud: `xyz` (float64, M×3), `sweep_id` (int32, M), `intensity` (when present) |
| `voxel_occupancy.npz` | Sparse int32 voxel coords for SAM4D / MinkUNet (all sweeps aggregated). Toggle via `save_voxel_occupancy` (default: true). |
| `voxel_occupancy_frame_NNNN.npz` | Per-frame sparse voxel coords (what `perception_2d` feeds to MinkUNet). Written when `save_per_frame_voxel_occupancy: true`. |
| `lidar_proc_index.parquet` | Updated with `n_points_static`, `n_points_dynamic`, `dynamic_mask_path` per sweep |

---

### Step C — Ground extraction (`ground/`)

**What it does.** Aggregates the per-sweep ground masks that Step A wrote into
the world NPZs, then builds a 2D height grid and surface-normal grid over the
extent of the chunk. Also intersects ground masks with the static-voxel set to
drop any "ground" point whose voxel ended up classified dynamic by Step B.

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
| `chunks/<chunk_id>/static_map.npz` | Accumulated static cloud + voxel keys |
| `chunks/<chunk_id>/dynamic_map.npz` | Accumulated dynamic cloud + `sweep_id` per point |
| `chunks/<chunk_id>/voxel_occupancy.npz` | Sparse int32 voxel coords, all sweeps aggregated |
| `chunks/<chunk_id>/voxel_occupancy_frame_NNNN.npz` | Per-frame sparse voxel coords (when `save_per_frame_voxel_occupancy: true`) |
| `chunks/<chunk_id>/ground.npz` | Height grid, normal grid, ground points |
| `global_static_map.npz` | Bag-level downsampled static cloud (from `reduce`) |
| `global_ground.npz` | Bag-level height grid + normal grid (from `reduce`) |

**Chunk summary schema** (`lidar_proc_summary.parquet`):

| Field | Type | Description |
|---|---|---|
| `bag_id`, `chunk_id` | str | Identity |
| `n_sweeps_total`, `n_sweeps_valid`, `n_sweeps_invalid` | int32 | Sweep counts |
| `n_points_total`, `n_points_static`, `n_points_dynamic`, `n_points_ground` | int32 | Aggregated point counts |
| `n_dropped_dynamic_ground` | int32 | Ground points dropped at dynamic-voxel intersection |
| `cache_auto_disabled` | bool | Whether cache was auto-disabled due to memory budget |
| `estimated_cache_bytes` | int64 | Estimated memory if full caching was used |
| `ground_status` | str | `"ok"`, `"skipped_no_ground_mask"`, or `"empty"` |
| `mf_mos_n_processed` | int32 (nullable) | Sweeps processed by MF-MOS |
| `mf_mos_n_skipped` | int32 (nullable) | Sweeps skipped by MF-MOS (pose gap, empty cloud) |
| `mf_mos_n_points_moving` | int32 (nullable) | Total points labeled moving across all sweeps |

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

# Run tests.
./watod test lidar_preprocessing
```

**Local development** (no container required; most tests run without pypatchworkpp):

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

All parameters live in [`config/lidar_preprocessing.yaml`](config/lidar_preprocessing.yaml).
The Pydantic schema is in [`src/wato_lidar_preprocessing/config.py`](src/wato_lidar_preprocessing/config.py).

### Step B — Classification

| Parameter | Default | Description |
|---|---|---|
| `voxel_size_m` | 0.15 | Voxel side length for static/dynamic classification (m) |
| `classification_method` | `"log_odds"` | `"log_odds"` (Bayesian AW ray-casting) or `"persistence"` (sweep-count threshold) |

**Log-odds parameters** (active when `classification_method: log_odds`):

| Parameter | Default | Description |
|---|---|---|
| `l_occ` | 0.85 | Log-odds increment per occupied endpoint hit |
| `l_free` | 0.40 | Log-odds decrement per free-space traversal |
| `log_odds_clamp` | 5.0 | Symmetric clamp preventing ossification after long history |
| `p_static_threshold` | 0.70 | `sigmoid(log_odds) >= this` → classified static |
| `p_dynamic_threshold` | 0.30 | `sigmoid(log_odds) < this` → classified dynamic (if evidenced) |
| `min_observations` | 3 | Voxels with fewer ray traversals stay "unknown" (not dynamic) |
| `min_occupied_hits` | 1 | Voxels with `n_hits < this` are free-space-only; not dynamic |
| `max_ray_length_m` | 80.0 | Rays beyond this range are truncated (noise dominates at long range) |
| `free_space_margin_voxels` | 1.0 | Stop free-space carving N voxels before the endpoint |
| `ground_endpoint_strategy` | `"skip_endpoint"` | `"skip_endpoint"` (traverse ground rays but skip `l_occ` at endpoint) or `"skip_ray"` (skip ground rays entirely; legacy) |

**Persistence parameters** (active when `classification_method: persistence`):

| Parameter | Default | Description |
|---|---|---|
| `static_sweep_fraction` | 0.30 | Fraction of chunk sweeps a voxel must be occupied in to be static |
| `static_sweep_min` | 5 | Minimum sweep count regardless of fraction |

### Step A.5 — MF-MOS

| Parameter | Default | Description |
|---|---|---|
| `mf_mos.enabled` | `false` | Enable MF-MOS inference (requires CUDA + pretrained weights) |
| `mf_mos.checkpoint_path` | `/data/models/mf_mos/mf_mos_semantic_kitti.pt` | Path to pretrained model checkpoint |
| `mf_mos.arch_config` | `/data/models/mf_mos/arch_cfg.yaml` | MF-MOS architecture config |
| `mf_mos.data_config` | `/data/models/mf_mos/data_cfg.yaml` | MF-MOS data config (range image dims, FoV) |
| `mf_mos.residual_steps` | `[1, 2, 4, 8]` | Past-sweep offsets for residual channels |
| `mf_mos.range_image_h` | 32 | Range image height (32 for NuScenes, 64 for KITTI) |
| `mf_mos.range_image_w` | 1024 | Range image width |
| `mf_mos.fov_up_deg` | 10.0 | LiDAR vertical FoV upper bound (NuScenes default) |
| `mf_mos.fov_down_deg` | -30.0 | LiDAR vertical FoV lower bound (NuScenes default) |
| `mf_mos.device` | `"cuda"` | Inference device (`"cpu"` for smoke tests) |
| `mf_mos.score_threshold` | 0.5 | Logit threshold for binary moving label |
| `mf_mos.save_scores` | `false` | Also write float32 `_mf_mos_score.npy` per sweep |
| `mf_mos.fusion_mode` | `"independent"` | `"independent"` \| `"union"` \| `"mfmos_only"` |
| `mf_mos.max_pose_gap_ms` | 200.0 | Skip sweep if pose gap to required history exceeds this |
| `mf_mos_vote_fraction_threshold` | 0.5 | Fraction of MF-MOS-observed sweeps that must vote a voxel moving (log_odds path only) |
| `min_mf_mos_votes` | 1 | Minimum absolute vote count required (log_odds path only) |

### Other parameters

| Parameter | Default | Description |
|---|---|---|
| `global_map_voxel_size_m` | 0.30 | Voxel size for global static map downsampling (m) |
| `point_time_unit` | `"seconds"` | Unit of `t_offset_us` field: `"seconds"` \| `"microseconds"` \| `"nanoseconds"` |
| `cache_world_xyz_in_memory` | `true` | Cache world-frame xyz in memory for Pass 2. Auto-disabled when estimated size exceeds `WATO_LIDAR_CACHE_BYTES`. |
| `save_voxel_occupancy` | `true` | Emit `voxel_occupancy.npz` (all sweeps aggregated — QA/visualization) |
| `save_per_frame_voxel_occupancy` | `false` | Emit one `voxel_occupancy_frame_NNNN.npz` per `frame_id` — what `perception_2d` feeds to SAM4D's MinkUNet encoder |
| `patchwork.sensor_height` | 1.8 | LiDAR height above ground (m) |
| `patchwork.th_dist` | 0.15 | Ground inlier distance threshold (m) |
| `patchwork.max_range` | 90.0 | Maximum range considered for ground (m) |
| `patchwork.ground_cell_size_m` | 0.25 | Height-grid cell resolution (m) |
| `frame_sync.canonical_lidar` | `null` | Canonical lidar for multi-lidar frame grouping (`null` = each sweep is its own frame) |
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
│   └── lidar_preprocessing.yaml      # algorithm parameters (Pydantic-validated)
├── src/wato_lidar_preprocessing/
│   ├── cli.py                         # Click CLI: `run` and `reduce` subcommands
│   ├── config.py                      # Pydantic schema: ComponentConfig, MFMosParams, etc.
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
│   │   ├── log_odds.py                # build_log_odds_grid (AW Pass 1 + vote accumulation),
│   │   │                              # classify_from_log_odds (thresholds + mf_mos_dynamic_arr)
│   │   ├── masking.py                 # apply_classification_to_sweep (Pass 2 per-sweep masks)
│   │   ├── persistence.py             # classify_persistence (sweep-count fallback)
│   │   ├── io_helpers.py              # load_world_full, load_mf_mos_world_mask, origin_from_index
│   │   └── occupancy_export.py        # write_chunk_voxel_occupancy, write_per_frame_voxel_occupancy
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
    ├── test_deskew.py                 # per-point world projection, 6 extrinsic configurations
    ├── test_classify.py               # persistence + log-odds classification, MF-MOS vote fusion
    ├── test_mf_mos.py                 # range projection, residuals, fusion modes (Groups 1–5)
    ├── test_ray_traversal.py          # AW kernel parity (Numba vs Python), voxel traversal
    ├── test_ground.py                 # flat/tilted planes, height grid, dynamic intersection
    ├── test_pipeline.py               # chunk summary, cache auto-disable, parallel workers
    └── test_reduce.py                 # two-chunk merge, downsampling, partial-run handling
```

## Testing

The test suite covers all processing steps without requiring Docker, a GPU, or
real bag data:

- **`test_deskew.py`:** Synthetic sweeps with parametrised extrinsic calibrations
  (6 mounting positions). Verifies per-point pose interpolation and world
  coordinate transforms.

- **`test_classify.py`:** Persistence and log-odds classification on synthetic
  chunks. Covers static accumulation, AW ray-carving (free-space marking),
  free-only voxels, under-evidenced-with-hits (bug fix #2), ground-in-skip-ray
  mode (bug fix #4), and voxel-level MF-MOS vote aggregation (3 tests).

- **`test_mf_mos.py`:** Range image projection, residual computation,
  point-level mask recovery, per-sweep mask writing, and fusion mode contracts
  (Groups 1–5).

- **`test_ray_traversal.py`:** Amanatides-Woo kernel correctness — same-voxel
  edge case, Numba/Python bit-for-bit parity across 50 random rays in all 8
  octants, and hard-fail behavior when Numba is unavailable.

- **`test_ground.py`:** Per-sweep ground aggregation, height-grid accuracy on
  flat and tilted planes, ground-dynamic intersection (dynamic ground drops),
  and Patchwork++ smoke test (auto-skipped if not installed).

- **`test_pipeline.py`:** End-to-end orchestration: chunk-level summary
  aggregation, cache auto-disable, failure isolation, parallel chunk processing.

## Dependencies

**Numba / LLVM (classify log-odds):** The Amanatides-Woo kernel in
`ray_traversal/` requires `numba>=0.59` (pulls `llvmlite` automatically). If
Numba is absent at import time, `classify` hard-fails with a clear error message
and a remediation hint. The Dockerfile installs Numba in a dedicated layer.
Fall back to `classification_method: persistence` to bypass this requirement.

**Patchwork++:** `pypatchworkpp` is built from C++ source during the Docker image
build (`uv pip install pypatchworkpp==1.0.4`), requiring `libeigen3-dev` (pinned
to match `patchworkpp_vendor` in `wato_monorepo`). If absent, Step C skips with
a warning. Tests that exercise Patchwork++ are auto-skipped via
`pytest.importorskip`.

**PyTorch (MF-MOS):** `torch>=2.7` is installed in the Dockerfile matched to
CUDA 12.8. When `mf_mos.enabled: false` (default), PyTorch is never imported at
runtime. The MF-MOS runtime (`mf_mos/_runtime.py`) uses a lazy import that
fails with a clear message if torch is absent but MF-MOS is enabled.

**Pure Python stack:** Everything else (voxel classify persistence path, ground
aggregation, reduce) uses only numpy, scipy, and PyArrow. Runnable in any Python
3.12+ environment without Docker.

## Possible follow-ups

- **Expand log-odds AW votes to the persistence path.** Currently voxel-level
  MF-MOS vote aggregation only works on the `log_odds` path (because persistence
  doesn't compute DDA-derived unique_keys). Adding a lightweight key-accumulation
  pass to `_run_pass_1_persistence` would unify the fusion paths.
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
