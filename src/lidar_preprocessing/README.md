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
B.   static/dynamic decomposition — picked by `--seg aw|mos|union`
     (cfg.segmentation). aw and mos are independent (share no code); union
     fuses them:

       seg=aw   classify/   voxel-based decomposition via log-odds
                            Amanatides-Woo ray traversal (or legacy
                            persistence counting). No MF-MOS, no model.

       seg=mos  mf_mos/     learned moving-object segmentation. Runs the
                            MF-MOS model (range-image residual MOS) and
                            derives static/dynamic purely from the per-sweep
                            moving masks. No ray traversal.

       seg=union union/     fusion. Runs aw (static basis) + mos, keeps aw's
                            high-precision static map, takes the dynamic cloud
                            from MF-MOS gated and vetoed:
                            dynamic = mf_mos_moving & ~ground & ~near_ego
                                      & ~near_ground & ~aw_static_dilated.
                            ~near_ground = below union.ground_height_veto_m
                            over Step C's ground grid (road FPs are invisible
                            to the static veto — road voxels are never
                            static). Step C therefore runs BEFORE the fusion
                            on this path only. A post-veto temporal motion
                            filter (union.motion_filter) then drops the
                            structure leakage the voxel vetoes miss — see
                            "Motion filter" below.
    │
    ▼
C.   ground/          aggregate per-sweep ground masks → height grid
    │
    ▼
D.   reduce/          [separate command] bag-level global static map
```

## Visualization workflow

The `viz` command is the shared entrypoint for native, browser, and external
viewer workflows:

```bash
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id>
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --backend html
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --backend web --port 8765
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --export ply
```

Use the default Open3D backend when native windows are working. It provides the
full stage viewer: deskewed per-sweep inspection (`--stage A --sweep N`),
static/dynamic accumulated views, ground-grid plots, and the bag-level reduced
map (`--stage D`).

Use `--backend html` for the cleaner debug loop. It writes a standalone WebGL
viewer under `<chunk>/viz/chunk.html` by default, with static/dynamic toggles,
dynamic sweep scrubbing, trail/all modes, point-size controls, and color modes
for sweep id, height, and intensity. No Open3D, DISPLAY forwarding, or X server
is required; open the emitted HTML file in a browser. For a single processed
sweep, add `--sweep N` and it writes `sweep_<N>.html` colored by static vs
dynamic classification.

Use `--backend web` for the most dynamic local workflow. It starts a local
browser app, serves point buffers over HTTP, and supports the same sweep
scrubbing/playback controls without embedding the whole cloud into one HTML
file. Voxel diagnostic color modes (`p_occ`, `n_obs`, `n_hits`,
`classification`) appear automatically when `voxel_diag.npz` exists.

Use `--export ply` for external tools such as CloudCompare or ParaView. The
binary PLY includes scalar fields for `dynamic`, `sweep_id`, `intensity`,
`p_occ`, `n_obs`, `n_hits`, and `classification`, so those viewers can color
and filter by WATO classifier state.

### How to use the viewers

**Open3D backend**

```bash
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id>
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --stage A --sweep <N>
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --stage B --sweep <N>
wato_lidar_preprocessing viz --bag <bag_id> --stage D
```

Use this when you need the full current pipeline viewer. Stage A shows one
deskewed sweep, with ground points highlighted when `ground_mask` is present.
Stage B shows static vs dynamic classification. Stage C opens the ground grid
matplotlib view. Stage D opens the bag-level reduced static map after `reduce`
has produced `global_static_map.npz`.

Common Open3D controls:

- `1`, `2`, `3`, `4`: snap to top/front/side/isometric views.
- Mouse drag: rotate/orbit.
- Mouse wheel: zoom.
- Stage/sweep windows support layer toggles shown in the terminal log, such as
  static/dynamic/ground/ego overlays where available.
- The dynamic chunk viewer includes sweep playback, mode selection, optional
  static backdrop, optional ego path, and color modes such as `p_occ`,
  `classification`, `n_obs`, `n_hits`, `intensity`, and `sweep_id` when those
  fields exist.

**Standalone HTML backend**

```bash
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --backend html
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --sweep <N> --backend html
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --backend html --out /tmp/lidar_viz
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --backend html --out viz_outputs --open
./watod run lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --backend html --open
```

This writes a self-contained `.html` file and prints its path. Add `--open` to
open the generated file in the default browser after writing; relative `--out`
paths are resolved from the current working directory. Use it when you want
something shareable or do not want DISPLAY/X11 or Open3D GUI support.
When invoked through `./watod run lidar_preprocessing viz ... --open`, the
host wrapper maps generated `/data/artifacts/...` HTML back to `data/artifacts`
and opens it from the host, so it also works from Docker on WSL.

Browser controls:

- `view`: top, iso, side, or front camera preset.
- `mode`: `single sweep`, `trail`, or `all dynamic`.
- `color`: static/dynamic, sweep id, height, or intensity when available.
- `sweep` slider: scrub the dynamic sweep.
- `prev`, `next`, `play`: step or animate sweeps.
- `static`, `dynamic`: toggle point layers.
- `point`: adjust point size.

**Local web backend**

```bash
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --backend web
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --backend web --host 0.0.0.0 --port 8765
```

This starts a local HTTP server and prints a URL. Open that URL in a browser.
It serves binary point buffers over HTTP instead of embedding the whole cloud
into one HTML file, so it is the better local interaction loop for larger
chunks.

The controls match the standalone HTML viewer, with additional diagnostic color
modes when `voxel_diag.npz` exists:

- `p_occ`: voxel occupancy probability, useful for distinguishing
  threshold-edge dynamics from heavily carved space.
- `n_obs`: number of sweeps that observed/touched the voxel.
- `n_hits`: endpoint hits in the voxel.
- `classification`: classifier bucket code from the voxel diagnostics.

Stop the server with `Ctrl-C` in the terminal.

**PLY export**

```bash
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --export ply
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --sweep <N> --export ply
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --export ply --out /tmp/lidar_exports
```

Open the resulting `.ply` in CloudCompare, ParaView, or another point-cloud
tool. Use scalar-field coloring/filtering on:

- `dynamic`: `0` static, `1` dynamic.
- `sweep_id`: originating sweep for dynamic chunk points, `-1` for accumulated
  static chunk points.
- `intensity`: LiDAR intensity, `-1` when absent.
- `p_occ`, `n_obs`, `n_hits`, `classification`: voxel diagnostics, `-1` when
  `voxel_diag.npz` is unavailable or the point has no matching voxel entry.

### Current usability gaps

The visualization stack is now usable, but it is still a debugging v1 rather
than a polished inspection app.

- The web backend is chunk-level only. Per-sweep inspection still uses
  `--backend html --sweep N`, `--export ply --sweep N`, or Open3D.
- Stage C ground-grid visualization and Stage D global-map visualization are
  still Open3D/matplotlib-only.
- The local web backend preloads and downsamples point buffers at server start;
  it does not yet stream individual sweeps on demand from disk.
- Browser viewers support coloring and playback, but not point picking,
  measurement tools, box drawing, DBSCAN cluster controls, or selected-point
  metadata panels.
- `p_occ` and classification controls are visualization-only. They do not
  re-run classification or preview threshold changes interactively.
- Camera projection and image overlays are not implemented yet.
- External integrations are PLY-only today. Rerun, Foxglove/MCAP, Potree, and
  LAS/LAZ export are still future work.
- PLY chunk export gives scalar fields, not a full temporal playback format.
  Use the web/Open3D viewers when sweep timing matters.
- The web frontend is embedded in Python for zero new build tooling. That keeps
  deployment simple, but makes UI iteration less comfortable than a dedicated
  frontend asset structure.

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
| `mf_mos_mask_path` | str (nullable) | URI to raw-frame MF-MOS mask (null on the `seg=aw` path) |
| `n_points_total`, `n_points_static`, `n_points_dynamic` | int32 | Point counts |
| `world_xmin/xmax/ymin/ymax/zmin/zmax` | float | Bounding box in world frame |
| `has_intensity`, `deskewed` | bool | Feature flags |
| `frame_id` | int64 (nullable) | Canonical-frame grouping per `frame_sync` config. When `canonical_lidar=null`, each lidar's sweeps are numbered sequentially. When set, non-canonical sweeps within `±tolerance_ms` inherit the canonical sweep's frame_id. |

---

### Step B (seg=mos) — MF-MOS learned segmentation (`mf_mos/`)

**Selected with `--seg mos`.** A fully self-contained alternative to the
Amanatides-Woo classifier below. It does **not** run when `--seg aw`, and it
never touches the AW log-odds / ray-traversal code. It runs the MF-MOS
(Multi-Frame Moving Object Segmentation) model on each sweep and derives the
static/dynamic split purely from the per-sweep moving masks.

**Two stages, both in `mf_mos/`:**

1. *Inference* (`mf_mos/_core.py`) — produces a per-point moving mask per sweep.
2. *Segmentation* (`mf_mos/segment.py`, `classify_chunk`) — turns those masks
   into `static_map.npz` / `dynamic_map.npz` / per-sweep `dynamic_mask.npy`:

   ```
   dynamic = mf_mos_moving & ~ground     # Patchwork++ ground is authoritative
   static  = ~mf_mos_moving & ~ground    # ground points belong to ground.npz
   ```

   This is the **pure-MOS split**: every non-ground point the model didn't flag
   moving is static. No ray traversal, no voxel vote aggregation, no fusion.

**Inference algorithm.** MF-MOS projects each sweep's points into a range image
(spherical projection). To detect motion it computes residual range images: for
each past-sweep offset in `residual_steps`, the current range image minus the
ego-motion-warped historical sweep. A moving object leaves a nonzero residual
after ego-motion correction; a static wall does not. The multi-frame residuals
are concatenated and fed to a lightweight encoder-decoder; per-pixel moving
probability above `score_threshold` is labeled moving and unprojected to points.

**Key design points:**
- Masks are raw-length (before nonfinite filtering) so they align to raw NPZs.
- Uses the stored `ego_T_lidar` extrinsic + SLAM poses to warp historical sweeps
  into the current viewpoint for residual computation.
- **Two distinct time caps** (do not confuse them):
  - `max_pose_gap_ms` gates *pose-interpolation* quality when warping.
  - `max_residual_gap_ms` caps the *sweep-to-sweep residual baseline*. A
    residual at step `k` spans `k * sweep_dt`, which for the longer
    `residual_steps` legitimately exceeds `max_pose_gap_ms`. These were once
    the same knob, which silently zeroed every long residual channel and
    collapsed multi-frame MOS to ~2 live channels; keep `max_residual_gap_ms`
    above `max(residual_steps) * sweep_dt`.
- A sweep with no usable mask is left entirely static (never fabricates dynamics).
- When `save_scores: true`, also writes a float32 `_mf_mos_score.npy` per sweep.

**Outputs per sweep (in addition to the shared static/dynamic artifacts):**

| Artifact | Description |
|---|---|
| `lidar_proc/<sweep_id:06d>_mf_mos_mask.npy` | `bool[N_raw]`, aligned to raw sweep NPZ length |
| `lidar_proc/<sweep_id:06d>_mf_mos_score.npy` | `float32[N_raw]`, moving scores (when `save_scores: true`) |

> **Evaluating MF-MOS on its own.** `--seg mos` is deliberately the *raw* model
> output (per-sweep threshold only) so its quality can be A/B'd against `--seg
> aw` without any geometric post-filtering muddying the comparison.

---

### Step B (seg=aw) — Voxel classify (`classify/`)

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
  accumulator (DDA or sweep-count). Only
  voxel-key dicts and arrays are kept in memory; large coordinate arrays are
  cached only when `cache_world_xyz_in_memory: true` (default) and the estimated
  size is below `WATO_LIDAR_CACHE_BYTES`.
- **Pass 2**: apply the resulting `static_arr` / `not_dynamic_arr` via
  searchsorted to each sweep, write the dynamic mask, and accumulate
  static/dynamic clouds.

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

### Step B (seg=union) — Motion filter (`union/motion_filter.py`)

**Why it exists.** The AW-static and ground-height vetoes only reach MF-MOS
false positives that land *on* the AW static map. Structure that map covers
sparsely — far walls, foliage, below-grade returns — slips through. On real
Velodyne data that left ~58% of the union dynamic cloud sitting within 25 cm of
a static surface (`scripts/compare_seg_dynamic`). MF-MOS is the wrong primitive
to fix this: it's an online per-scan model trained on SemanticKITTI (HDL-64E),
applied out-of-domain, so it over-fires on textured static surfaces. The motion
filter instead exploits the offline batch setting — the accumulated cloud over
the whole chunk — and pure geometry, so it has no domain gap.

**The persistence gate** (the workhorse, default on) targets *currently-moving*
semantics — a genuinely-moving point sweeps **through** a 0.5 m voxel in a few
sweeps, while static structure dwells in the same voxel the whole time it's in
view. Drop any point whose voxel is occupied across ≥ `persistence_max_sweeps`
distinct sweeps.

This is a **recall/precision trade**, set by `persistence_max_sweeps`. It has a
real recall cost: an *extended* mover is the problem — a 4.5 m car at 5 m/s
keeps each voxel along its path occupied for ~9 sweeps (≈ car-length / speed),
so a tight threshold cuts the bodies of normally-moving vehicles, not just
structure. The default `20` is recall-biased; on-static leakage climbs faster
than recall past ~24. The ceiling is ~75–80% mover-recall, because a long/slow
mover is indistinguishable from structure by per-voxel occupancy alone — pushing
past it needs the learned/tracking signal downstream, not more geometry here.

Measured on the WATO ring-road bag (`scripts/compare_seg_dynamic`; mover-recall
proxied by distance from the static map):

| `persistence_max_sweeps` | dynamic pts | on-static | mover-recall |
|---|---|---|---|
| (no filter) | 360.7K | 58.3% | 100% |
| 5 | 86.0K | 6.6% | 53% |
| 12 | 117.6K | 9.3% | 71% |
| **20 (default)** | **128.9K** | **12.6%** | **75%** |
| 28 | 143.3K | 16.0% | 80% |

**The coherence gate is OFF by default** (`coherence_min_life: 0`). It drops
points whose per-sweep cluster doesn't link into a ≥ `coherence_min_life`-sweep
track — useful as a speck denoiser on *dense* clouds, but on sparse (32-/64-beam)
LiDAR it cuts ~20% of real movers (distant/fragmented movers don't cluster) for
< 1% precision, so it's opt-in. It is membership-only, never a velocity test:
per-sweep visibility makes a connected-component centroid drift as the ego
passes structure, faking velocity — a velocity gate, and a translating-cluster
*rescue* of persistent points, both tested *worse* (the rescue re-admitted ~77%
structure via wall-sliding). An `MF-MOS ∩ AW-dynamic` intersection was likewise
rejected — AW-dynamic voxels hug static surfaces, so it *raised* leakage
(17% → 72% on NuScenes).

**A/B-ing.** `motion_filter.enabled: false` writes the raw post-veto cloud;
each gate is independently disabled by setting its threshold to 0. Raise
`persistence_max_sweeps` for more recall (more leakage), lower it for a cleaner
cloud (fewer movers).

The filter rewrites only the dynamic side (`dynamic_map.npz`, per-sweep
`dynamic_mask.npy`, each index row's `n_points_dynamic`); `static_map.npz` is
untouched, so Steps C/D stay method-agnostic. Drop counts are recorded in the
chunk summary as `motion_filter_n_persistence_dropped` /
`motion_filter_n_coherence_dropped`.

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

# Process all chunks of a bag (steps A + B + C per chunk).
# Automatically runs the bag-level reduce (step D) after all chunks finish.
./watod run lidar_preprocessing --bag data/bags/NuScenes-v1.0-mini-scene-1100/
./watod run lidar_preprocessing --bag NuScenes_v1_0_mini_scene_1100   # equivalent

# Pick the Step-B segmentation method (default from config; aw if unset).
./watod run lidar_preprocessing --bag <bag> --seg aw     # Amanatides-Woo only
./watod run lidar_preprocessing --bag <bag> --seg mos    # MF-MOS only (needs GPU + weights)
./watod run lidar_preprocessing --bag <bag> --seg union  # fusion: aw static + MF-MOS dynamic vetoed by it

# Score how much of a method's dynamic cloud is actually static structure
# (lower = cleaner); run after each --seg to A/B them on the same chunk:
python -m wato_lidar_preprocessing.scripts.compare_seg_dynamic <bag> 0000

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

### Step B — Segmentation method

| Parameter | Default | Description |
|---|---|---|
| `segmentation` | `"aw"` | `"aw"` (Amanatides-Woo log-odds, `classify/`), `"mos"` (MF-MOS, `mf_mos/`), or `"union"` (fusion, `union/`). Override per-run with `--seg aw\|mos\|union`. |
| `union.aw_static_veto` | `true` | seg=union: drop MF-MOS dynamics whose voxel aw confirmed static (the core of the method). `false` → raw MF-MOS dynamic, for A/B'ing the veto. |
| `union.keep_aw_dynamic` | `false` | seg=union: also union in aw's own dynamic verdict (recall mode, read from the `aw_dynamic_mask.npy` snapshot). Off by default — aw dynamics hug static surfaces. |
| `union.veto_score_exempt` | `null` | seg=union: MF-MOS movers with moving probability ≥ this survive the aw-static veto (parked-then-moving objects). Needs `mf_mos.save_scores: true`; `null` = off. |
| `union.ground_height_veto_m` | `0.25` | seg=union: drop dynamic candidates below this height over Step C's ground grid (MF-MOS road false positives are invisible to the static veto). 0.0 = off. |
| `union.veto_dilation_voxels` | `1` | seg=union: dilate the aw-static veto by this many voxels (Chebyshev) to catch the leakage shell straddling voxel boundaries; candidates in aw's own dynamic voxels are exempt from the dilated part. 0 = exact voxel only. |
| `union.motion_filter.enabled` | `true` | seg=union: run the post-veto temporal motion filter (persistence + coherence). `false` → raw post-veto cloud, for A/B'ing the filter. |
| `union.motion_filter.persistence_max_sweeps` | `20` | Drop a dynamic point whose `persistence_voxel_m` voxel is occupied across ≥ this many distinct sweeps (structure dwells; movers sweep through). The recall/precision knob — lower = cleaner but cuts more slow/large movers, higher = more recall but more leakage. Sweep count — scale with sensor Hz. 0 = off. |
| `union.motion_filter.persistence_voxel_m` | `0.5` | Voxel edge (m) for the persistence sweep-count. |
| `union.motion_filter.coherence_min_life` | `0` (off) | Opt-in denoiser: drop a dynamic point whose per-sweep cluster doesn't link into a track spanning ≥ this many sweeps. Membership only, not velocity. Off by default — cuts ~20% of real movers on sparse LiDAR; enable only on dense clouds. |
| `union.motion_filter.coherence_cell_m` | `0.4` | Connected-components cell (m) for per-sweep clustering. |
| `union.motion_filter.coherence_link_gate_m` | `3.0` | Max centroid step (m) between sweeps when linking clusters into a track. |
| `union.motion_filter.coherence_max_object_m` | `7.0` | Per-sweep cluster extent cap (m); larger clusters are treated as structure and never tracked. |
| `voxel_size_m` | 0.15 | Voxel side length for static/dynamic classification (m) |
| `classification_method` | `"log_odds"` | seg=aw backend: `"log_odds"` (Bayesian AW ray-casting) or `"persistence"` (sweep-count threshold) |

**Log-odds parameters** (seg=aw, active when `classification_method: log_odds`):

| Parameter | Default | Description |
|---|---|---|
| `l_occ` | 0.85 | Log-odds increment per occupied endpoint hit |
| `l_free` | 0.40 | Log-odds decrement per free-space traversal |
| `log_odds_clamp` | 5.0 | Symmetric clamp preventing ossification after long history |
| `p_static_threshold` | 0.70 | `sigmoid(log_odds) >= this` → classified static |
| `p_dynamic_threshold` | 0.30 | `sigmoid(log_odds) < this` → classified dynamic (if evidenced) |
| `min_observations` | 3 | Voxels with fewer ray traversals stay "unknown" (not dynamic) |
| `min_occupied_hits` | 1 | Voxels with `n_hits < this` are free-space-only; not dynamic |
| `min_hit_fraction_dynamic` | 0.10 | A voxel may be DYNAMIC only if hit on ≥ this fraction of its traversals (`n_hits/n_obs`). Demotes the near-ego scan-plane shell (free space + stray returns) to CARVED_NOISE. 0.0 = off |
| `dynamic_min_range_m` | 2.5 | Points within this horizontal range of the sensor are never dynamic (ego self-returns + max carving). Applies to every seg method. 0.0 = off |
| `max_ray_length_m` | 80.0 | Rays beyond this range are truncated (noise dominates at long range) |
| `free_space_margin_voxels` | 1.0 | Stop free-space carving N voxels before the endpoint |
| `ground_endpoint_strategy` | `"skip_endpoint"` | `"skip_endpoint"` (traverse ground rays but skip `l_occ` at endpoint) or `"skip_ray"` (skip ground rays entirely; legacy) |

**Persistence parameters** (active when `classification_method: persistence`):

| Parameter | Default | Description |
|---|---|---|
| `static_sweep_fraction` | 0.30 | Fraction of chunk sweeps a voxel must be occupied in to be static |
| `static_sweep_min` | 5 | Minimum sweep count regardless of fraction |

### Step B (seg=mos) — MF-MOS

Active only when `segmentation: mos` (or `--seg mos`).

| Parameter | Default | Description |
|---|---|---|
| `mf_mos.checkpoint_path` | `/data/models/mf_mos/mf_mos_semantic_kitti.pt` | Path to pretrained model checkpoint |
| `mf_mos.arch_config` | `/data/models/mf_mos/arch_cfg.yaml` | MF-MOS architecture config |
| `mf_mos.data_config` | `/data/models/mf_mos/data_cfg.yaml` | MF-MOS data config (range image dims, FoV) |
| `mf_mos.residual_steps` | `[1, 2, 4, 8]` | Past-sweep offsets for residual channels |
| `mf_mos.range_image_h` | 32 | Range image height (32 for NuScenes, 64 for KITTI) |
| `mf_mos.range_image_w` | 1024 | Range image width |
| `mf_mos.fov_up_deg` | 10.0 | LiDAR vertical FoV upper bound (NuScenes default) |
| `mf_mos.fov_down_deg` | -30.0 | LiDAR vertical FoV lower bound (NuScenes default) |
| `mf_mos.device` | `"cuda"` | Inference device (`"cpu"` for smoke tests) |
| `mf_mos.score_threshold` | 0.5 | Moving-probability threshold for the binary mask |
| `mf_mos.save_scores` | `false` | Also write float32 `_mf_mos_score.npy` per sweep |
| `mf_mos.max_pose_gap_ms` | 200.0 | Max pose-interpolation gap when warping a historical sweep |
| `mf_mos.max_residual_gap_ms` | 1000.0 | Max residual time baseline. Keep above `max(residual_steps) * sweep_dt` or long channels get zeroed |

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
│   ├── viz.py                         # multi-backend (open3d/plotly/matplotlib) point-cloud viewer
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
│   ├── mf_mos/                        # Step B (seg=mos) — MF-MOS, self-contained
│   │   ├── __init__.py                # public: process_chunk, MFMosResult,
│   │   │                              #         classify_chunk, MosSegmentResult
│   │   ├── _core.py                   # range projection, residual computation, mask writing
│   │   ├── _runtime.py                # model loading, PyTorch inference (lazy import)
│   │   └── segment.py                 # classify_chunk: masks → static/dynamic clouds (no AW)
│   │
│   ├── classify/                      # Step B (seg=aw) — Amanatides-Woo, self-contained
│   │   ├── __init__.py                # public: process_chunk, ClassifyResult
│   │   ├── pipeline.py                # two-pass orchestration (pure AW; no MF-MOS)
│   │   ├── log_odds.py                # build_log_odds_grid (AW Pass 1),
│   │   │                              # classify_from_log_odds (thresholds + classification)
│   │   ├── masking.py                 # apply_classification_to_sweep (Pass 2 per-sweep masks)
│   │   ├── persistence.py             # classify_persistence (sweep-count fallback)
│   │   ├── io_helpers.py              # load_world_full, origin_from_index
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
    ├── test_motion_filter.py          # seg=union post-veto persistence + coherence gates
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
CUDA 12.8. On the `seg=aw` path (default), PyTorch is never imported at
runtime. The MF-MOS runtime (`mf_mos/_runtime.py`) uses a lazy import that
fails with a clear message if torch is absent but `--seg mos` is requested.

**MF-MOS vendored code is a git submodule.** The model definition lives in
`third_party/MF-MOS` (`SCNU-RISLAB/MF-MOS`, pinned). It must be checked out
before `--seg mos` will run — otherwise `_runtime.py` fails with
`ModuleNotFoundError: No module named 'modules'`:

```bash
git submodule update --init src/lidar_preprocessing/third_party/MF-MOS
```

In dev mode the host checkout is bind-mounted into the container, so this is
all you need. For a non-dev image build, run the submodule init **before**
`./watod build` — the Dockerfile `COPY src/lidar_preprocessing` bakes in
whatever the host has at build time. (`seg=aw` needs none of this.)

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
