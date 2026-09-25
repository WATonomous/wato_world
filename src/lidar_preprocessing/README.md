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
                            Amanatides-Woo ray traversal; constants derived
                            from the sensor model. No MF-MOS, no model.

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
D.   reduce/          bag-level global static map + ground grid
                      (optional --two-pass: a global-map prior re-runs B)
    │
    ▼
E.   iwu/             bag-level UniLiPs Iterative Weighted Update: refines the
                      global static map, evicts floaters (parked-then-moved
                      objects) → global_iwu.npz
    │
    ▼
F.   motion_proposals/ per-chunk recall-oriented moving-object proposals:
                      every heuristic's verdict per point (source_bits) +
                      HDBSCAN clusters with soft motion features (Chen et al.)
```

D, E and F run automatically at the end of a whole-bag `run` (`--no-proposals`
skips E and F); each is also its own subcommand (`reduce`, `iwu`, `proposals`)
for multi-machine runs. E and F are seg-agnostic: they consume whatever Step B
method produced.

### Two dynamic artifacts — which one to read

| Artifact | Semantics | Use it for |
|---|---|---|
| `lidar_proc/*_dynamic_mask.npy`, `dynamic_map.npz` | **Precision.** The chosen seg method's verdict (aw / mos / union, incl. vetoes and the motion filter). | Excluding movers: depth anchors (perception_2d), static-map building, anything that must trust "not dynamic". |
| `lidar_proc/*_motion_proposals.npz`, `motion_clusters.parquet` | **Recall.** Every point *any* heuristic calls a mover, and per-cluster soft motion features. False positives expected. | Proposing objects: proposal_generation, tracking seeds, SAM2 LiDAR prompts. Threshold the soft features downstream. |

Never widen one into the other: a false positive in `dynamic_mask` costs
perception_2d a depth anchor, while a false negative in `motion_proposals` loses
an object for good.

## Visualization

The default viewer is a standalone HTML file. It shows static and dynamic
points together and does not require Open3D, DISPLAY forwarding, or an X server.

```bash
./watod run lidar_preprocessing viz \
  --bag <bag_id> --chunk <chunk_id> --open
```

Without `--open`, the command prints the generated file path. Use `--sweep N`
to inspect one classified sweep or `--out PATH` to choose the output location.

Browser controls:

- `view`: top, isometric, side, or front.
- `mode`: one sweep, a five-sweep trail, or all dynamic points.
- `color`: static/dynamic, sweep ID, height, or intensity.
- `prev`, `next`, `play`, and the sweep slider: move through time.
- `static`, `dynamic`, and `point`: toggle layers and change point size.

Other workflows remain available from the same command:

```bash
# Serve larger point buffers from a local HTTP server.
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --backend web

# Export classifier fields for CloudCompare or ParaView.
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --export ply

# Use a native viewer for an individual pipeline artifact.
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> \
  --backend open3d --layer dynamic
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> \
  --backend matplotlib --layer ground
wato_lidar_preprocessing viz --bag <bag_id> \
  --backend plotly --layer global

# Step F motion proposals (chunk-level HTML): grey = attach-only
# (AMBIGUOUS/UNMAPPED), yellow = seed source, magenta = IWU_EVICTED,
# orange = BOX_FILL, red = member of a cluster with motion_score > 1.
wato_lidar_preprocessing viz --bag <bag_id> --chunk <chunk_id> --layer proposals
```

HTML is the default because it covers the normal classification-debugging loop.
The native backends remain for ground-grid and global-map views that the HTML
viewer does not yet implement. PLY files include `dynamic`, `sweep_id`,
`intensity`, `p_occ`, `n_obs`, `n_hits`, and `classification` scalar fields;
missing optional values are `-1`.

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

**Deskew is only as good as the pose samples it interpolates between.** The
interpolation is linear in time between the two `poses.parquet` samples around
each point — constant velocity is assumed, not measured. The samples are read
through `wato_common.pose_lookup`, the loader every component shares. Ingest therefore
requires a dense, smooth pose stream (ingest README, "Pose requirements"): it
aborts on sparse streams such as eidos's keyframe-rate `slam/odometry` (one
pose per 5 m), and marks a sweep `valid_pose=False` when the samples around it
are > 250 ms apart or imply a jump. Deskew skips those sweeps (below) rather
than smearing them.

**Where the per-point times come from** (`deskew/_core.py` `_deskew_sweep`):

1. The raw NPZ has a time field with at least one non-zero value → header +
   that offset, scaled by `point_time_unit`.
2. No usable time field — missing, or present but all zero — and
   `synthesize_per_point_times: true` (both profiles) → offsets synthesized
   from each point's azimuth over the profile's rotation period, anchored at
   the first point in the cloud. `header_stamp_at` says where the header stamp
   sits in the rotation:
   - `sweep_start` counts forward from the stamp (nuScenes; unverified —
     camera trigger offsets only locate the cut angle, and a rotation starts
     and ends at the same angle).
   - `sweep_end` counts back from it: the first-fired point is one rotation
     before the stamp (WATO: the car's Velodyne driver sets
     `timestamp_first_packet: false`, and each sweep is recorded 1.5 ms after
     its stamp, which a start-of-sweep stamp can't be).
3. Neither → the sweep fails with `deskew_failed`, unless
   `allow_uncompensated_motion: true`, in which case every point uses the
   header time (no deskew).

**WATO sweeps take path 2.** Their Velodyne clouds carry a per-point `time`
field, but every value in it is 0 (all 153 sweeps sampled across
`lidar_cc`/`lidar_ne`/`lidar_nw` on `ring_road_corrected`). Without
compensation the car moves ~42 cm (median) during one 50 ms sweep there, so a
point 30 m away was misplaced by 52 cm median, 83 cm p95. Deskew logs how many
sweeps per chunk had an all-zero field.

**Known limitation — VLP-32C seam points.** The azimuth anchor is the first
point in the cloud. The VLP-32C's lasers point at slightly different
horizontal angles, so a few points of the first firing sit just behind that
anchor. They are then timed at the end of the rotation instead of the start:
~0.5 % of `lidar_cc` points, off by one rotation (~40 cm at WATO speeds). Using
array position to unwrap them was not done because nuScenes clouds are
ordered differently (their last points wrap past the start angle), and
unwrapping there is unverified.

The `deskewed` column in `lidar_proc_index.parquet` is True when per-point
times were applied (path 1 or 2), and False only when every point got the
header pose (path 3 with `allow_uncompensated_motion`).

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
| `has_intensity`, `deskewed` | bool | Feature flags. `deskewed` = per-point times were applied, from the sweep's own time field or synthesized from azimuth |
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
each past-sweep offset (derived from the scanner's rate — see "Not
configurable" below), the current range image minus the ego-motion-warped
historical sweep. A moving object leaves a nonzero residual
after ego-motion correction; a static wall does not. The multi-frame residuals
are concatenated and fed to a lightweight encoder-decoder; per-pixel moving
probability above `score_threshold` is labeled moving and unprojected to points.

**Key design points:**
- Masks are raw-length (before nonfinite filtering) so they align to raw NPZs.
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

- **Two distinct time caps** (do not confuse them):
  - `max_pose_gap_ms` gates *pose-interpolation* quality when warping.
  - `max_residual_gap_ms` caps the *sweep-to-sweep residual baseline*. A
    residual at offset `k` spans `k * sweep_dt`, which for the longer offsets
    legitimately exceeds `max_pose_gap_ms`. These were once the same knob,
    which silently zeroed every long residual channel and collapsed
    multi-frame MOS to ~2 live channels. Offsets are rate-scaled to KITTI's
    10 Hz, so the longest baseline is ~800 ms on any rig; keep
    `max_residual_gap_ms` above that.
- A sweep with no usable mask is left entirely static (never fabricates dynamics).
- Per-sweep speckle is removed at mask-generation time by a 3D
  connected-component denoise (`moving_cluster_voxel_m`,
  `moving_min_cluster_pts`); temporal confirmation of movers is the
  downstream tracker's job.

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
- **Pass 2**: apply the resulting `static_arr` / `dynamic_arr` via
  searchsorted to each sweep, drop points within `dynamic_min_range_m` of the
  sensor from the dynamic side, write the dynamic mask, and accumulate
  static/dynamic clouds. On `seg=union` the mask is also snapshotted to
  `aw_dynamic_mask.npy`, since union later overwrites `dynamic_mask.npy`.

**Voxel key encoding.** Each voxel `(vx, vy, vz)` is encoded into a single
`int64` as `vx << 40 | vy << 20 | vz` (20 bits per axis), supporting a ±524 km
range per axis at 0.15 m resolution. All sorted arrays support O(log K) lookup
via `np.searchsorted` — no Python dict overhead in Pass 2.

**Outputs:**

| Artifact | Description |
|---|---|
| `lidar_proc/<sweep_id:06d>_dynamic_mask.npy` | `bool[N]`, True = dynamic point |
| `static_map.npz` | Accumulated static cloud: `xyz` (float64, M×3), `intensity`, `voxel_size`, `origin`, `static_voxel_keys`, `dynamic_voxel_keys` (the carved-dynamic voxel set Step C intersects against), `ambiguous_voxel_keys` (evidenced, hit, p_occ between the two thresholds — used by neither cloud; Step F's AW_AMBIGUOUS) |
| `dynamic_map.npz` | Accumulated dynamic cloud: `xyz` (float64, M×3), `sweep_id` (int32, M), `intensity` (when present) |
| `voxel_occupancy.npz` | Sparse int32 voxel coords for SAM4D / MinkUNet (all sweeps aggregated). Toggle via `save_voxel_occupancy` (default: true). |
| `voxel_occupancy_frame_NNNN.npz` | Per-frame sparse voxel coords (what `perception_2d` feeds to MinkUNet). Written when `save_per_frame_voxel_occupancy: true`. |
| `voxel_diag.npz` | Per-voxel `log_odds` / `p_occ` / `n_obs` / `n_hits` / `classification` for every touched voxel, including carved ones. Toggle via `save_voxel_diagnostics`. |
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

**Optional two-pass global-map prior (`--two-pass`, off by default).** Re-runs
Step B's AW classify on every chunk with the freshly reduced map as a prior:
a one-time, credibility-weighted log-odds boost for voxels matched in the bag
map (`classify/global_map_prior.py`). It improves static recall on structure
sparsely seen in any one chunk and sharpens `union`'s static veto, at roughly
2× classify wall time. It was previously labelled "UniLiPs IWU"; it is not —
IWU is Step E.

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

### Step E — UniLiPs Iterative Weighted Update (`iwu/`)

**What it does.** Gives every point of `global_static_map.npz` a static
probability *P* and updates it sweep by sweep (UniLiPs, arXiv 2601.05105,
Eqs. 3–4, α = 0.7):

```
reinforce:  P ← α·P + (1−α)·r*·(1+C)          a return lands within the match radius
decay:      P ← α·P + (1−α)·(1−r*)·(1−C)      the sweep saw THROUGH the point
```

Points ending below τ = 0.5 (with ≥ `min_observations` updates) are
**evicted**: floaters — parked-then-moved cars, pedestrians that stood still
for a chunk. A single chunk's log-odds grid calls those static because the
evidence it sees says so; IWU compares every sweep of the *bag* against the
bag map, so an object that is static in one chunk and gone in another loses.
That is exactly the `union` failure its `veto_score_exempt` knob works around.

**Hybrid rule (deviations from the paper, each measured or argued):**

- *Reinforce every supported map point*, not only each return's nearest one:
  the map is voxel-snapped at the same 0.30 m pitch as the match radius, so
  "nearest" starves map points a 32-beam scan lands between.
- *Decay only on explicit free-space evidence.* The paper decays a return's
  nearest map point whenever it is > 30 cm away, which lets a pedestrian in
  front of a wall erode the wall and cannot tell "occluded" from "gone".
  Here a point decays only if **every** return whose ray passes within
  ρ = match radius + half-sweep sensor travel of it ended beyond it: a
  range-adaptive min filter over a world-aligned full-sphere range image
  (UniLiPs' own Eq. 1 min-over-neighbourhood device), widened by one ring and
  one column for ring/grid aliasing. Occluded points, points with no nearby
  return, and points within ≈ 9 m (too close for one mid-sweep origin to
  resolve) are not updated.
- *r\** is the sensor model's beam-footprint credibility (crossover
  `global_map_voxel_size_m / divergence`, ≈ 100 m) rather than the paper's
  fixed r_max = 200 m; *C* = 0 (no semantics here; `consensus` is the hook for
  semantic_lifting's label counts later).
- Sweeps are sampled per lidar at `iwu.update_rate_hz` (4 Hz): the EMA is
  dominated by its last ~10 updates, so consecutive 20 Hz sweeps are nearly
  redundant.

**Measured (nuScenes scene-0061, one 19 s chunk, `seg=union`, 191k map
points, 77 sampled sweeps, 9 s):** the paper-literal single-pixel test evicted
54% of an AW-static map; the rule above evicts 7.0%, and eviction is 2.0× more
likely within 0.5 m of the union dynamic cloud than elsewhere (the precision
proxy available without labels). Against the IWU-refined map, `IWU_EVICTED`
proposals leak +20.6 pts above chance onto static structure — the cleanest
seed source (see Step F). EMA semantics are order-dependent: *P* reads as
"consistent with the most recent looks".

**Output:** `raw/<bag_id>/global_iwu.npz` — `xyz` (the global map), `p_static`,
`n_match`, `n_seen_through`, `evicted`, plus the constants it ran with. A
separate file so the post-run re-reduce can never overwrite it.

---

### Step F — Motion proposals (`motion_proposals/`)

**What it does.** Flags every point any heuristic calls a mover, clusters
them per frame, and scores each cluster's motion — the offline MOS
auto-labeling recipe of Chen et al. (arXiv 2201.04501: coarse dynamics →
HDBSCAN → Kalman/Hungarian tracking → "moved further than its own size"),
changed in two ways because this is a *proposal* source:

- **Nothing is dropped on motion evidence.** Chen relabels non-moving
  clusters static; here every cluster keeps a `motion_clusters.parquet` row
  with soft features for downstream to threshold.
- **The coarse stage is a union of every heuristic**, not one map-cleaning
  method, and each point records which fired:

| bit | name | role | fires when |
|---|---|---|---|
| 0 | `AW_DYNAMIC` | seed | voxel in `static_map.npz:dynamic_voxel_keys` (aw, union) |
| 1 | `AW_AMBIGUOUS` | attach | voxel in `static_map.npz:ambiguous_voxel_keys` (aw, union) |
| 2 | `IWU_EVICTED` | seed | within the match radius of an IWU-evicted map point |
| 3 | `MF_MOS` | seed | raw MF-MOS moving mask (mos, union) |
| 4 | `SEG_DYNAMIC` | seed | the run's final `dynamic_mask.npy` (any method) |
| 5 | `BOX_FILL` | — | inside the (ground-extended) box of a moving cluster |
| 6 | `UNMAPPED` | attach | no IWU-refined static-map point nearby (UniLiPs' "no correspondence in the refined map"; falls back to the chunk static cloud when IWU hasn't run) |

*Seed* bits start HDBSCAN clusters (`min_cluster_pts` = Chen's N_min 5,
clusters longer than `max_side_m` = Chen's T_size 20 m are structure). *Attach*
bits never seed — AMBIGUOUS covers vegetation and fences, UNMAPPED every
unmapped surface — they only join a cluster whose box they fall inside.
Every source is optional: a missing artifact just leaves its bit clear (logged
once per chunk). Patchwork++ ground, near-ego points (`dynamic_min_range_m`)
and points lower than `min_height_above_ground_m` are never flagged.

**Clustering and tracking.** Clusters are formed per `frame_sync` frame (the
three WATO sweeps of one tick form one object; single-lidar bags use one frame
per sweep and one tracker per lidar). Boxes are BEV min-area rectangles.
Association uses the shared `wato_common.tracking` primitives — Chen's cost
(centre distance + 1−IoU + volume ratio, gates 2 m / 0.95 / 0.7), Hungarian
assignment, a constant-velocity Kalman filter, `n_old` = 5 — the same code the
`tracking` component is meant to build on, so the two never disagree about
what "moved" means.

**Per-cluster features** (`motion_clusters.parquet`, `MotionClusterRow`):

| Field | Meaning |
|---|---|
| `motion_score` | BEV distance between the median Kalman-filtered centre of the first and last 3 frames of the track ÷ largest box side the track ever showed. Chen: moving if > 1. Net, not path length, so jitter does not accumulate. |
| `track_life`, `track_hint_id` | Frames in the geometry-only track; the id is chunk-local and **not** an identity for the tracking component. |
| `n_sources`, `source_bits` | How many / which heuristics fired inside the cluster (excl. BOX_FILL). |
| `frac_seg_dynamic` | Fraction of members the seg method itself called dynamic. |
| `frac_persistent` | Fraction of members in voxels occupied ≥ `union.motion_filter.persistence_max_sweeps` sweeps — the persistence statistic `union`'s motion filter gates on, here a feature. |
| `box_filled` | Whether BOX_FILL painted it (below). |
| `cx cy cz w l h heading` | Box, in `ProposalRow`'s column names so a row maps 1:1 to a proposal (`provenance="lidar_mos"`). |

**BOX_FILL** (Chen's box fill, recall): non-ground, non-near-ego points
inside a cluster's box — grown down to the ground by the height floor, so the
wheels and feet the floor removed come back — get bit 5, for clusters with
`motion_score > 1`, `track_life ≥ 3` **and** `frac_persistent < 0.5`. The
last condition is ours: size-normalised displacement alone is fooled by a thin
wall fragment whose visible window slides further than the fragment is long
(nuScenes: a 1.6×0.1 m fragment drifting 3 m, `frac_persistent` 1.0; the
clear movers there sat ≤ 0.11). It is the same wall-sliding effect that sank
the translating-cluster rescue tested for `union` (see Motion filter above).
The cost is recall on slow movers that linger in their voxels — a
pedestrian-sized track at 0.52 went unfilled — and it only affects the
painting: such a cluster's row still carries its `motion_score`.

**Measured (same nuScenes chunk, 382 sweeps, 47 s):** 5.4% of points carry a
bit; 21.6k clusters (57/frame), 404 with `motion_score > 1`, 12 box-filled
tracks. On-static leakage above chance (`compare_seg_dynamic --proposals`,
reference = IWU-refined map): IWU_EVICTED +20.6, BOX_FILL +21.4, moving
clusters +25.6, SEG_DYNAMIC +31.2, MF_MOS +55.7, AW_DYNAMIC +64.1,
AW_AMBIGUOUS +72.7 pts — the union's leakage comes from the AW bits, which
hug surfaces exactly as the `union` notes warn. Downstream should weight
clusters by `n_sources`, `frac_persistent` and `motion_score`, not trust any
bit alone.

**Outputs:**

| Artifact | Description |
|---|---|
| `lidar_proc/<sweep_id:06d>_motion_proposals.npz` | `source_bits` uint8[N], `cluster_id` int32[N] (−1 = none); aligned to the world NPZ. Deterministic path (`artifact_store.motion_proposals_path`), not an index column. |
| `motion_clusters.parquet` | One `MotionClusterRow` per cluster. Written last — it is the completion marker. |
| `lidar_proc_summary.parquet` | Row updated in place: `n_points_proposal`, `n_clusters`, `n_clusters_moving`. |

A chunk is re-run when `motion_clusters.parquet` is older than its summary,
index, static map or the bag's `global_iwu.npz` (or with `--force`).

---

## Inputs

| Input | Source | Notes |
|---|---|---|
| `chunks/index.parquet` | ingest | Chunk window timestamps; drives the main loop |
| `chunks/<chunk_id>/lidar_sweeps.parquet` | ingest | Per-sweep metadata |
| `chunks/<chunk_id>/lidar/<sweep_id:06d>.npz` | ingest | Raw sensor-frame point cloud. `sweep_id` is unique within the chunk across all LiDARs, which is why every `lidar_proc/<sweep_id>_*` path can key on it alone. Chunks ingested before ingest made it chunk-unique restart it per LiDAR, so on a multi-LiDAR rig their world files overwrite each other — re-run ingest on them. |
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
| `chunks/<chunk_id>/static_map.npz` | Accumulated static cloud + static / dynamic / ambiguous voxel-key sets |
| `chunks/<chunk_id>/dynamic_map.npz` | Accumulated dynamic cloud + `sweep_id` per point |
| `chunks/<chunk_id>/voxel_occupancy.npz` | Sparse int32 voxel coords, all sweeps aggregated |
| `chunks/<chunk_id>/voxel_occupancy_frame_NNNN.npz` | Per-frame sparse voxel coords (when `save_per_frame_voxel_occupancy: true`) |
| `chunks/<chunk_id>/voxel_diag.npz` | Per-voxel log-odds diagnostics incl. carved voxels (when `save_voxel_diagnostics: true`) |
| `chunks/<chunk_id>/ground.npz` | Height grid, normal grid, ground points |
| `global_static_map.npz` | Bag-level downsampled static cloud (from `reduce`) |
| `global_ground.npz` | Bag-level height grid + normal grid (from `reduce`) |
| `global_iwu.npz` | Step E: per-map-point static probability, update counts, `evicted` floaters |
| `chunks/<chunk_id>/lidar_proc/<sweep_id:06d>_motion_proposals.npz` | Step F: per-point `source_bits` + `cluster_id` (recall-oriented; see "Two dynamic artifacts") |
| `chunks/<chunk_id>/motion_clusters.parquet` | Step F: per-cluster boxes + soft motion features |
| `chunks/<chunk_id>/manifest_lidar_preprocessing.json` | Traceability record: image provenance, content-hashed ingest inputs, outputs, config hash. Step F rewrites it to add `motion_clusters` and the `global_iwu` input. |

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
| `segmentation_method` | str (nullable) | `aw` / `mos` / `union` — which Step-B method produced the chunk; the skip check compares it with the current `--seg` |
| `seg_n_sweeps_no_mask` | int64 (nullable) | mos/union: sweeps with no usable MF-MOS mask |
| `union_n_points_vetoed`, `union_n_points_ground_vetoed` | int64 (nullable) | union: candidates removed by the AW-static and ground-height vetoes |
| `motion_filter_n_persistence_dropped`, `motion_filter_n_coherence_dropped` | int64 (nullable) | union: points removed by the motion filter's gates |
| `n_points_proposal`, `n_clusters`, `n_clusters_moving` | int64 (nullable) | Step F: points with any source bit, clusters, clusters with `motion_score > 1`; null until `proposals` ran |
| `mf_mos_n_processed` | int64 (nullable) | Sweeps processed by MF-MOS |
| `mf_mos_n_skipped` | int64 (nullable) | Sweeps MF-MOS **failed** on: deskew-invalid, pose gap, empty cloud, inference error |
| `mf_mos_n_unsupported` | int64 (nullable) | Sweeps from scanners below `MIN_BEAMS` (e.g. VLP-16) — skipped by design, not a failure |
| `mf_mos_n_points_moving` | int64 (nullable) | Total points labeled moving across all sweeps |

The three sweep counts add up to `n_sweeps_valid`. All `mf_mos_*` fields are
null on `--seg aw`, where MF-MOS never runs — "didn't run" is distinct from
"ran, found nothing". On the WATO rig expect `mf_mos_n_unsupported` ≈ 2/3 of valid sweeps:
only the VLP-32C centre lidar is projectable.

## How to run

```bash
# Build the image (includes pypatchworkpp C++ build, ~3-5 min first time).
./watod build

# Process all chunks of a bag (steps A + B + C per chunk), then the bag-level
# reduce (D), IWU (E) and motion proposals (F).
./watod run lidar_preprocessing --bag data/bags/NuScenes-v1.0-mini-scene-1100/
./watod run lidar_preprocessing --bag NuScenes_v1_0_mini_scene_1100   # equivalent

# Pick the Step-B segmentation method (default from config; aw if unset).
./watod run lidar_preprocessing --bag <bag> --seg aw     # Amanatides-Woo only
./watod run lidar_preprocessing --bag <bag> --seg mos    # MF-MOS only (needs GPU + weights)
./watod run lidar_preprocessing --bag <bag> --seg union  # fusion: aw static + MF-MOS dynamic vetoed by it

# Score how much of a method's dynamic cloud is actually static structure
# (lower = cleaner); run after each --seg to A/B them on the same chunk:
python -m wato_lidar_preprocessing.scripts.compare_seg_dynamic <bag> 0000
# ...or Step F's proposals, one row per source bit:
python -m wato_lidar_preprocessing.scripts.compare_seg_dynamic <bag> 0000 --proposals

# Skip Steps E/F, or add the (off-by-default) two-pass global-map prior.
./watod run lidar_preprocessing --bag <bag> --no-proposals
./watod run lidar_preprocessing --bag <bag> --two-pass

# Process a single chunk only (auto-reduce and IWU are skipped on single-chunk
# runs; Step F runs for that chunk using any existing global_iwu.npz).
./watod run lidar_preprocessing --bag data/bags/NuScenes-v1.0-mini-scene-1100/ --chunk 0000

# Re-process already-completed chunks (e.g. after a code change).
./watod run lidar_preprocessing --bag data/bags/NuScenes-v1.0-mini-scene-1100/ --force

# Disable auto-reduce when processing chunks across multiple machines.
./watod run lidar_preprocessing --bag <bag> --no-auto-reduce
./watod -t lidar_preprocessing_dev   # open a shell in the dev container
python -m wato_lidar_preprocessing reduce --bag NuScenes_v1_0_mini_scene_1100
python -m wato_lidar_preprocessing iwu --bag NuScenes_v1_0_mini_scene_1100
python -m wato_lidar_preprocessing proposals --bag NuScenes_v1_0_mini_scene_1100 --workers 4

# WATO 3-Velodyne rig bags: use the rig profile (per-corner lidars, velodyne
# sensor model, frame_sync).  Ingest the bag with ingest.wato.yaml first.
./watod run lidar_preprocessing --bag <wato_bag> \
    --config /ws/src/lidar_preprocessing/config/lidar_preprocessing.wato.yaml

# Run tests.
./watod test lidar_preprocessing
```

**Local development.** The full suite needs `numba` and `pypatchworkpp`, which
the image has. On a bare host without them, a large part of the suite fails
with `ImportError` (classify, MF-MOS fusion, deskew/pipeline integration). Only
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
The "Default" columns below are the **schema** defaults (what an omitted key
gets). The shipped profiles override a few: both set `voxel_size_m: 0.25` and
`save_voxel_diagnostics: true`; the nuScenes profile sets
`sensor_model.profile: hdl32e`; both set `mf_mos.score_threshold: 0.7`,
`save_scores: true` and `max_pose_gap_ms: 6000`.

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

Each profile also carries its datasheet `firing_cycle_us`; with the spin rate it
fixes the azimuth step (`azimuth_res_deg`, e.g. VLP-32C at 20 Hz → 0.40°),
which sizes IWU's range image.

### Steps E / F — IWU and motion proposals

| Parameter | Default | Description |
|---|---|---|
| `iwu.enabled` | `true` | Run Step E after reduce on whole-bag runs. |
| `iwu.update_rate_hz` | `4.0` | IWU updates per second per lidar; each lidar's stride is `round(datasheet rate / this)`. α, τ and P₀ are the paper's and live in `iwu/_core.py`; the match radius is `global_map_voxel_size_m`. |
| `motion_proposals.enabled` | `true` | Run Step F. |
| `motion_proposals.min_height_above_ground_m` | `0.25` | Points lower than this over the ground grid are never flagged (road false positives); BOX_FILL recovers the slice on moving clusters. 0 = off. |
| `motion_proposals.min_cluster_pts` | `5` | Chen N_min — HDBSCAN `min_cluster_size`. |
| `motion_proposals.max_side_m` | `20.0` | Chen T_size — longer clusters are structure. |
| `motion_proposals.box_fill` | `true` | Chen box fill (with the track-life and persistence conditions above). |

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

### Step B (seg=aw) — Classification

| Parameter | Default | Description |
|---|---|---|
| `voxel_size_m` | 0.15 | Voxel side length for static/dynamic classification (m). A real trade-off with no datasheet answer: smaller is more faithful but more sensitive to pose drift and beam spacing. |
| `min_observations` | 3 | The only evidence gate: voxels with fewer ray traversals stay under-evidenced (neither static nor dynamic). "Was anything ever measured here?" needs no threshold and is not configurable. |
| `dynamic_min_range_m` | 2.5 | Points within this horizontal range of the sensor are never dynamic (ego self-returns + maximal carving). Applies to every seg method. 0.0 = off. |

### Step B (seg=mos) — MF-MOS

Active only when `segmentation: mos` (or `--seg mos`).

| Parameter | Default | Description |
|---|---|---|
| `mf_mos.checkpoint_path` | `/data/models/mf_mos/mf_mos_semantic_kitti.pt` | Path to pretrained model checkpoint |
| `mf_mos.arch_config` | `/data/models/mf_mos/arch_cfg.yaml` | MF-MOS architecture config |
| `mf_mos.data_config` | `/data/models/mf_mos/data_cfg.yaml` | MF-MOS data config (normalisation stats) |
| `mf_mos.device` | `"cuda"` | Inference device (`"cpu"` for smoke tests) |
| `mf_mos.score_threshold` | 0.5 | Moving-probability threshold for the binary mask |
| `mf_mos.save_scores` | `false` | Also write float32 `_mf_mos_score.npy` per sweep |
| `mf_mos.max_pose_gap_ms` | 200.0 | Max pose-interpolation gap when warping a historical sweep |
| `mf_mos.max_residual_gap_ms` | 1000.0 | Max residual time baseline. Keep above the longest derived offset's span (~800 ms) or long channels get zeroed |
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
| `global_map_voxel_size_m` | 0.30 | Voxel size for global static map downsampling (m). Doubles as the match radius of everything compared against that map — the two-pass prior, IWU (the paper's 30 cm) and Step F's IWU_EVICTED / UNMAPPED — since reduce snaps points to voxel centres, "within one map voxel" is what a match means. |
| `point_time_unit` | `"seconds"` | Unit of `t_offset_us` field: `"seconds"` \| `"microseconds"` \| `"nanoseconds"` |
| `synthesize_per_point_times` | `true` | Synthesize per-point timestamps from azimuth when the raw NPZ lacks them or its time field is all zero. The rotation period and direction come from the sensor profile. |
| `header_stamp_at` | `"sweep_start"` | Where the header stamp sits in the rotation for synthesized times: `"sweep_start"` (first-fired point at the stamp) or `"sweep_end"` (last-fired point at the stamp). A driver property, set per dataset profile: `sweep_end` in `lidar_preprocessing.wato.yaml`. Unused when the sweep has real per-point times. |
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
│   ├── cli.py                         # Click CLI: run, reduce, iwu, proposals, viz
│   ├── config.py                      # Pydantic schema: ComponentConfig, MFMosParams, etc.
│   ├── sensor_model.py                # datasheet profiles → derived classifier constants
│   ├── pipeline.py                    # orchestration: deskew → Step B (--seg aw|mos|union) → ground;
│   │                                  # run_proposals(): Steps E + F
│   ├── range_image.py                 # spherical projection shared by mf_mos/ and iwu/
│   ├── voxel.py                       # shared voxel-key packing: voxel_indices(), pack_voxel_key()
│   ├── io.py                          # reader helpers for downstream components
│   ├── viz.py                         # multi-backend (open3d/plotly/matplotlib) point-cloud viewer
│   ├── html_viz.py                    # standalone WebGL HTML viewer (the default `viz` backend)
│   ├── web_viz.py                     # local browser backend with streamed buffers
│   ├── viz_data.py                    # shared data adapters for the viz backends
│   ├── viz_export.py                  # external export helpers for viz data
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
│   │   ├── log_odds.py                # build_log_odds_grid (AW Pass 1 + normals + global prior),
│   │   │                              # classify_from_log_odds (static_arr / dynamic_arr)
│   │   ├── masking.py                 # apply_classification_to_sweep (Pass 2 per-sweep masks)
│   │   ├── global_map_prior.py        # bag-level KDTree prior for two-pass mode
│   │   ├── io_helpers.py              # load_world_full, origin_from_index
│   │   └── occupancy_export.py        # write_chunk_voxel_occupancy, write_per_frame_voxel_occupancy,
│   │                                  # write_chunk_voxel_diagnostics
│   │
│   ├── union/                         # Step B (seg=union) — fusion of aw + mos
│   │   ├── __init__.py                # public: classify_chunk, UnionSegmentResult
│   │   ├── segment.py                 # aw-static veto, ground-height veto, near-ego gate
│   │   └── motion_filter.py           # post-veto persistence + coherence gates
│   │
│   ├── ground/                        # Step C — ground mask aggregation + height grid
│   │   ├── __init__.py                # public: process_chunk, GroundResult
│   │   └── _core.py                   # ground-dynamic intersection, height grid builder
│   │
│   ├── reduce/                        # Step D — bag-level global static map
│   │   ├── __init__.py                # public: reduce_static_map, reduce_ground_map
│   │   └── _core.py                   # voxel-snap downsample, global height grid
│   │
│   ├── iwu/                           # Step E — UniLiPs IWU over the bag static map
│   │   ├── __init__.py                # public: run_iwu, update_with_sweep, load_global_iwu
│   │   └── _core.py                   # hybrid reinforce/decay, windowed seen-through test
│   │
│   └── motion_proposals/              # Step F — recall-oriented moving-object proposals
│       ├── __init__.py                # public: process_chunk, bit constants, decode_bits
│       └── _core.py                   # source bits, HDBSCAN, tracking features, box fill
│
└── tests/
    ├── test_sensor_model.py           # profile sanity, derived constants, per-lidar resolution
    ├── test_deskew.py                 # per-point world projection, 6 extrinsic configurations
    ├── test_classify.py               # log-odds classification, dynamic-default regressions,
    │                                  # near-range gate, union snapshot
    ├── test_mf_mos.py                 # range projection, residuals, seg=mos split, denoise, priming
    ├── test_union.py                  # seg=union vetoes, snapshots, re-fusion
    ├── test_motion_filter.py          # seg=union post-veto persistence + coherence gates
    ├── test_ray_traversal.py          # AW kernel parity (Numba vs Python), voxel traversal
    ├── test_ground.py                 # flat/tilted planes, height grid, dynamic intersection
    ├── test_global_map_prior.py       # two-pass global-map prior, range weighting
    ├── test_iwu.py                    # Step E: EMA, eviction, occlusion, window semantics
    ├── test_motion_proposals.py       # Step F: bits, clusters, motion score, box fill
    ├── test_run_proposals.py          # Steps E+F orchestration, idempotency, manifest
    ├── test_pipeline.py               # chunk summary, cache auto-disable, parallel workers
    ├── test_cli.py                    # CLI flags (--seg, viz --open)
    ├── test_viz_backends.py           # html/web viz backends
    ├── test_reduce.py                 # two-chunk merge, downsampling, partial-run handling
    └── _staging.py                    # artifact staging helpers for the E/F tests
```

The box / Kalman / association primitives Step F tracks with live in
`src/common/src/wato_common/tracking/` (tests: `src/common/tests/test_tracking.py`).

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

- **`test_iwu.py`:** Step E — the EMA against a hand computation, a
  repeatedly seen wall staying static, a parked-then-gone object evicted
  while the wall survives, occluded map points never decayed (the paper's
  literal rule would), the seen-through window (a return on the next ring
  vetoes; sensor travel widens the window; too-close points are never
  decayed), reinforce-all, sweep dedup across chunk overlaps, and the
  per-lidar stride.

- **`test_motion_proposals.py`:** Step F end to end on a synthetic scene: a
  mover scores > 1 and is box-filled (including its wheels, below the height
  floor), a parked car and a sliding wall window stay below 1, ground and
  near-ego points are never flagged, each missing source clears only its own
  bit, outputs stay length-aligned, and staleness tracks the inputs.

- **`test_run_proposals.py`:** Steps E + F orchestration — IWU then F with
  the manifest recording both, `--chunk` skipping IWU, idempotency until
  `--force`.

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

**scikit-learn (Step F):** HDBSCAN comes from `sklearn.cluster` (already in
the lock; imported lazily, only by Step F).

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

- **VLP-32C seam points** (Step A, "Known limitation").
- **Per-point ray origins for the DDA and IWU.** Deskew stores one mid-sweep
  `origin` per sweep; at 20 Hz with fast ego motion the true sensor position
  drifts up to ±(v·25 ms) within a sweep, slightly mis-tracing carve paths near
  the vehicle. IWU pays for the same thing: its seen-through window is widened
  by that travel and it cannot judge points within ≈ 9 m. Deskew already
  interpolates per-point poses, so exporting per-point (or per-time-bucket)
  origins is mechanical.
- **Semantic consensus in IWU.** UniLiPs' C(m) term (label consensus per map
  point) is 0 here. Once semantic_lifting accumulates `(label, count)` over
  the map, pass it as `consensus` to `update_with_sweep`.
- **Consolidate the coherence gate** onto `wato_common.tracking` after
  re-measuring it with `compare_seg_dynamic` — it is tuned against its own
  greedy linker today.
- **perception_2d depth anchors** use `~dynamic_mask`, which also admits
  ambiguous, under-evidenced and near-ego points (and, on `union`, is
  MF-MOS-derived). Static-voxel membership would be the stricter anchor set.
- **Validate MF-MOS on the WATO rig.** The released checkpoint is KITTI-trained
  (64-beam) and both configs default to `segmentation: aw` until mask quality
  on `lidar_cc` has been audited (`scripts/debug_mfmos_contribution.py`). A
  finetuned or re-projected model would justify `--seg union` as the default. Worth measuring
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
