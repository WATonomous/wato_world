# lidar_preprocessing

Turns ingest's raw sensor-frame sweeps into deskewed world-frame sweeps, a
per-point static/dynamic split, a ground height grid, bag-level static and
ground maps, and recall-oriented motion proposals.

## Running

```bash
./watod run lidar_preprocessing --bag <bag>                 # Steps A–C per chunk, then D, E, F
./watod run lidar_preprocessing --bag <bag> --seg union     # Step B method: aw (default) | mos | union
./watod run lidar_preprocessing --bag <bag> --chunk 0000    # one chunk: no reduce/IWU; F uses any existing global_iwu.npz
./watod run lidar_preprocessing --bag <bag> --force         # redo completed chunks
./watod run lidar_preprocessing --bag <bag> --no-proposals  # skip E and F
./watod run lidar_preprocessing --bag <bag> --two-pass      # add the global-map prior (Step D)
./watod run lidar_preprocessing --bag <wato_bag> \
    --config /ws/src/lidar_preprocessing/config/lidar_preprocessing.wato.yaml
./watod test lidar_preprocessing
```

`--bag` takes a bag_id or a bag path. For multi-machine runs, pass
`--no-auto-reduce` and run the bag-level steps once every chunk is done:
`python -m wato_lidar_preprocessing reduce | iwu | proposals --bag <bag>`.

WATO bags must be ingested with `ingest.wato.yaml` / `ingest.wato_6cam.yaml`
(per-corner LiDAR topics), then processed with `lidar_preprocessing.wato.yaml`.

## Visualization

The default viewer is a standalone HTML file:

```bash
./watod run lidar_preprocessing viz --bag <bag> --chunk <chunk> --open
```

Scroll or two-finger trackpad scroll to zoom, left-drag to orbit, and
right-drag to pan. The `--backend web` viewer uses the same controls.

## Steps

```
A  deskew/            per-point motion compensation → world-frame sweep (+ Patchwork++ ground mask)
B  static/dynamic     --seg aw    classify/  Amanatides-Woo log-odds ray casting
                      --seg mos   mf_mos/    MF-MOS learned moving-object segmentation
                      --seg union union/     AW static map + MF-MOS dynamics vetoed by it
C  ground/            per-sweep ground masks → chunk height grid
D  reduce/            bag-level global static map + ground grid
E  iwu/               bag-level UniLiPs IWU: evict floaters from the global map
F  motion_proposals/  per-point source bits + HDBSCAN clusters with motion features
```

`aw` and `mos` never import each other; `union` imports both. Every method
writes the same artifacts, so C–F don't care which ran. A chunk counts as done
when its summary exists for the same `--seg`, so A/B runs never reuse each
other's outputs.

### Two dynamic artifacts — which one to read

| Artifact | Semantics | Use it for |
|---|---|---|
| `lidar_proc/*_dynamic_mask.npy`, `dynamic_map.npz` | **Precision.** The chosen method's verdict, vetoes included. `~mask` is trusted static. | Excluding movers: depth anchors (perception_2d), static maps. |
| `lidar_proc/*_motion_proposals.npz`, `motion_clusters.parquet` | **Recall.** Every point *any* heuristic calls a mover, plus per-cluster soft features. False positives expected. | Proposing objects: proposal_generation, tracking seeds. Threshold the features downstream. |

Never widen one into the other: a false positive in `dynamic_mask` costs a
depth anchor; a false negative in `motion_proposals` loses an object for good.
Consumers read the world sweep and mask through
`wato_common.artifacts.lidar_sweeps` (which checks they align) and the
proposals through `wato_common.artifacts.motion_proposals`.

### A — Deskew

Each point is placed in the world at its own time: the ego pose is interpolated
(`wato_common.pose_lookup`) at every per-point timestamp, composed with
`ego_T_lidar`. Coordinates are float64. Patchwork++ runs per sweep in sensor
frame first and its mask rides along as `ground_mask`. Sweeps ingest marked
`valid_pose=False` are skipped, as is a LiDAR whose extrinsic is missing; both
get no world file.

Per-point times come from, in order:

1. the sweep's own time field, if any value is non-zero (`point_time_unit`
   converts it);
2. otherwise, with `synthesize_per_point_times: true`, the point's azimuth over
   the profile's rotation period. `header_stamp_at` places the stamp:
   `sweep_start` counts forward (nuScenes; unverified), `sweep_end` counts back
   one rotation (WATO, whose Velodyne `time` fields are all zero);
3. otherwise the sweep fails (`deskew_failed`), unless
   `allow_uncompensated_motion: true` puts every point at the header pose.

`lidar_proc_index.deskewed` is False only in case 3.

### B — Static/dynamic

**`aw` (`classify/`).** Each sweep's rays are traced through a voxel grid
(Numba Amanatides-Woo): `-l_free` along the ray, `+l_occ` at the endpoint.
Carving stops a sensor-derived margin short of the endpoint, is down-weighted
∝ 1/d past the beam-footprint crossover, and skips occupied voxels the ray only
grazes. Rays past `max_range_m` skip the carve but still count their hit.
Ground rays carve but add no hit. Every constant comes from the sensor profile
(`sensor_model.py`). Then, for voxels with ≥ `min_observations` traversals and
at least one hit:

```
static   p_occ >= p_static (= p_hit)
dynamic  p_occ <  p_dynamic (= 1 - p_hit)
```

A point is dynamic iff its voxel is dynamic, it isn't ground, and it lies
beyond `dynamic_min_range_m` of the sensor. Never-observed, free-only,
under-evidenced and ambiguous voxels are neither.

**`mos` (`mf_mos/`).** MF-MOS (KITTI-trained range-image model) predicts a
per-sweep moving mask from range residuals against past sweeps, warped into the
current frame. Residual offsets are rate-scaled to KITTI's 10 Hz, so the
longest baseline is ~800 ms on any rig (`max_residual_gap_ms` must stay above
it). Scanners with fewer than 32 channels (`MIN_BEAMS`; the WATO corner
VLP-16s) are skipped. Masks are denoised by 3D cluster size and stored aligned
to the *raw* sweep. Then:

```
dynamic = mf_mos_moving & ~ground & ~near_ego
static  = ~mf_mos_moving & ~ground
```

A sweep with no usable mask is left all static. `--seg mos` is the raw model
output, so it can be A/B'd against `aw`.

**`union` (`union/`).** AW's static map, verbatim, plus:

```
dynamic = mf_mos_moving & ~ground & ~near_ego & ~near_ground & ~aw_static_dilated
```

`near_ground` is below `union.ground_height_veto_m` over Step C's grid (so Step
C runs before the fusion on this path). The AW-static veto is dilated by
`veto_dilation_voxels`, except for candidates whose own voxel AW called dynamic.
A post-veto motion filter then drops points whose voxel stays occupied for
≥ `persistence_max_sweeps` sweeps (structure dwells; movers pass through). Its
tuning and the alternatives that tested worse are in the
[measurements](../../docs/lidar_preprocessing_measurements.md).

### C — Ground

Unions the per-sweep Patchwork++ masks, drops ground points in voxels Step B
called dynamic (low-riding movers), and rasterises a median-Z height grid with
nearest-filled holes and finite-difference normals. With no ground masks
(Patchwork++ unavailable) it writes a sentinel (`ground_status:
skipped_no_ground_mask`).

### D — Reduce

Merges every chunk's static map into `global_static_map.npz` (one point per
`global_map_voxel_size_m` voxel) and every chunk's ground into
`global_ground.npz`, skipping unfinished chunks. `--two-pass` (aw/union) then
re-classifies every chunk with the global map as a one-time log-odds prior —
roughly 2× classify time, for better recall on sparsely seen far structure. It
is not IWU.

### E — IWU

Every global-map point gets a static probability P, reinforced when a sweep
returns on it and decayed only when a sweep sees *through* it (UniLiPs Eqs. 3–4,
α = 0.7). Points ending below 0.5 with ≥ `min_observations` updates are evicted
as floaters: parked-then-moved cars, pedestrians that stood still for a chunk.
Sweeps are sampled at `iwu.update_rate_hz` per lidar. The deviations from the
paper are documented in `iwu/_core.py`.

### F — Motion proposals

Chen et al.'s offline MOS recipe (coarse candidates → HDBSCAN → Kalman /
Hungarian tracking → "moved further than its own size"), except that nothing is
dropped: every cluster keeps a row with soft features. Each point records which
heuristics fired:

| bit | name | role | fires when |
|---|---|---|---|
| 0 | `AW_DYNAMIC` | seed | voxel AW called dynamic (aw, union) |
| 1 | `AW_AMBIGUOUS` | attach | voxel in AW's ambiguous band (aw, union) |
| 2 | `IWU_EVICTED` | seed | near an IWU-evicted map point |
| 3 | `MF_MOS` | seed | raw MF-MOS mask (mos, union) |
| 4 | `SEG_DYNAMIC` | seed | the run's `dynamic_mask.npy` |
| 5 | `BOX_FILL` | — | inside the box of a moving cluster |
| 6 | `UNMAPPED` | attach | no IWU-refined (or chunk) static-map point nearby |

Seed bits start clusters; attach bits only join one. A missing source leaves its
bit clear. Ground, near-ego points and points below
`min_height_above_ground_m` are never flagged; BOX_FILL restores wheels and feet
on clusters with `motion_score > 1`, `track_life ≥ 3` and
`frac_persistent < 0.5`. Clusters are per `frame_sync` frame, tracked with
`wato_common.tracking`. Feature meanings: `schemas.MotionClusterRow`. A chunk
re-runs when `motion_clusters.parquet` is older than any input, or with
`--force`.

## Inputs

From ingest: `chunks/index.parquet`, and per chunk `lidar_sweeps.parquet` + raw
sweeps, `poses.parquet`, `frame_index.parquet` (`valid_pose`), plus
`calibration.json`. `sweep_id` must be chunk-unique across LiDARs; chunks
ingested before that restart it per LiDAR and must be re-ingested.

## Outputs

Per chunk, under `raw/<bag_id>/chunks/<chunk_id>/`:

| Artifact | Contents |
|---|---|
| `lidar_proc/<sweep_id:06d>_world.npz` | Deskewed sweep: float64 xyz, mid-sweep `origin`, `ground_mask`, intensity, ring |
| `lidar_proc/<sweep_id:06d>_dynamic_mask.npy` | bool, aligned to the world sweep |
| `lidar_proc/<sweep_id:06d>_mf_mos_mask.npy` / `_mf_mos_score.npy` | mos/union: raw-aligned mask; scores when `save_scores` |
| `lidar_proc/<sweep_id:06d>_aw_dynamic_mask.npy` | union: AW's own verdict, for re-fusing |
| `lidar_proc/<sweep_id:06d>_motion_proposals.npz` | Step F `source_bits` + `cluster_id` |
| `lidar_proc_index.parquet` | One `ProcessedSweepMeta` per sweep (`valid=False` rows for failures) |
| `lidar_proc_summary.parquet` | One `ChunkSummaryRow`: counts, method, MF-MOS / union / Step F stats |
| `static_map.npz` | Static cloud + static / dynamic / ambiguous voxel-key sets |
| `dynamic_map.npz` | Dynamic cloud + per-point `sweep_id` |
| `ground.npz` | `height_grid`, `normal_grid`, `grid_origin`, `cell_size`, `ground_xyz` |
| `motion_clusters.parquet` | One `MotionClusterRow` per cluster (written last) |
| `voxel_occupancy.npz`, `voxel_diag.npz`, `voxel_occupancy_frame_NNNN.npz` | Optional exports (`save_*` flags); no downstream consumer yet |
| `manifest_lidar_preprocessing.json` | Provenance, hashed inputs, outputs |

Per bag, under `raw/<bag_id>/`: `global_static_map.npz`, `global_ground.npz`,
`global_iwu.npz` (P, update counts, `evicted`).

Schemas: `wato_common.schemas` for the tables, `wato_common.artifacts` for the
cross-component files, `wato_lidar_preprocessing.io` for the rest. The summary's
`mf_mos_*` fields are null on `--seg aw`; `processed + skipped + unsupported`
equals the valid sweeps (≈ 2/3 unsupported on the WATO rig).

## Configuration

`config/lidar_preprocessing.yaml` (nuScenes, default) and
`config/lidar_preprocessing.wato.yaml` (WATO rig: `vlp32c` centre and `vlp16`
corners via `sensor_model.per_lidar`, `frame_sync.canonical_lidar: lidar_cc`,
`header_stamp_at: sweep_end`). Every field is documented in
[`config.py`](src/wato_lidar_preprocessing/config.py).

**The config states no physics.** Beam count, FoV, spin rate and direction,
range accuracy, divergence, usable range, intensity scale and every log-odds
constant come from the scanner profile in `sensor_model.py` (`vlp32c`, `vlp16`,
`hdl32e`), which also records why its few hand-picked probabilities have the
values they do. The YAML picks scanners and the genuinely free choices: voxel
size, evidence count, which steps run, union's vetoes. The MF-MOS projection
likewise comes from each scanner's profile plus checkpoint constants in
`mf_mos/_core.py`.

`WATO_LIDAR_CACHE_BYTES` (default 4 GB) caps classify's in-memory sweep cache;
a larger chunk is read from disk twice instead.

## Visualization

```bash
./watod run lidar_preprocessing viz --bag <bag> --chunk 0000 --open    # standalone HTML/WebGL
```

`--sweep N` shows one sweep, `--layer proposals` shows Step F (grey = attach,
yellow = seed, magenta = IWU_EVICTED, orange = BOX_FILL, red = moving cluster).
`--backend web` serves larger buffers; `--export ply` writes classifier fields
for CloudCompare. `--backend open3d|plotly|matplotlib` with `--layer
dynamic|static|ground|global` covers the views HTML doesn't (native backends
need a display).

## Dependencies

- **Numba** runs the ray-traversal kernel; classify fails at import without it
  (no slow fallback).
- **Patchwork++** (`pypatchworkpp`, built in the image). With
  `require_patchwork: true` (default), deskew raises without it.
- **MF-MOS** (`mos` / `union` only) needs torch, the weights under
  `/data/models/mf_mos/`, and the pinned submodule:
  `git submodule update --init src/lidar_preprocessing/third_party/MF-MOS`
  (before `./watod build` for a deploy image; dev bind-mounts it).
- **scikit-learn** for Step F's HDBSCAN.

Everything is locked in `docker/requirements/lidar_preprocessing.txt`
(MF-MOS at `ARG MF_MOS_COMMIT`). The lock floated to `torch==2.11.0` +
`cuda-toolkit`, unlike perception_2d's 2.7.1; that's known drift (root
`CLAUDE.md`). Tests need numba and pypatchworkpp, so run them in the container.

## Known limitations

- **VLP-32C seam points**: ~0.5 % of `lidar_cc` points are timed one rotation
  late ([measurements](../../docs/lidar_preprocessing_measurements.md)).
- **One mid-sweep origin per sweep.** Ray traversal and IWU treat a sweep as cast
  from one point; IWU widens its window for that travel and can't judge points
  within ≈ 9 m. Exporting per-point origins from deskew would fix both.
- **IWU's consensus term C(m) is 0** until semantic_lifting provides labels.
- **The coherence gate** is tuned against its own greedy linker, not
  `wato_common.tracking`.
- **perception_2d depth anchors use `~dynamic_mask`**, which also admits
  ambiguous, under-evidenced and near-ego points. Static-voxel membership would
  be stricter.
- **MF-MOS is unvalidated on the WATO rig**; both profiles default to `aw`.
- **Voxel keys** cover [0, 2²⁰) voxels per axis from the chunk origin (157 km at
  0.15 m) and raise beyond it.
