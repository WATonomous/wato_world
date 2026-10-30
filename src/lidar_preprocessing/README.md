# lidar_preprocessing

Turns ingest's raw sensor-frame sweeps into deskewed world-frame sweeps, a
per-point static/dynamic split, a ground height grid, bag-level static and
ground maps, and recall-oriented motion proposals.

## Running

```bash
./watod run lidar_preprocessing --bag <bag>                 # Steps A–C per chunk, then D, E, F
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

## Steps

```
A  deskew/            per-point motion compensation → world-frame sweep (+ Patchwork++ ground mask)
B  classify/          Amanatides-Woo log-odds static/dynamic split, then a point-level dynamic filter
C  ground/            per-sweep ground masks → chunk height grid
D  reduce/            bag-level global static map + ground grid
E  iwu/               bag-level UniLiPs IWU: evict floaters from the global map
F  motion_proposals/  per-point source bits + HDBSCAN clusters with motion features
```

A chunk counts as done when its summary exists, so a run after a Step B change
needs `--force`.

### Two dynamic artifacts — which one to read

| Artifact | Semantics | Use it for |
|---|---|---|
| `lidar_proc/*_dynamic_mask.npy`, `dynamic_map.npz` | **Precision.** Step B's verdict after its filter. `~mask` is trusted static. | Excluding movers: depth anchors (perception_2d), static maps. |
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

Each sweep's rays are traced through a voxel grid (Numba Amanatides-Woo):
`-l_free` along the ray, `+l_occ` at the endpoint. Carving stops a sensor-derived
margin short of the endpoint, is down-weighted ∝ 1/d past the beam-footprint
crossover, and skips two kinds of voxel:

- occupied voxels the ray only grazes;
- voxels within one voxel of a non-ground return from the same frame, any
  scanner. A voxel holding part of a pole, foliage or an edge is also partly
  free, and a ray through the free part says nothing about the surface.

Rays past `max_range_m` skip the carve but still count their hit. Ground rays
carve but add no hit. Every constant comes from the sensor profile
(`sensor_model.py`). Log-odds sum unclamped, so the verdict is the chunk's total
evidence whatever the sweep order. Then, for voxels with ≥ `min_observations`
traversals and at least one hit:

```
static   p_occ >= p_static (= p_hit)
dynamic  p_occ <  p_dynamic (= 1 - p_hit)
```

A point is a dynamic candidate iff its voxel is dynamic, it isn't ground, and it
lies beyond `dynamic_min_range_m` of the sensor. Never-observed, free-only,
under-evidenced and ambiguous voxels are neither.

`dynamic_filter` then cleans the candidates in memory, before anything is
written, and a point must pass both tests:

- **cluster:** within its frame, the point's connected cluster on a
  `cluster_voxel_m` grid has ≥ `min_cluster_pts` points. Isolated returns are
  mixed pixels and foliage speckle.
- **persistence:** its `persistence_voxel_m` voxel holds candidates for fewer
  sweeps than `persistence_max_s` covers at the chunk's sweep rate, every
  scanner counted. Structure dwells; movers pass through.

The filter touches `dynamic_mask` and `dynamic_map` only. The voxel-key sets in
`static_map.npz` keep the raw verdict, so Step F's `AW_DYNAMIC` still seeds on
every point the filter drops. Tuning is in the
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
`global_ground.npz`, skipping unfinished chunks. `--two-pass` then
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
| 0 | `AW_DYNAMIC` | seed | voxel Step B called dynamic, before its filter |
| 1 | `AW_AMBIGUOUS` | attach | voxel in Step B's ambiguous band |
| 2 | `IWU_EVICTED` | seed | near an IWU-evicted map point |
| 4 | `SEG_DYNAMIC` | seed | the run's `dynamic_mask.npy` |
| 5 | `BOX_FILL` | — | inside the box of a moving cluster |
| 6 | `UNMAPPED` | attach | no IWU-refined (or chunk) static-map point nearby |

Bit 3 is unused; the positions are the on-disk encoding. Seed bits start
clusters; attach bits only join one. A missing source leaves its bit clear. Ground, near-ego points and points below
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
| `lidar_proc/<sweep_id:06d>_motion_proposals.npz` | Step F `source_bits` + `cluster_id` |
| `lidar_proc_index.parquet` | One `ProcessedSweepMeta` per sweep (`valid=False` rows for failures) |
| `lidar_proc_summary.parquet` | One `ChunkSummaryRow`: counts, filter drops, Step F stats |
| `static_map.npz` | Static cloud + static / dynamic / ambiguous voxel-key sets |
| `dynamic_map.npz` | Dynamic cloud + per-point `sweep_id` |
| `ground.npz` | `height_grid`, `normal_grid`, `grid_origin`, `cell_size`, `ground_xyz` |
| `motion_clusters.parquet` | One `MotionClusterRow` per cluster (written last) |
| `voxel_occupancy.npz`, `voxel_diag.npz`, `voxel_occupancy_frame_NNNN.npz` | Optional exports (`save_*` flags); no downstream consumer yet |
| `manifest_lidar_preprocessing.json` | Provenance, hashed inputs, outputs |

Per bag, under `raw/<bag_id>/`: `global_static_map.npz`, `global_ground.npz`,
`global_iwu.npz` (P, update counts, `evicted`).

Schemas: `wato_common.schemas` for the tables, `wato_common.artifacts` for the
cross-component files, `wato_lidar_preprocessing.io` for the rest.

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
size, evidence count, which steps run, the dynamic filter.

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
- **scikit-learn** for Step F's HDBSCAN.

Everything is locked in `docker/requirements/lidar_preprocessing.txt`. The image
is CPU only. Tests need numba and pypatchworkpp, so run them in the container.

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
  ambiguous, under-evidenced and near-ego points, and points the dynamic filter
  dropped. Static-voxel membership would be stricter.
- **Small movers' Step F tracks fragment.** Their points stay in clusters, but a
  pedestrian's motion_score can fall below 1
  ([measurements](../../docs/lidar_preprocessing_measurements.md)).
- **A car parked for most of a chunk reads static there**, even if it then
  leaves: the verdict is the chunk's total evidence. IWU evicts it at bag level.
- **Voxel keys** cover [0, 2²⁰) voxels per axis from the chunk origin (157 km at
  0.15 m) and raise beyond it.
