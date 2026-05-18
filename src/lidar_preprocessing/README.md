# lidar_preprocessing

Turns raw per-sweep `.npz` files from `ingest` into a deskewed, world-frame,
static/dynamic-split, ground-extracted point cloud that every downstream
stage consumes. Step D (`reduce`) aggregates per-chunk outputs into bag-level
maps after all chunks finish.

## Pipeline

```
ingest artifacts
    │
    ▼
A. deskew      per-point motion comp + world-frame projection
               (also runs Patchwork++ per sweep; ground mask saved in world NPZ)
    │
    ▼
B. classify    voxel occupancy → static / dynamic split
    │
    ▼
C. ground      aggregate per-sweep ground masks → height grid + normals
    │
    ▼
D. reduce      [bag-level] global static map + global ground grid
```

## Inputs (from `ingest`)

| Input | Path under `raw/<bag_id>/` |
|---|---|
| chunks index | `chunks/index.parquet` |
| per-sweep raw cloud | `chunks/<chunk>/lidar/<sweep>.npz` |
| sweep metadata | `chunks/<chunk>/lidar_sweeps.parquet` |
| ego poses | `chunks/<chunk>/poses.parquet` |
| calibration | `calibration.json` (`ego_T_lidar` per lidar) |

## Outputs (under `data/artifacts/raw/<bag_id>/`)

Per chunk (`chunks/<chunk_id>/`):

| Artifact | Content |
|---|---|
| `lidar_proc/<sweep>_world.npz` | deskewed world-frame xyz + intensity + ground_mask |
| `lidar_proc/<sweep>_dynamic_mask.npy` | `bool[N]`, True = dynamic |
| `lidar_proc_index.parquet` | per-sweep stats + paths + `frame_id` |
| `lidar_proc_summary.parquet` | per-chunk rollup (point counts, cache info, ground status) |
| `static_map.npz` | accumulated static cloud |
| `dynamic_map.npz` | accumulated dynamic cloud + per-point `sweep_id` |
| `voxel_occupancy.npz` | sparse int32 voxel coords (toggle via `save_voxel_occupancy`) |
| `ground.npz` | height grid + normal grid + raw ground points |

Bag-level (after `reduce`):

| Artifact | Content |
|---|---|
| `global_static_map.npz` | downsampled bag-level static cloud |
| `global_ground.npz` | bag-level height grid + normals (spans all chunks) |

## How to run

```bash
# Edit ACTIVE_MODULES in watod-config.sh, e.g. ACTIVE_MODULES="lidar_preprocessing:dev"
./watod build
./watod run lidar_preprocessing --bag data/bags/<bag>/                 # all chunks → auto-reduce
./watod run lidar_preprocessing --bag <bag_id> --chunk 0000            # single chunk
./watod run lidar_preprocessing --bag <bag_id> --force                 # re-process completed chunks
./watod run lidar_preprocessing --bag <bag_id> --no-auto-reduce        # skip step D
./watod -t lidar_preprocessing_dev                                     # shell into dev container
python3 -m wato_lidar_preprocessing reduce --bag <bag_id>               # manual reduce inside container
```

## How to visualize

Open interactive Open3D / matplotlib windows on the artifacts above. Each
window blocks until closed — press `1/2/3/4` for top/front/side/iso views,
`S/D/G` to toggle layers. Requires `DISPLAY` forwarded into the container
(configured in `modules/docker-compose.dev.yaml`; works under WSLg on WSL2).

```bash
./watod run lidar_preprocessing viz --bag <bag_id>
```

```bash
./watod -t lidar_preprocessing_dev    # open a shell into the dev container

# Inside the container:
python3 -m wato_lidar_preprocessing viz --bag <bag_id>                                  # all chunks, all stages
python3 -m wato_lidar_preprocessing viz --bag <bag_id> --chunk 0000                     # single chunk
python3 -m wato_lidar_preprocessing viz --bag <bag_id> --chunk 0000 --stage C           # ground grid only
python3 -m wato_lidar_preprocessing viz --bag <bag_id> --chunk 0000 --sweep 42 --stage A
python3 -m wato_lidar_preprocessing viz --bag <bag_id> --stage D                        # bag-level (after reduce)


```

Stages:

- **A** — per-sweep world-frame cloud (one window per sweep).
- **B** — chunk-level static (blue) + dynamic (red) classified cloud; press S/D to toggle.
- **C** — height grid + normals in matplotlib.
- **D** — global static map + global ground grid (run `reduce` first).

## Configuration

`config/lidar_preprocessing.yaml`. The knobs that actually matter day-to-day:

| Parameter | Default | Notes |
|---|---|---|
| `voxel_size_m` | 0.15 | Voxel size for static/dynamic classification. |
| `static_sweep_fraction` | 0.30 | Min fraction of sweeps a voxel must be occupied in to be static. |
| `point_time_unit` | `"seconds"` | Unit of `t_offset_us` from ingest. **Wrong unit → wildly displaced points.** Velodyne is seconds; some LiDARs use microseconds/nanoseconds. |
| `frame_sync.canonical_lidar` | `null` | Set to e.g. `"lidar_cc"` for multi-lidar rigs so non-canonical sweeps inherit a shared `frame_id`. |
| `patchwork.sensor_height` | 1.8 | LiDAR mount height (m). Tune if your vehicle differs significantly. |
| `save_per_frame_voxel_occupancy` | `false` | Enable when developing `perception_2d` (SAM4D MinkUNet input). |

Env var: `WATO_LIDAR_CACHE_BYTES` (default 4 GB) caps the in-memory sweep
cache; if a chunk exceeds it, classify does two NPZ loads per sweep instead
of one. `cache_auto_disabled` in the summary parquet tells you which chunks
hit that path.

## Testing

```bash
./watod test lidar_preprocessing                       # in container
# Or locally (Patchwork++ tests skip if pypatchworkpp isn't installed):
PYTHONPATH=src/common/src:src/lidar_preprocessing/src \
    python3 -m pytest src/lidar_preprocessing/tests -q
```
