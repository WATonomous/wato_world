# ingest

Ingest is the boundary between ROS bag data and the artifact-based labeling
pipeline. It reads a rosbag2 recording once, normalizes the raw sensor streams,
and writes durable files that every downstream component can consume without
depending on ROS or rosbag2 APIs.

The component makes the recordings reproducible: bag metadata, virtual chunk windows, decoded
camera frames, LiDAR sweeps, ego poses, frame alignment, quality tags, and
traceability manifests.

## Purpose

- Register a bag under a stable `bag_id`.
- Validate that required camera, LiDAR, and pose topics are present.
- Freeze per-bag calibration into the artifact tree.
- Split the bag into virtual chunks without copying or rewriting the bag.
- Decode raw sensor messages into files and Parquet indexes.
- Build the `frame_index.parquet` table that aligns each LiDAR sweep with the
  nearest camera frame per camera and interpolated ego pose.
- Write quality metrics and manifests so downstream reruns know what they read.

## Inputs

| Input | Where it comes from | Notes |
|---|---|---|
| rosbag2 recording | `data/bags/<bag>` or a bind-mounted path | The bag remains the source of truth. |
| Ingest config | `src/ingest/config/ingest.yaml` (component-owned) | Chunk size, topic mapping, timing tolerance, quality thresholds. Override per host with `--config <path>`. |
| Calibration | Auto-extracted from the bag's `/camera_*/camera_info` + `/tf_static` | Override with `--calibration <file.json>` when the bag's intrinsics or extrinsics are wrong. |
| Artifact root | `ARTIFACT_ROOT_URI`, defaulting to `file:///data/artifacts` | Backed by `wato_common.artifact_store`. |

## Outputs

All outputs are written under `data/artifacts/raw/<bag_id>/`.

| Artifact | Purpose |
|---|---|
| `bag_meta.json` | Source path, duration, topic counts, storage backend, and bag metadata. |
| `calibration.json` | Frozen calibration used for this bag. |
| `chunks/index.parquet` | Virtual chunk windows, including overlap ranges. |
| `chunks/<chunk_id>/camera_frames.parquet` | One row per decoded camera image. |
| `chunks/<chunk_id>/lidar_sweeps.parquet` | One row per decoded LiDAR sweep `.npz`. |
| `chunks/<chunk_id>/poses.parquet` | Sparse ego poses copied from `topics.pose` (default: eidos `slam/odometry`). One row per Odometry message in the chunk window. |
| `chunks/<chunk_id>/frame_index.parquet` | LiDAR sweep to camera-frame and pose alignment table. |
| `chunks/<chunk_id>/quality.json` | Per-chunk quality metrics and tags. |
| `chunks/<chunk_id>/manifest.json` | Inputs, config path, artifact paths, and rerun traceability. |

The artifact tree is the metadata index for the current pipeline. There is no
database service dependency.

## Sub-System Diagram

```text
rosbag2 + ingest config + optional calibration
        |
        v
inputs/
  bags.py          -> inspect and register the bag
  topics.py        -> validate configured topics
  chunks.py        -> compute virtual chunk windows
  calibration.py   -> freeze authored calibration
        |
        v
decoders/
  cameras.py       -> image files + camera_frames.parquet
  lidar.py         -> sweep .npz files + lidar_sweeps.parquet
  poses.py         -> sparse ego poses
        |
        v
artifacts/
  frame_index.py   -> synchronized sensor index
  quality.py       -> quality metrics and tags
  manifest.py      -> traceability metadata
        |
        v
data/artifacts/raw/<bag_id>/
```

## Calibration

Ingest auto-builds `calibration.json` from data already in the bag:

- **Intrinsics + distortion + image dimensions + frame_id** come from the first
  `sensor_msgs/CameraInfo` message on each `/camera_*/camera_info` topic.
- **LiDAR frame_id** comes from the first `sensor_msgs/PointCloud2` per LiDAR.
- **Extrinsics (`ego_T_cam`, `ego_T_lidar`)** are resolved by walking
  `/tf_static` from `ego_frame` (default `base_footprint`) to each sensor's
  `frame_id`.  Multi-hop chains (`base -> roof_rack -> camera`) are composed
  automatically.
- **All static transforms** are dumped to `static_transforms` for debugging.

Default behaviour: `pipeline.run_bag()` calls `calibration.freeze_from_bag(...)`.
No hand-authoring required.

Override path: pass `--calibration path/to/file.json` to substitute an authored
calibration when the bag's intrinsics/extrinsics are missing or wrong.  This
calls `freeze_from_file(...)` instead.

`calibration.json` schema:

```json
{
  "calibration_version": "auto_from_<bag_id>",
  "ego_frame": "base_footprint",
  "cameras": {
    "CAM_LOWER_NE": {
      "frame_id": "camera_lower_ne",
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "distortion": [...],
      "distortion_model": "plumb_bob",
      "width": 1920, "height": 1080,
      "ego_T_cam": [[...]]   // 4x4 row-major, or null if /tf_static path missing
    }
  },
  "lidars": { "LIDAR_CC": { "frame_id": "lidar_cc", "ego_T_lidar": [[...]] } },
  "static_transforms": { "base_footprint__camera_lower_ne": [[...]], ... },
  "checks": { "sanity": "auto" | "warn" | "skipped", "notes": "..." }
}
```

## Pose source

Pose comes from a single `nav_msgs/Odometry` topic, configured by `topics.pose`
in `config/ingest.yaml`. The canonical source is **eidos's `slam/odometry`**
(see [`wato_monorepo/src/world_modeling/eidos`](../../wato_monorepo/src/world_modeling/eidos/README.md)),
which gives:

- Pose in the SLAM-optimized **`map` frame**, child frame **`base_footprint`**.
- Stamps at the **LiDAR keyframe sensor time** (not wall-clock), so interpolating
  to a LiDAR sweep timestamp is well-defined.
- Loop-closure-corrected and globally-consistent, unlike raw INS.

Ingest does **not** consume `/tf` or `/tf_static` for pose. Eidos doesn't
publish to TF; that's `eidos_transform`'s job, and its TF stream is
wall-clock-stamped which would desync from LiDAR sweeps.

If a bag predates eidos integration:

1. **Preferred**: run eidos against the bag in `wato_monorepo` to produce a bag
   with `slam/odometry` populated, then point ingest at that bag.
2. **Temporary fallback**: edit `topics.pose` to `/novatel/oem7/odom` (raw INS).
   This is degraded — no SLAM optimization, no loop closure — and should be
   replaced with eidos output before any auto-labeling outputs are trusted.

### How pose flows through ingest

```
slam/odometry (eidos)
    │  one row per message
    ▼
poses.parquet                         ← decoders/poses.py
    │  sparse, ~10 Hz
    ▼
pose_interpolation.interpolate_at()   ← linear translation + SLERP rotation
    │  evaluated at every LiDAR sweep timestamp
    ▼
frame_index.parquet                   ← artifacts/frame_index.py
    └─ each row carries world_T_ego_flat (16 floats, row-major 4×4)
```

Every downstream component reads `world_T_ego_flat` from `frame_index.parquet`
— never `poses.parquet` directly:

- `lidar_preprocessing` uses it for sweep deskewing and multi-sweep aggregation in the world frame.
- `proposal_generation` uses it to project LiDAR points into camera images and lift 2D masks into 3D.
- `tracking` uses it to track in the world frame and stitch tracks across chunk boundaries.
- `label_refinement` uses it to smooth box trajectories.

## How To Run

Use the repo entrypoint for normal development. The container gets the same
mounts and environment as the rest of the pipeline.

```bash
# Build or start the ingest service.
watod -c ingest build
watod -c ingest up

# Run ingest end-to-end on a bag.  The path is the in-container view of
# <repo>/data/bags/ — that directory is bind-mounted at /data/bags, so a
# host path like /home/you/wato_world/data/bags/example will NOT resolve
# inside the container.
watod run ingest /data/bags/example
# Equivalent flag form (both are accepted by watod-run.sh):
watod run ingest --bag /data/bags/example

# Run one chunk after chunks/index.parquet already exists.  Extra flags
# after the bag arg are forwarded to `python -m wato_ingest run`.
watod run ingest /data/bags/example --chunk chunk_000000

# Run tests in the dev container.
watod test ingest
```

The Python entrypoint is useful for local debugging and for rerunning one part
of the workflow.

```bash
python -m wato_ingest inspect-bag --bag /data/bags/example
python -m wato_ingest run --bag /data/bags/example --calibration /config/calibration.json
python -m wato_ingest split --bag /data/bags/example --bag-id example
python -m wato_ingest decode-chunk --bag /data/bags/example --bag-id example --chunk-id chunk_000000
python -m wato_ingest build-frame-index --bag-id example --chunk-id chunk_000000
python -m wato_ingest quality --bag-id example --chunk-id chunk_000000
```

## Package Layout

```text
src/ingest/
|-- config/
|   `-- ingest.yaml        # parameter values (topics, chunk size, thresholds)
|-- src/wato_ingest/
|   |-- cli.py             # Click CLI; calls leaf functions, prints JSON
|   |-- config.py          # Pydantic schema for config/ingest.yaml
|   |-- pipeline.py        # top-level orchestration for a full bag run
|   |-- inputs/            # bag registration, topic checks, chunking, calibration
|   |-- decoders/          # raw ROS message decoding into files and tables
|   `-- artifacts/         # derived indexes, quality reports, manifests
`-- tests/
```

New code should go into the package that owns the artifact boundary:

- Put bag-level validation and source discovery under `inputs/`.
- Put message decoding and file materialization under `decoders/`.
- Put tables derived from already-decoded artifacts under `artifacts/`.
- Keep `pipeline.py` thin. It should sequence operations, not own decoding logic.
- Keep `cli.py` thin. It should parse options, call package functions, and print
  machine-readable summaries.

## Configuration

All ingest parameters live in [`config/ingest.yaml`](config/ingest.yaml).
The Python module [`config.py`](src/wato_ingest/config.py) only defines the
Pydantic schema — values are never hardcoded there.

What lives in the YAML:

- `chunk_seconds` and `chunk_overlap_seconds` — virtual chunk size and overlap.
- `reference_clock` — sensor stream used as the alignment reference.
- `max_cam_offset_ms` — maximum allowed camera-to-LiDAR pairing delta.
- `storage_id` — rosbag2 storage backend, usually `sqlite3`.
- `topics` — mapping from logical sensor names (used in artifact paths) to
  bag topic names.  **Edit this section per recording** if your bag uses
  different topic names — no Python change required.
- `quality_thresholds` — drive the tags emitted in `quality.json`.

To use a different config (e.g. per-host overrides), pass `--config <path>`
to any subcommand.  The container's default is `/ws/src/ingest/config/ingest.yaml`,
which gets there via `COPY src/ingest /ws/src/ingest` (deploy mode) or via
the source bind-mount (dev mode).

Downstream components should depend on the artifact schema, not on the original
bag topic names.
