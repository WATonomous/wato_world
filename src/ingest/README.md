# ingest

Reads a rosbag2 recording once and writes everything downstream needs —
images, LiDAR sweeps, ego poses, calibration, and the `frame_index` that aligns
them — under `data/artifacts/raw/<bag_id>/`. Nothing downstream touches ROS or
the bag.

## Running

```bash
watod run ingest /data/bags/<bag> --config /ws/src/ingest/config/ingest.wato.yaml
watod run ingest /data/bags/<bag> --chunk 0003          # one chunk
watod run ingest validate-bag /data/bags/<bag>.mcap --config <profile> --static-topic /tf_static
watod run ingest trim-bag /data/bags/<bag>.mcap --static-topic /tf_static
watod test ingest
```

`--config` defaults to `ingest.yaml` (nuScenes). `--calibration <file.json>`
replaces calibration from the bag. Individual steps can be rerun with
`python -m wato_ingest <cmd>`: `inspect-bag`, `register`,
`freeze-calibration`, `split`, `decode-chunk`, `build-frame-index`, `quality`,
`manifest`.

| Profile | `topics.pose` | `ego_frame` | Use for |
|---|---|---|---|
| `ingest.yaml` | `/odom` | `base_link` | nuScenes (default) |
| `ingest.wato.yaml` | `/world_modeling/liso/odometry` | `base_footprint` | WATO bags with eidos topics |
| `ingest.wato_6cam.yaml` | same | same | WATO recordings without `camera_pano_ee` / `camera_pano_ww` (May 30 ring-road tests 2–3, `ring_road_corrected`). Must otherwise match `ingest.wato.yaml`; a test checks it. |

## Outputs

Under `data/artifacts/raw/<bag_id>/`:

| Artifact | Contents |
|---|---|
| `bag_meta.json` | Source path, duration, per-topic counts and types, detected storage plugin |
| `calibration.json` | Per-bag calibration (format: `wato_common.artifacts.calibration`) |
| `chunks/index.parquet` | Virtual chunk windows, with overlap ranges |
| `chunks/<chunk_id>/camera_frames.parquet` | One row per decoded image |
| `chunks/<chunk_id>/lidar_sweeps.parquet` | One row per sweep `.npz` (format: `wato_common.artifacts.lidar_sweeps`) |
| `chunks/<chunk_id>/poses.parquet` | Pose samples over the chunk window ± (`max_bracket_ms` + 1 s), each with `interval_drop_reason` for the stretch to the next sample |
| `chunks/<chunk_id>/frame_index.parquet` | One row per (LiDAR sweep, camera): nearest image + the pose at the sweep's time |
| `chunks/<chunk_id>/quality.json` | Per-chunk metrics and tags |
| `chunks/<chunk_id>/manifest.json` | Provenance, hashed inputs, outputs |

Contracts every consumer relies on:

- **`sweep_id` is one counter per chunk across all LiDARs** (record order), so
  `(bag_id, chunk_id, sweep_id)` names one sweep on a multi-LiDAR rig.
- **`frame_index` repeats an image once per sweep it is nearest to** (~4.9× on
  the 3-LiDAR WATO rig). Per-image work collapses rows by
  `(cam_id, camera_seq)` first.
- **`frame_index.world_T_ego` is the sweep's pose.** Projecting into an image
  looks the pose up at `camera_timestamp_ns` through
  `wato_common.pose_lookup.PoseLookup`.

## Ingesting a new bag

Ingest has no per-dataset code: a new recording needs only a new YAML (copy
`config/ingest.yaml`).

1. `python -m wato_ingest inspect-bag --bag <bag>` lists topics and types.
2. Fill in `topics`. Logical names (left) become artifact paths; the right
   side is the bag topic.

   | Config key | Message type |
   |---|---|
   | `cameras.<cam>.image` | `sensor_msgs/CompressedImage` (JPEG/PNG, stored verbatim), or `sensor_msgs/Image` in `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8` (stored as PNG) |
   | `cameras.<cam>.info` | `sensor_msgs/CameraInfo` |
   | `lidars.<lidar>` | `sensor_msgs/PointCloud2` with `x`, `y`, `z`; `intensity`, `ring` and a per-point time field (`t`, `time`, `timestamp`, `time_stamp`, `t_offset_us`) are kept when present. One topic per physical LiDAR, never a merged cloud. |
   | `pose` | `nav_msgs/Odometry`, `geometry_msgs/PoseStamped` or `geometry_msgs/PoseWithCovarianceStamped` |
   | `tf_static` | `tf2_msgs/TFMessage` (`/tf_static`; nuScenes uses `/tf`) |

   `info` and `tf_static` only feed calibration; omit them with
   `--calibration <file>`. Set `cameras.<cam>.rectified: true` for an
   `image_rect*` topic (see [Calibration](#calibration)).
3. Set `ego_frame` to the frame the pose describes (an Odometry's
   `child_frame_id`).
4. Leave `pose_requirements` alone unless the vehicle exceeds
   `max_speed_mps` (30 m/s).

The storage plugin (mcap/sqlite3) is detected; `storage_id` only forces one.

Ingest stops with a named error rather than writing artifacts that look
complete but aren't:

| Problem | Where it stops |
|---|---|
| A configured topic is missing or has a type not in the table | before decoding (`inputs/topics.py`) |
| A camera or LiDAR has no intrinsics or no transform from `ego_frame` | after calibration (`calibration.require_complete`) |
| An image is not JPEG/PNG or uses another raw encoding (Bayer, 16-bit, YUV) | at the first such image (`decoders/cameras.py`) |
| The pose stream is too sparse, or its `child_frame_id` isn't `ego_frame` | per chunk, before images and LiDAR (`decoders/poses.py`) |

It can't check whether different sensors' stamps share one clock, whether
pose stamps are measurement times, or the unit of a LiDAR's per-point time
field (that is lidar_preprocessing's `point_time_unit` / `header_stamp_at`).

## Preparing a bag: `validate-bag` and `trim-bag`

Ingest only checks that topics exist. A camera that started 14 s late, or a
recorder dropout, shows up later as `valid_camera=False` rows. These two
commands catch that first. Both judge every topic on **log (record) time**.

`--static-topic` names a topic that can't span a window (`/tf_static`, and the
one-shot eidos `slam/odometry`, `slam/pose`, `liso/submap` messages); it is
excluded from gaps, coverage and the window. Nothing is exempt automatically:
leave a one-shot topic unnamed and there is no common window.

**`validate-bag`** prints, per topic, coverage (first/last time, rate, time
missing at either end), gaps (consecutive messages more than `--gap-factor` ×
the median period apart, default 1.5), header − log skew, image ↔
`camera_info` pairing, unresolved extrinsics for the profile's sensors, and the
same topic/type check `run` does. It exits 1 on a profile mismatch, an
unresolved profile sensor frame, or no common window. Bursty topics (`/tf`,
`/novatel/oem7/oem7raw`) report thousands of gaps; read that as irregular, not
dropped.

**`trim-bag`** writes `data/bags/trimmed/<bag>_trimmed.mcap` (or `--out`;
`--overwrite` to replace), keeping non-static topics inside [latest first
message, earliest last message] and static topics whole at their original log
time. It copies with the pinned `mcap` CLI, not rosbag2, whose writer would
empty custom-type schemas, and checks every topic's message count before the
output takes its final name. MCAP only. `data/bags/trimmed/` is the one
read-write mount under `/data/bags`. Re-validating a trimmed bag shows a
slightly narrower window (1 Hz topics); don't trim again.

## Calibration

Built from the bag unless `--calibration` is given:

- **Intrinsics** from the first `CameraInfo` per camera. `K` / `distortion`
  always describe the images ingest wrote: `CameraInfo.K` / `D` for a raw
  topic; `CameraInfo.P[:3, :3]` and no distortion for a `rectified: true`
  camera (WATO's `image_rect*` topics — K / D describe the raw sensor image,
  and P's focal length is ~2% lower). A rectified camera with unset `P`, a
  stereo baseline, or a non-identity `R` is refused. The raw CameraInfo is kept
  under `camera_info`.
- **LiDAR frame_id** from the first PointCloud2 per LiDAR.
- **Extrinsics** by a BFS over the `tf_static` topic from `ego_frame` to each
  sensor frame, in either edge direction, composing multi-hop chains.

`require_complete` then stops ingest unless every camera has `ego_T_cam` and
every LiDAR has `ego_T_lidar`. An authored file is parsed by the same reader
and rejected, naming the field, if it lacks one.

## Poses

Every pose in the pipeline — a sweep's, a deskewed point's, a camera frame's —
is interpolated (linear translation, SLERP rotation) between the two
`poses.parquet` samples around it, through `wato_common.pose_lookup`, at that
datum's own timestamp. Constant velocity is assumed in between, so the stream
must be dense and smooth. Ingest applies the rules once, when it writes
`poses.parquet`, and records a verdict for each stretch between samples; every
component honours those verdicts.

| Rule (`pose_requirements`) | Level | Default | On violation |
|---|---|---|---|
| `min_dense_fraction` | chunk | 0.9 | Share of the pose span lying between samples ≤ `max_bracket_ms` apart. Below it ingest **aborts** with `PoseRequirementError`. Coverage, not median spacing, so a bursty stream can't pass. |
| `max_bracket_ms` | stretch | 250 ms | Stretch marked `pose_gap_<ms>ms`; nothing inside gets a valid pose. At 200–250 ms a point 30 m away moves ~11 cm p95. |
| `max_speed_mps` | stretch | 30 m/s | Implied speed above it is a jump (loop closure, INS reset): `pose_jump_<mps>mps`. Measured over ≥ 50 ms, since stamps jitter by milliseconds. Raise for highway data. |

Two structural checks in `decoders/poses.py`:

- **An Odometry's `child_frame_id` must equal `ego_frame`**, or ingest aborts
  (a mismatch on WATO, `base_footprint` vs `base_link`, offsets every point by
  1.76 m). Pose messages name no body frame and are taken on trust.
- **Held positions are dropped**: a sample repeating the previous position
  bit-for-bit while its twist says > 0.5 m/s is an old fix under a new stamp
  (NovAtel: 100 Hz messages, 50 Hz position).

Poses are never extrapolated, so time before the first sample (SLAM not yet
converged) has no pose.

**Which topic.** Passing the checks is necessary, not sufficient — they can't
see a mis-stamped or attitude-biased stream. Measurements:
[docs/ingest_measurements.md](../../docs/ingest_measurements.md).

- `liso/odometry` (eidos front end, one pose per matched `lidar_cc` scan,
  stamped at the scan) is what `ingest.wato.yaml` reads. It is only as dense as
  the recorded scans: `recording_20261004_185825_1` dropped scans and aborts.
- `slam/odometry` is the back end's newest-keyframe pose: one per 5 m on eidos
  `main` (aborts); on `ring_road_corrected` it is LISO poses stamped 244–450 ms
  late, which no constant shift corrects.
- `/novatel/oem7/odom` is not validated for labeling (~2° attitude bias vs the
  LiDAR; mount vs INS error undetermined). No profile ships for it.
- `/tf` is never read: eidos_transform stamps it with wall-clock time.
- A bag without eidos topics: run `wato_monorepo/tools/eidos_offline_to_bag.sh`
  and ingest the `<bag>_with_eidos.mcap` it writes.

## Frame validity

Ingest never deletes `frame_index` rows; it flags them, with a reason, and
downstream decides. Two independent flags:

- **`valid_camera`** / `camera_drop_reason`: `no_camera_frames_for_camera`
  (none in the chunk), or `offset_<N>ms_exceeds_threshold` (nearest image
  farther than `max_cam_offset_ms`; the image is still recorded).
- **`valid_pose`** / `pose_drop_reason`: `no_pose_samples`,
  `outside_pose_span`, `pose_gap_<ms>ms`, `pose_jump_<mps>mps`. Invalid rows
  leave `world_T_ego_flat`, `pose_timestamp_ns` and `pose_interp_error` null.
  The same reasons apply when a consumer looks up a camera frame's pose.

`quality.json` rolls them into tags: `POSE_MISSING` (pose-valid fraction below
`min_pose_availability`), `POSE_JUMPS`, `CAMERA_DROPS`, `STATIONARY`,
`LOW_LIGHT`, `LOW_LIDAR_POINTS`, with pose density and per-reason sweep counts.
The first chunk of a bag is often `STATIONARY` + `POSE_MISSING` (idling while
SLAM converges).

## Code layout

`pipeline.py` sequences; `cli.py` parses options and prints. Bag-level
validation and discovery go in `inputs/`, message decoding in `decoders/`,
tables derived from decoded artifacts in `artifacts/`.
