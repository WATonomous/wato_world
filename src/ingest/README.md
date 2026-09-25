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
- Validate that the configured camera, LiDAR, and pose topics are present with
  message types ingest can decode, and that the pose stream is dense and
  smooth enough to interpolate ([Pose requirements](#pose-requirements)).
- Freeze per-bag calibration into the artifact tree, and refuse to continue if
  any configured sensor lacks intrinsics or extrinsics.
- Split the bag into virtual chunks without copying or rewriting the bag.
- Decode raw sensor messages into files and Parquet indexes.
- Build the `frame_index.parquet` table that aligns each LiDAR sweep with the
  nearest camera frame per camera and the ego pose at the sweep's time.
- Write quality metrics and manifests so downstream reruns know what they read.

## Inputs

| Input | Where it comes from | Notes |
|---|---|---|
| rosbag2 recording | `data/bags/<bag>` or a bind-mounted path | The bag remains the source of truth. |
| Ingest config | `src/ingest/config/ingest.yaml` (nuScenes, default), `ingest.wato.yaml` (WATO + dense eidos pose), `ingest.wato_novatel.yaml` (WATO + NovAtel INS pose) | Chunk size, topic mapping, timing tolerance, pose requirements, quality thresholds. Pick one with `--config <path>`. The config is the only per-dataset part of ingest — see [Ingesting a new bag](#ingesting-a-new-bag). |
| Calibration | Auto-extracted from each camera's `info` topic + the `tf_static` topic | Override with `--calibration <file.json>` when the bag's intrinsics or extrinsics are missing or wrong. |
| Artifact root | `ARTIFACT_ROOT_URI`, defaulting to `file:///data/artifacts` | Backed by `wato_common.artifact_store`. |

## Outputs

All outputs are written under `data/artifacts/raw/<bag_id>/`.

| Artifact | Purpose |
|---|---|
| `bag_meta.json` | Source path, duration, topic counts and message types, the storage plugin rosbag2 detected, and bag metadata. |
| `calibration.json` | Frozen calibration used for this bag. |
| `chunks/index.parquet` | Virtual chunk windows, including overlap ranges. |
| `chunks/<chunk_id>/camera_frames.parquet` | One row per decoded camera image. |
| `chunks/<chunk_id>/lidar_sweeps.parquet` | One row per decoded LiDAR sweep `.npz`. `sweep_id` is unique within the chunk across **all** LiDARs (one counter, record order), so `(bag_id, chunk_id, sweep_id)` names one sweep on a multi-LiDAR rig too. |
| `chunks/<chunk_id>/poses.parquet` | Ego poses from `topics.pose`: one row per distinct pose sample in the chunk window, padded on each side by `pose_requirements.max_bracket_ms` + 1 s so sweeps and camera frames at the chunk edges are bracketed even when the pose is recorded late. Held NovAtel repeats are dropped. Each row's `interval_drop_reason` says whether the stretch to the next sample can be interpolated across. Every component looks poses up from this file through `wato_common.pose_lookup`. See [Pose requirements](#pose-requirements). |
| `chunks/<chunk_id>/frame_index.parquet` | LiDAR sweep to camera-frame alignment table, plus the ego pose at each sweep's time. |
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
  topics.py        -> validate configured topics and their message types
  chunks.py        -> compute virtual chunk windows
  calibration.py   -> freeze calibration (bag or file), require it complete
        |
        v
decoders/
  cameras.py       -> image files + camera_frames.parquet
  lidar.py         -> sweep .npz files + lidar_sweeps.parquet
  poses.py         -> ego pose samples + which stretches are trusted
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

## Ingesting a new bag

Ingest has no per-dataset code. A recording from another vehicle or dataset
needs only a new config YAML (copy `config/ingest.yaml`); nothing in Python
changes. nuScenes and the WATO rig already run through the same code this way.

1. List the bag's topics and message types:
   `python -m wato_ingest inspect-bag --bag <bag>`.
2. Fill in `topics`. The logical names on the left (`CAM_FRONT`, `lidar_cc`)
   become artifact paths; the right side is the bag topic. Accepted types:

   | Config key | Message type |
   |---|---|
   | `cameras.<cam>.image` | `sensor_msgs/CompressedImage` holding JPEG or PNG, or `sensor_msgs/Image` in `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8` |
   | `cameras.<cam>.info` | `sensor_msgs/CameraInfo` |
   | `lidars.<lidar>` | `sensor_msgs/PointCloud2` with `x`, `y`, `z`; `intensity`, `ring` and a per-point time field (`t`, `time`, `timestamp`, `time_stamp`, `t_offset_us`) are kept when present. One topic per physical LiDAR, not a merged cloud. |
   | `pose` | `nav_msgs/Odometry`, `geometry_msgs/PoseStamped` or `geometry_msgs/PoseWithCovarianceStamped` |
   | `tf_static` | `tf2_msgs/TFMessage` (often `/tf_static`; nuScenes uses `/tf`) |

   `info` and `tf_static` only feed calibration; leave them out and pass
   `--calibration <file>` for a bag without them.
3. Set `ego_frame` to the frame the pose describes (an Odometry's
   `child_frame_id`). Extrinsics are resolved from it.
4. Leave `pose_requirements` alone unless the vehicle is faster than
   `max_speed_mps` (30 m/s); the rules are physics, not dataset tuning.
5. Run `python -m wato_ingest run --bag <bag> --config <your.yaml>`.

The storage plugin (mcap or sqlite3) is detected from the bag; set
`storage_id` only to force one.

Ingest stops with an error, naming the cause, rather than writing artifacts
that would look complete but aren't:

| Problem | Where it stops |
|---|---|
| A configured topic is missing, or carries a message type not in the table above | before anything is decoded (`inputs/topics.py`) |
| A camera or LiDAR has no intrinsics or no transform from `ego_frame` | after calibration (`calibration.require_complete`) |
| An image is not JPEG/PNG, or a raw encoding outside the list above (Bayer, 16-bit, YUV) | at the first such image (`decoders/cameras.py`) |
| The pose stream is too sparse, or its `child_frame_id` isn't `ego_frame` | per chunk, before images and LiDAR are decoded ([Pose requirements](#pose-requirements)) |

What ingest can't check: whether header stamps of different sensors share one
clock, whether pose stamps are the measurement times (see [How good are the
poses](#how-good-are-the-poses-measured-on-ring_road_corrected)), and the
meaning and unit of a LiDAR's per-point time field. The last one is
lidar_preprocessing's `point_time_unit` / `header_stamp_at` profile settings;
a field holding absolute times rather than offsets from the header stamp is
rejected there.

## Reading the bag

Every decoder reads the bag through `wato_common.io.rosbag_reader.messages`,
which filters by topic and by **record time** (when the message was recorded,
not its header stamp). Ingest reads the bag several times: once for
calibration, and three times per chunk (poses, cameras, LiDAR). Each read
starts at the beginning of the bag and stops once a message is more than 1 s
past the chunk's window.

rosbag2 returns messages in record-time order: through the index when the bag
has one, and in write order when it doesn't. Every bag measured came back in
order, including `ring_road_corrected_0-001.mcap`, which was never finalized
and has no index. An unindexed bag still costs a full scan up to the chunk's
end on every read, because the storage plugin can't skip data for unrequested
topics; on that 19.5 GiB bag a scan of the whole file takes about 35 s.
`mcap recover` writes an indexed copy, if there's disk space for one.

## Calibration

Ingest auto-builds `calibration.json` from data already in the bag:

- **Intrinsics + distortion + image dimensions + frame_id** come from the first
  `sensor_msgs/CameraInfo` message on each camera's `info` topic.
- **LiDAR frame_id** comes from the first `sensor_msgs/PointCloud2` per LiDAR.
- **Extrinsics (`ego_T_cam`, `ego_T_lidar`)** are resolved by walking the
  `tf_static` topic from `ego_frame` to each sensor's `frame_id`, in either
  edge direction.  Multi-hop chains (`base -> roof_rack -> camera`) are
  composed automatically.
- **All static transforms** are dumped to `static_transforms` for debugging.

Default behaviour: `pipeline.run_bag()` calls `calibration.freeze_from_bag(...)`.
No hand-authoring required.

Override path: pass `--calibration path/to/file.json` to substitute an authored
calibration when the bag's intrinsics/extrinsics are missing or wrong.  This
calls `freeze_from_file(...)` instead, and the bag's `info` / `tf_static`
topics are then neither read nor required.

Either way, `calibration.require_complete` then checks that every configured
camera has `K` and `ego_T_cam` and every configured LiDAR has `ego_T_lidar`,
and stops ingest with `CalibrationError` otherwise — no downstream component
can place that sensor's data without them.  `checks.sanity` is still `warn`
for softer problems such as an empty distortion vector.

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
      "ego_T_cam": [[...]]   // 4x4 row-major
    }
  },
  "lidars": { "LIDAR_CC": { "frame_id": "lidar_cc", "ego_T_lidar": [[...]] } },
  "static_transforms": { "base_footprint__camera_lower_ne": [[...]], ... },
  "checks": { "sanity": "auto" | "warn" | "skipped", "notes": "..." }
}
```

## Pose requirements

**Ingest requires a dense, smooth pose stream and refuses anything else.**
Every pose the pipeline uses is linearly interpolated between the two pose
samples around it — translation linearly, rotation by SLERP, by elapsed time:

- a LiDAR sweep's (`frame_index.world_T_ego`),
- each deskewed LiDAR point's (lidar_preprocessing Step A),
- a camera frame's (perception_2d depth anchors, semantic_lifting).

No velocity is used; constant velocity between the two samples is *assumed*.
That is harmless when samples are ~100 ms apart and wrong when they are a
second apart: if the car brakes or stops inside the gap, the interpolated car
keeps creeping, static structure smears, and lidar_preprocessing labels walls
dynamic.

All of these go through one lookup, `wato_common.pose_lookup.PoseLookup`, and
each component asks for the pose at the timestamp of its own data. Ingest
applies the rules below once, when it writes `poses.parquet`, and records the
verdict for every stretch between two consecutive samples
(`interval_drop_reason`). The lookup honours those verdicts, so every component
trusts the same stretches without needing ingest's config.

Three rules, all in `pose_requirements` in the config:

| Rule | Level | Default | On violation |
|---|---|---|---|
| `min_dense_fraction` | chunk | 0.9 | Share of the chunk's pose span (first → last sample) that lies between samples at most `max_bracket_ms` apart. Below it, **ingest aborts** with `PoseRequirementError` before decoding any images or LiDAR for that chunk. Coverage rather than median spacing, so a bursty stream (fixes 2 ms apart, then seconds of nothing) can't pass on paper. |
| `max_bracket_ms` | stretch | 250 ms | The stretch is marked `pose_gap_<ms>ms`; nothing inside it gets a valid pose (a sweep: `valid_pose=False`, `pose_drop_reason=pose_gap_<ms>ms`). Measured on `ring_road_corrected` by thinning `liso/odometry` and interpolating the dropped samples back: for brackets of 200–250 ms, a point 30 m away is displaced by 3.7 cm median, 10.8 cm p95, 40 cm max. Most of that is rotation, not translation. At 350–500 ms that becomes 23 cm p95 and 1.4 m max, about one 0.25 m voxel at p95. The reference is LISO itself, so its own jitter is included. |
| `max_speed_mps` | stretch | 30 m/s | The two samples imply a jump (loop closure, INS reset), not motion. The stretch is marked `pose_jump_<mps>mps`, and nothing inside it gets a valid pose. Speed is measured over at least 50 ms (one sweep): pose stamps jitter by milliseconds, so only a displacement above `max_speed_mps` × 50 ms (1.5 m) can count as a jump. Raise it for highway data. |

Plus two structural checks in [`decoders/poses.py`](src/wato_ingest/decoders/poses.py):

- **Child frame = `ego_frame`.** The pose describes its `child_frame_id`;
  extrinsics are resolved relative to `ego_frame`. If they differ, every point
  is offset by the transform between them — 1.76 m vertically on the WATO rig
  (`base_footprint` → `base_link`). Mismatch aborts ingest.
- **Held positions are dropped.** A sample repeating the previous position
  bit-for-bit while its own twist says the ego is moving (> 0.5 m/s) is an old
  fix under a new stamp. `/novatel/oem7/odom` publishes at 100 Hz but its INS
  position updates at 50 Hz, so every other message is such a repeat; keeping
  them makes the pose stall 10 ms then double its speed. Eidos leaves twist at
  zero, so its streams are never filtered.

The chunk rule catches a *sparse stream*; the stretch rules catch *local*
problems in a dense one (a dropout, a loop-closure jump). A time before the
first sample or after the last (the start of a bag before SLAM converges) has
no pose either, because poses are never extrapolated. Frames failing a rule are
kept with a reason — see [Frame validity & dropping](#frame-validity--dropping).

### Which pose sources pass the checks

Measured on the recordings we have (pose topic only, read-only). The two
`ring_road_corrected` recordings contain identical LiDAR (every `liso/odometry`
stamp in `_0-001` equals a `lidar_cc` header in `_part1`) but different eidos
output (2,274 vs 2,041 `slam/odometry` poses): they are two separate eidos
runs over the same data.

| Recording | Topic | Distinct-pose spacing | Dense fraction | Checks |
|---|---|---|---|---|
| nuScenes mini | `/odom` | 20 ms median, 45 ms max | 1.0 | pass (`ingest.yaml`) |
| `ring_road_corrected_part1` (ingested; bag no longer on disk) | `/world_modeling/slam/odometry` | 100 ms median, 175 ms max | 1.0 (every chunk) | pass (`ingest.wato.yaml`); its largest step, 4.45 m in 100 ms, trips `max_speed_mps` — cause not verifiable without the bag |
| `ring_road_corrected_0-001` | `/world_modeling/slam/odometry` | 100 ms median | 1.0 | pass; largest step 2.46 m in 100 ms, a stamping artifact (below) |
| `ring_road_corrected_0-001` | `/world_modeling/liso/odometry` | 50 ms median, 202 ms max | 1.0 | pass; no jumps |
| `may_30_ring_road_test_3_3` | `/world_modeling/slam/odometry` | 908 ms median, 5.8 s max — one pose per 5 m | 0.0 | **abort** |
| `ring_road_July_1-2` | `/novatel/oem7/odom` | 20 ms median after dropping held repeats (29,755 of 67,797 messages) | 1.0 | pass (`ingest.wato_novatel.yaml`) |
| `may_30_ring_road_test_3_3` | `/novatel/oem7/odom` | 2 ms median, but in bursts: fixes held up to 11.7 s, jumps up to 34 m (INS not converged) | 0.10 | **abort** |

**What the `may_30` abort prevents (measured on that bag).** Its 586
`slam/odometry` messages carry 40 distinct poses, 4.95–5.74 m apart. 39 of
them are stamped exactly at a `lidar_cc` scan header, so the stamps are
correct; the stream is just sparse. The eidos code on `main` is unchanged
since the day before the recording, apart from `package.xml`, and the bag
behaves as that code does. I forced the interpolation anyway and located every
`lidar_cc` sweep between keyframes independently: point-to-plane ICP against
the keyframe scans placed at their eidos poses.

Results, as the displacement of a point 30 m away (821 sweeps; one clear ICP
failure excluded):

| | Median | p95 | Max |
|---|---|---|---|
| All sweeps between keyframes | 0.46 m | 4.4 m | 5.8 m |
| Brackets longer than 1.5 s (the car slowing and stopping) | 3.5 m | 5.6 m | 5.8 m |
| Brackets of 1.5 s or less | 0.31 m | 1.1 m | 1.7 m |

- **Rotation dominates.** At the stop the car turned about 3° right, but
  interpolation swings it up to 7° left toward the next keyframe, which comes
  after a 37° left turn. That is 8.7° of heading error; translation error
  peaks at 1.2 m.
- **The last 309 sweeps get no pose at all.** They cover 15.6 s after the last
  keyframe, with the car stopped, and ingest never extrapolates.
- **How far to trust the method.** Registering each keyframe scan against its
  neighbours reproduces its own pose to 12 cm median, 34 cm max. The stop
  position is the same whether located against the earlier keyframe's scan or
  the later one's (within 13 cm).
- **The independent check only partly agrees.** eidos_transform's odom-frame
  EKF, which uses no LiDAR from this measurement, gives the same right turn at
  the stop (−4.5° vs ICP's −3.1°). It places the stop 0.8 m further along;
  that disagreement is unresolved.

**Passing the checks is necessary, not sufficient.** They look at the pose
stream alone and cannot see a mis-stamped stream or a biased attitude — both
occur on the WATO recordings; see the next section.

### How good are the poses (measured on `ring_road_corrected`)

No ground truth exists for these drives. Each test below states what it
assumes; together they bound the error without trusting any single source.

**1. What `slam/odometry` actually contains (`_0-001`; no assumptions).**
Every one of its 2,041 poses is bit-identical (0.0 mm, 0.000°) to a
`liso/odometry` pose, stamped 244–450 ms (median 310 ms, 90% within
266–357 ms) *after* that LISO pose's scan. Each stamp equals the message's own
recording time, i.e. it is the publish time, and 92% are 100 ms apart. The lag
is LISO's processing latency (239–337 ms, median 286 ms, from its own stamp to
its recording time) plus the wait for the next SLAM tick. Both vary per
message, so no constant shift removes it (section 2, last row). How we know the
LISO stamp is the right one: it equals a `lidar_cc` scan header exactly, and
independently, matching speed against the INS needs −20 ms for LISO but
−325 ms for `slam/odometry`. Between consecutive ticks the
content advances by 0/1/2/3 LISO poses (12/1,166/859/3 times), which is why
its implied speed jumps 4.7 → 18.6 → 4.9 → 14.7 m/s while the INS reads a
steady 9.5 m/s. So in this run `slam/odometry` is `liso/odometry` resampled
at 10 Hz with late, varying timestamps, and **no correction beyond LISO**. Its
largest step (2.46 m in 100 ms) is three LISO poses spanning 252 ms at
9.7 m/s. For per-sweep poses `liso/odometry` is therefore strictly better on
this recording: same numbers, correct times, more of them.

**2. Local motion vs the NovAtel INS (independent of the LiDAR).** Compared
using two frame-free quantities per window: distance travelled, and rotation
angle. Neither depends on how either sensor's frame is mounted. The only
fitted value is one constant time shift per source. This assumes the INS is
locally accurate (it is IMU-driven); two independent sensors agreeing is
evidence for both.

| Stream (`_0-001`) | Best shift | 0.2 s: distance Δ median / p95 | 1 s: distance Δ | 1 s: rotation Δ | 5 s: distance Δ |
|---|---|---|---|---|---|
| `liso/odometry` as stamped | −20 ms | 1.9 / 8.5 cm | 4.8 / 17.1 cm | 0.06° / 0.31° | 11 / 34 cm |
| `slam/odometry` as stamped | −325 ms | 11.4 / 53.3 cm | 23.4 / 64.9 cm | 0.26° / 2.72° | 47 / 202 cm |
| `slam/odometry` shifted −325 ms | | 12.1 / 47.6 cm | 14.4 / 48.4 cm | 0.07° / 0.51° | 18 / 59 cm |

**3. Drift at a revisit (independent of GNSS).** The drive passes the same
spot twice, 115 m of driving apart (15 s and 40 s after LISO starts). Placed
with `liso/odometry`, the second pass needs 7.6–8.3 cm and ≤ 0.12° of ICP
correction to sit on the first. ICP's own floor, measured on a sweep 1 s
later in the same pass, is 4.8 cm and 0.07°. This is the only revisit in the
drive, so it bounds drift over 115 m only.

**4. Whole drive vs GNSS.** The NovAtel fix is autonomous only (NavSatStatus
0, no RTK/SBAS) with self-reported σ ≈ 1 m horizontal and 2 m vertical. After
one rigid 2D alignment over the full 1.4 km, `liso/odometry` differs from it
by 0.38 m median, 1.0 m p95, 1.38 m max horizontally — within the GNSS noise.
So LISO drifts less than roughly 1–2 m over 1.4 km; GNSS this coarse can't
resolve smaller drift.

**5. LiDAR sweep overlap (not independent; reported because it is what
lidar_preprocessing consumes).** The same `lidar_cc` sweeps (chunks 0003,
0006) were projected with each pose, and overlap was measured between sweeps
0.5 s and 2 s apart (nearest-neighbour distance, later sweep → earlier).
Absolute values are inflated by single-sweep sparsity; compare rows. Two
caveats:

- LISO registers exactly these scans, so this metric favours it by
  construction.
- A *constant* pose-time offset is nearly invisible here: under constant
  speed and yaw rate the sweeps shift together. The metric mainly measures
  jitter.

The `slam/odometry` row is from `_part1`'s run.

| Pose source | 0.5 s apart: within 10 cm | 2 s apart: median NN | 2 s apart: within 10 cm |
|---|---|---|---|
| `liso/odometry` | 15–18 % | 0.50–0.52 m | 7.6–8.1 % |
| `/novatel/oem7/odom`, constant attitude correction | 14–17 % | 0.50–0.53 m | 5.6–5.7 % |
| `slam/odometry` (`_part1`) | 8–10 % | 0.55–0.56 m | 3.7–4.4 % |
| `/novatel/oem7/odom` as recorded | 3–5 % | 0.81–0.87 m | 0.3–0.6 % |

**6. NovAtel attitude.** As recorded, NovAtel poses smear LiDAR badly over
2 s. Timing isn't the cause: the best time shift gains nothing. One constant
rotation removes most of it (yaw +1.35°, pitch −1.93°, roll +0.34°); it was
estimated against LISO, so it isn't independent evidence. The INS reports its
own attitude σ on this drive as 1.2° roll/pitch and 8.1° yaw. The rotation
could therefore be INS attitude error rather than a mounting offset, and one
recording can't tell the two apart. A fixed correction is only valid if it's
the mount. Separately, NovAtel's own interpolation is fine: holding out every
other sample and interpolating it back errs by 0.3 cm median (7 cm p95).

**Not measured: eidos `main`'s `slam/odometry` against `liso/odometry`.** No
recording has both. On `main` those poses are keyframe poses after iSAM2 with
GPS and loop-closure factors. With a GNSS of this quality (~1 m autonomous),
GPS factors can move keyframes by up to about a metre, and whether that helps
or hurts at sweep level is unknown.

### Why `liso/odometry` is dense and `slam/odometry` is not

Both come from eidos (`wato_monorepo` `world_modeling/eidos`), at different
stages:

- **`liso/odometry`** is published from LISO's LiDAR callback — once per scan
  that GICP matches — stamped with that scan's header time
  (`liso_factor.cpp` `lidarCallback`). Scans are skipped when GICP falls
  behind: 2,894 of ~4,000 `lidar_cc` scans in `_0-001` (~13 Hz of 19.8 Hz).
- **`slam/odometry`** is the factor graph's view. LISO hands the graph a new
  state only after ≥ 5 m of travel (`min_scan_distance`,
  `liso_factor.cpp:476-486`). GNSS and IMU never create states; they attach
  to existing ones. This keeps iSAM2 real-time and gives loop closure a
  keyframe database. `eidos_node.cpp` `publishOdometry` then republishes the
  newest state's optimized pose on every 10 Hz tick. On `main` it uses that
  state's own stamp, so `may_30`'s 586 messages carry 40 distinct poses. The
  `_0-001` run behaved differently (section 1).

Neither is the final smoothed trajectory; that exists only in the saved
`.map` file, and only at keyframes.

## Pose source

Pose comes from a single topic, `topics.pose`: a `nav_msgs/Odometry`, whose
`child_frame_id` must equal `ego_frame`, or a `geometry_msgs/PoseStamped` /
`PoseWithCovarianceStamped`, which names no body frame, so `ego_frame` is
taken on trust.  A pose that exists only on `/tf` is not read (see the
`poses.py` docstring).

### eidos: `slam/odometry` vs `liso/odometry`

Both are map-frame poses with child `base_footprint`, so either works with
`ingest.wato.yaml`'s `ego_frame`:

| | `/world_modeling/liso/odometry` | `/world_modeling/slam/odometry` |
|---|---|---|
| What it is | LISO front end: each LiDAR scan GICP-registered against a submap of recent keyframe clouds, placed at their graph-optimized poses (IMU gives only the rotation initial guess) | Back end: iSAM2-optimized pose of the **newest keyframe** (LISO between-factors + GPS + IMU + loop closures) |
| When | every matched scan | every 10 Hz SLAM tick |
| Stamp | the scan's header time (all 2,894 in `_0-001` equal a scan header) | `main`: the keyframe's scan time. `_0-001`: ~310 ms after the scan |
| Distinct poses | one per matched scan | `main`: one per keyframe (≥ 5 m apart). `_0-001`: one per tick, each a LISO pose |
| Global corrections | follows them going forward (the submap is rebuilt from corrected keyframes); past poses are never revised | `main`: includes them for the newest keyframe (unmeasured here). `_0-001`: none — identical to LISO |
| Source | `liso_factor.cpp` `lidarCallback` | `eidos_node.cpp` `publishOdometry` |

Where both exist (`_0-001`), `liso/odometry` is the better per-sweep pose
(section 1). Whether GPS/loop-closure-corrected keyframe poses would improve on
it across a whole bag is unmeasured. The way to combine them is to anchor
`liso/odometry` to the keyframe poses: interpolate the slowly varying keyframe
correction and apply it to the per-scan poses. That is not implemented.

**Re-stamping `_0-001`'s `slam/odometry` doesn't help.** A message doesn't
carry the time of the scan its pose came from, and the lag varies by 200 ms.
Its exact time can only be recovered by finding the bit-identical
`liso/odometry` pose, which gives back a 10 Hz subset of `liso/odometry`
(2,029 of 2,894 poses). On `main` the stamp is already the keyframe's scan time
(`eidos_node.cpp` `publishOdometry`); the problem there is density, not
stamping. To use
`liso/odometry`, set `topics.pose: /world_modeling/liso/odometry`; it must be
recorded in the bag (`may_30` didn't record it).

### Profiles

| Profile | `topics.pose` | `ego_frame` | Use for |
|---|---|---|---|
| `ingest.yaml` | `/odom` | `base_link` | nuScenes |
| `ingest.wato.yaml` | `/world_modeling/slam/odometry` | `base_footprint` | WATO bags with eidos topics. Aborts on eidos `main`'s keyframe-rate stream. On `ring_road_corrected` it passes, but the stream is late-stamped LISO (section 1); `liso/odometry` is better there. |
| `ingest.wato_novatel.yaml` | `/novatel/oem7/odom` | `base_link` | WATO bags without eidos topics. **Not validated for labeling**: as recorded its attitude smears LiDAR, and whether that's a fixed mount offset or INS attitude error is undetermined (section 6). |

**The `ring_road_July` bags can't be ingested yet under either profile**: they
record no camera images (only `/camera_pano_nw/camera_info`) and no `lidar_nw`
messages, and ingest requires every configured camera topic. Their NovAtel
pose passes the checks; camera-less ingest is a separate change.

The two WATO profiles are identical apart from `topics.pose` and `ego_frame`
(`tests/test_pose_requirements.py` enforces this — edit them together).

NovAtel is a GNSS+IMU INS pose: no scan matching or loop closure, and in
**absolute UTM coordinates** (x ≈ 5×10⁵, y ≈ 4.8×10⁶ m). lidar_preprocessing's
voxel keys are chunk-relative, so UTM magnitudes are fine there; anything that
casts world coordinates to float32 loses ~0.5 m at that magnitude (the HTML
viz does).

**If a bag has none of these:** replay it through eidos in `wato_monorepo` and
record `liso/odometry`. Keyframe-rate `slam/odometry` alone will abort ingest by
design.

Ingest does **not** consume `/tf` for pose. `eidos_transform`'s TF stream is
wall-clock-stamped, which would desync from LiDAR sweeps.

### How pose flows through ingest

```
topics.pose (Odometry / PoseStamped / PoseWithCovarianceStamped)
    │  read over the chunk window ± (max_bracket_ms + 1 s)
    │  child_frame_id == ego_frame?          else abort (Odometry only)
    │  dedupe stamps, drop held positions
    │  dense fraction >= min_dense_fraction? else abort
    │  mark each stretch between samples: gap > max_bracket_ms or
    │  implied speed > max_speed_mps → interval_drop_reason
    ▼
poses.parquet                         ← decoders/poses.py
    │  dense (10–50 Hz)
    ▼
wato_common.pose_lookup.PoseLookup    ← linear translation + SLERP rotation;
    │                                   invalid inside a marked stretch or
    │                                   outside the sampled span
    ├─ at each LiDAR sweep's time → frame_index.parquet (artifacts/frame_index.py):
    │     world_T_ego_flat (16 floats, row-major 4×4), valid_pose, pose_drop_reason
    ├─ at each point's time       → lidar_preprocessing deskew
    └─ at each camera frame's time → perception_2d, semantic_lifting
```

Downstream components get ego pose in two ways:

- **Through `PoseLookup`, at their own data's timestamp.** lidar_preprocessing
  interpolates per point for deskewing, and still honours `frame_index`'s
  `valid_pose` to skip whole sweeps. perception_2d and semantic_lifting look the
  pose up at `camera_timestamp_ns` to project LiDAR into an image: the car
  moves between a sweep and the nearest image (up to 44 ms apart on the WATO
  rig), so the sweep's pose would misplace a point 30 m away by 17 cm median
  and up to 87 cm. `proposal_generation` should do the same when it projects
  into images.
- **As `frame_index.world_T_ego`, the pose at the sweep's time.** For work in
  the LiDAR sweep's own time: `tracking` (world-frame tracks, stitching across
  chunks) and `label_refinement` (smoothing box trajectories).

## Frame validity & dropping

Ingest never deletes frames. When people say a chunk "dropped N frames" (e.g.
the `dropped` column in `quality`'s summary), they mean `frame_index.parquet`
rows flagged **invalid** — the row is still written, with a reason, so nothing
is silently lost and downstream stages decide whether to skip it.

Two **independent** validity flags live on every `frame_index` row, one per
alignment concern:

| Flag | Set by | Meaning |
|---|---|---|
| `valid_camera` | camera↔sweep pairing in [`artifacts/frame_index.py`](src/wato_ingest/artifacts/frame_index.py) | The sweep has a usable image for this camera. |
| `valid_pose` | `PoseLookup.at` ([`wato_common/pose_lookup.py`](../common/src/wato_common/pose_lookup.py)) at the sweep's time | The sweep lies in a stretch of the pose stream that ingest marked trusted: two samples close together that don't imply a jump. |

They are computed separately: a row can have a valid camera but no pose, or a
valid pose but no camera. `dropped_camera_count` counts only
`valid_camera=False` rows.

### `valid_camera` and `camera_drop_reason`

For each LiDAR sweep, ingest finds the nearest camera frame **per camera** and
compares the timestamp offset against `max_cam_offset_ms`. Invalid rows carry a
human-readable `camera_drop_reason`:

- `no_camera_frames_for_camera` — that camera has no images in the chunk window
  at all (`image_path` and `camera_offset_ms` are null).
- `offset_<N>ms_exceeds_threshold` — the nearest image exists but is farther
  than `max_cam_offset_ms` from the sweep. The image is still recorded in the
  row (with its `camera_offset_ms`), just flagged unusable.

Valid rows have `camera_drop_reason = null`.

### `valid_pose`

`PoseLookup.at()` marks `valid_pose=False` and sets `pose_drop_reason`:

- `no_pose_samples` — the chunk has no pose samples at all.
- `outside_pose_span` — the sweep is before the first or after the last sample
  (poses are never extrapolated; e.g. before SLAM converges at bag start).
- `pose_gap_<ms>ms` — the two samples around the sweep are more than
  `pose_requirements.max_bracket_ms` apart.
- `pose_jump_<mps>mps` — the two samples imply an ego speed above
  `pose_requirements.max_speed_mps` (loop closure, INS reset).

The same reasons apply to a camera frame's pose when perception_2d or
semantic_lifting look it up; such a frame is skipped for LiDAR projection.

A sweep exactly on a sample is always valid. Invalid rows leave
`world_T_ego_flat`, `pose_timestamp_ns`, and `pose_interp_error` null
(`pose_interp_error` is the distance to the nearest sample). Valid rows have
`pose_drop_reason = null`.

### Relationship to quality tags

The per-row flags roll up into the chunk-level tags in `quality.json`:

- **`POSE_MISSING`** — the fraction of `valid_pose=True` rows falls below
  `quality_thresholds.min_pose_availability`.
- **`POSE_JUMPS`** — at least one sweep lost its pose to `pose_jump_*`.
  `quality.json` also records `pose_dense_fraction`, `pose_median_spacing_ms`,
  `pose_max_spacing_ms`, and the number of sweeps dropped per reason (`pose_gap_sweeps`,
  `pose_jump_sweeps`, `pose_outside_span_sweeps`).
- **`STATIONARY`** — mean ego speed over the chunk is below
  `quality_thresholds.stationary_speed_mps`.

These commonly co-occur on the **first chunk of a bag**: the vehicle is still
idling (→ `STATIONARY`) and eidos SLAM has not converged yet, so early sweeps
have no interpolatable pose (→ `POSE_MISSING`, and those same pose-less rows
inflate the dropped count). Mid-drive chunks with converged SLAM and a moving
ego typically tag `OK`. See [Configuration](#configuration) for the thresholds.

## How To Run

Use the repo entrypoint for normal development. The container gets the same
mounts and environment as the rest of the pipeline.

```bash
# Build or start the ingest service.
watod -c ingest build
watod -c ingest up

# Run ingest end-to-end on a bag.
watod run ingest /data/bags/example

# Run one chunk after chunks/index.parquet already exists.
watod run ingest /data/bags/example chunk_000000

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
- `reference_clock` — currently unused: `frame_index` always has one row per
  (sweep of every LiDAR, camera).
- `max_cam_offset_ms` — maximum allowed camera-to-LiDAR pairing delta; beyond
  it a `(sweep, cam)` row is flagged `valid_camera=False`.
- `pose_requirements` — the dense/smooth pose rules (`min_dense_fraction`
  aborts ingest; `max_bracket_ms` and `max_speed_mps` mark untrusted stretches
  of the pose stream, which invalidates every sweep, point or camera frame
  inside them).
  See [Pose requirements](#pose-requirements).
- `storage_id` — optional; rosbag2 detects the storage plugin (mcap, sqlite3)
  from the bag when it's omitted, as in every shipped profile.
- `topics` — mapping from logical sensor names (used in artifact paths) to
  bag topic names and the accepted message types.  **Edit this section per
  recording** — no Python change required
  ([Ingesting a new bag](#ingesting-a-new-bag)).
- `quality_thresholds` — drive the tags emitted in `quality.json`.

To use a different config (e.g. per-host overrides), pass `--config <path>`
to any subcommand.  The container's default is `/ws/src/ingest/config/ingest.yaml`,
which gets there via `COPY src/ingest /ws/src/ingest` (deploy mode) or via
the source bind-mount (dev mode).

Downstream components should depend on the artifact schema, not on the original
bag topic names.
