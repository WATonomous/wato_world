# Drive bag survey: which recordings are complete enough to label

Survey of the shared WATO bag folder on Google Drive
(`127Kw3o7Org474rkK1wwMDRZKiYHaR0Db`), done 2026-10-26. The question: which
recordings have `lidar_cc` plus 6–8 cameras and are good enough to run through
the pipeline? This came up because `recording_20260309_211754_0` turned out to
be useless: it was recorded with the car parked inside the garage.

## How this was checked

Nothing was downloaded in full. The folder holds 3.4 TB in 180 MCAP files
across 46 recordings.

- **Topics and rates.** For each MCAP I read only the summary section at the
  end of the file (about 1–9 MB). That gives every topic, its message type and
  its exact message count. Rates below are message count ÷ recorded seconds.
- **Unfinalized files.** These are files where the recorder was killed, so
  there's no summary. If another file in the same recording had a summary, I
  used that one. Four recordings consist only of unfinalized files. For those I
  scanned two 128 MB blocks from the middle of the file and identified the
  sensors from the `frame_id` and layout of each message.
- **Did the car move?** I sampled `/novatel/oem7/odom` (twist → speed, plus
  position) from up to 8 files per recording, at the start and end of each.
  `/novatel/oem7/fix` was the fallback.
- **What do the cameras see?** I pulled 1–3 `camera_pano_nn` frames from
  different points in each candidate and looked at them.

Ingest compatibility is judged against the current code. `wato_ingest` accepts
`Image` in `rgb8/bgr8/rgba8/bgra8` and `CompressedImage` in JPEG/PNG. It
rejects Bayer and h264. `ingest.wato.yaml` maps the 8 `camera_pano_*` cameras
(`ingest.wato_6cam.yaml`: the same without `ee`/`ww`)
and the per-corner `lidar_*/velodyne_points`. It takes poses from eidos
`/world_modeling/liso/odometry`.

## Recommended set

All of these have `lidar_cc` plus `lidar_ne` and `lidar_nw` at about 19.8 Hz,
`camera_info` for every camera, `/tf_static`, and a moving car.

| # | Recording | Cameras | Length | Size | Driving | Scene | Ready to ingest? |
|---|---|---|---|---|---|---|---|
| 1 | `May_30_ring_road_rosbags/may_30_ring_road_test_5` | **8 pano** (all of `ingest.wato.yaml`), JPEG, 11.9 Hz | 14.8 min | 90 GB, 5 files, all finalized | ≥2.9 km, ~8 m/s | Sunny summer daytime. Campus roads and a public road with signalled intersections, traffic and pedestrians | Needs eidos pass (no `liso/odometry`) |
| 2 | `recording_20260224_213947` | **12** (8 pano + 4 lower), JPEG, 11.8 Hz | 9.7 min (+2 GB unfinalized tail) | 45 GB | ≥1.7 km, ~8 m/s | Overcast winter afternoon on campus roads with pedestrians, cyclists and parked/moving cars | Needs eidos pass |
| 3 | `recording_20260215_215733` | **12**, raw `bgr8` `image_rect`, 12.0 Hz | 21.2 min (+1 unfinalized file) | **773 GB**, 36 files | ≥7.5 km, up to 17.8 m/s | Sunset on arterial roads with strong low-sun glare. Part of it is on a snow-covered field/track | Needs eidos pass. Huge, so ingest a few files, not all 36 |
| 4 | `recording_20260217_224728` `_0` + `eidos/recording_20260217_224728_1_with_eidos.mcap` | **12**, JPEG, 11.5 Hz | 5.1 min (`_0`) + 2.6 min (`_1` eidos) | 21 GB + 8.6 GB | ~8 m/s, ≥550 m per file | Dusk on Ring Road (snow) | **The eidos `_1` file is ready now** (has `liso/odometry`). `_0` needs an eidos pass. The original `_1` is unfinalized |
| 5 | `ring_road_corrected` (= `may_30_ring_road_test_3` with eidos) | **6 pano** (no `ee`/`ww`), JPEG, 12.0 Hz | 12.5 min | 70 GB | ≥2.2 km, up to 7.5 m/s | Sunny summer daytime, campus Ring Road | **Ready** with `ingest.wato_6cam.yaml` (has `liso/odometry`, 8.5 Hz) |
| 6 | `May_30_ring_road_rosbags/may_30_ring_road_test_2` | **6 pano** (no `ee`/`ww`), JPEG, 11.8 Hz | 3.2 min | 18 GB | ≥560 m | Sunny summer daytime, campus | Needs eidos pass; `ingest.wato_6cam.yaml` |

Notes:
- **Best for a full 8-camera ring: #1.** It's the only bag that has exactly
  the 8 pano cameras the WATO profile uses, all finalized, in daylight, while
  driving among other road users.
- **Best winter / variety pick: #2.** It has the most labelable objects
  (pedestrians and cyclists) of any frame I looked at.
- **#3 is the longest drive by far,** but it's raw images (773 GB) into
  low sun, and part of it isn't on a road. Pick files from the first half.
- **#4's eidos file is the only 12-camera bag with eidos poses already on
  Drive.**
- **#5 / #6:** `may_30_ring_road_test_3` is the same drive as #5 before
  eidos. Only 6 cameras: `camera_pano_ee` and `camera_pano_ww` weren't
  recorded, so ingest them with `ingest.wato_6cam.yaml`.
- **Poses:** everything except #4's eidos file and #5 has to go through
  `wato_monorepo/tools/eidos_offline_to_bag.sh` first. Their own
  `slam/odometry` is keyframe-rate (ingest aborts on it), and
  `/novatel/oem7/odom` isn't validated for labeling (see
  `docs/ingest_measurements.md`).

## Meets the sensor criteria but not worth using

| Recording | Sensors | Problem |
|---|---|---|
| `recording_20260309_211754` (the one already ingested; local copies deleted 2026-10-27) | 12 cams @ 8 Hz, 3 LiDARs | **Parked inside the garage.** GNSS has no fix (status −1, lat/lon 0). Ingest tagged it `STATIONARY` (mean speed 0.09 m/s) and `CAMERA_DROPS`. 47 s only |
| `recording_20260224_011359` | 12 cams, 3 LiDARs | **Night, parked in a lot.** 0 m/s at both samples, 6 m of drift over 4 min |
| `recording_20260301_223023` | 12 cams, 3 LiDARs, 25.6 min | **Barely moves.** Never more than 70 m from its start over 25 min (parking-lot manoeuvres). No `/novatel/oem7/odom` |
| `recording_20260117_161533` | 12 cams @ 60 Hz, 3 LiDARs, 8 min, 2.6 km of driving | **Bayer images** (`rgb/image_raw`, `bayer_bggr8`). Ingest rejects Bayer, so it would need a debayer pass first. 475 GB |
| `recording_20260120_101600` | 12 cams, 3 LiDARs | **The recorder couldn't keep up.** Cameras at ~2 Hz, LiDARs at ~5 Hz |
| `recording_20261004_185825` | 11 cams (no `lower_se`) @ 3.5 Hz, `lidar_cc` only | Cameras at 3.5 Hz. `lidar_cc` at 13.5 Hz (dropped scans; `_1` fails pose density, see `docs/ingest_measurements.md`). No `lidar_ne`/`lidar_nw` |
| `recording_20261004_184219` | same rig as above | Same rate problems, and files `_1` and `_3` are **missing from Drive** (two ~63 s holes) |
| `Dec 22, 2025/recording_20251222_160131` | 8 pano cams (raw, 62 Hz), `lidar_top/left/right` | Old rig naming (`lidar_top`, no `lidar_cc`). Only 32 s finalized; the 20 GB second file is unfinalized |

## Doesn't meet the criteria

| Recordings | What they have |
|---|---|
| `recording_20260406_094629`, `_172236`, `_185305`, `traffic_lights_rosbags/traffic_lights`, `May_30_ring_road_rosbags/all_topics` | `lidar_cc` but only the 2–3 front cameras (`pano_ne/nn/nw`). `_172236` and `_185305` are long (22–24 min) with eidos poses, so they're good LiDAR-only data |
| `recording_20260309_215941`, `_225904`, `_230502`, `_231130`, `recording_20260325_211221`, `_212628`, `_213913` | LiDAR only (3 Velodynes), no cameras |
| `Nov 24, 2025/*` | Old rig: 3 FLIR cameras + one `/velodyne_points` at 5 Hz |
| `Dec 20, 2025/*` | LiDAR only (`lidar_top/left/right`) |
| `joystick_steering_rosbags/*` | CAN odometry only, no sensors |
| `NuScenes-v1.0-mini-scene-*` | nuScenes (6 cams + `LIDAR_TOP`); not WATO data |

## Housekeeping

- **Duplicates, about 156 GB.** These "Copy of" files are byte-for-byte the
  same size as their originals:
  - `recording_20260325_211221`: 3 copies of `_0`
  - `recording_20260217_224728`: 2 copies of `_1`
  - `ring_road_corrected`: 1 copy
- **Unfinalized files (no summary/footer).** Most multi-file recordings end
  in one. These recordings consist only of unfinalized files:
  - `recording_20251220_181343`
  - `recording_20251124_175305`
  - `recording_20260325_212628`
  - `recording_20260325_213913`

  `mcap recover` (the CLI is already in the ingest image) rewrites such a
  file with a proper summary. Run it before ingesting one.
