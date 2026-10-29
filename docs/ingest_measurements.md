# Ingest: measurements behind the pose rules and pose profiles

The measurements that set ingest's `pose_requirements` and the choice of
`topics.pose` in each profile. The rules themselves are in the ingest README
("Poses"). No ground truth exists for these drives. Each test states what it
assumes, and together they bound the error without trusting any single source.

The two `ring_road_corrected` recordings contain identical LiDAR (every
`liso/odometry` stamp in `_0-001` equals a `lidar_cc` header in `_part1`) but
different eidos output (2,274 vs 2,041 `slam/odometry` poses): they are two
separate eidos runs over the same data. Several of the bags below are no longer
on disk.

## Which pose sources pass the checks

Pose topic only, read-only.

| Recording | Topic | Distinct-pose spacing | Dense fraction | Checks |
|---|---|---|---|---|
| nuScenes mini | `/odom` | 20 ms median, 45 ms max | 1.0 | pass (`ingest.yaml`) |
| `ring_road_corrected_part1` | `/world_modeling/slam/odometry` | 100 ms median, 175 ms max | 1.0 (every chunk) | pass; its largest step, 4.45 m in 100 ms, trips `max_speed_mps` — cause not verifiable without the bag |
| `ring_road_corrected_0-001` | `/world_modeling/slam/odometry` | 100 ms median | 1.0 | pass; largest step 2.46 m in 100 ms, a stamping artifact (section 1 below) |
| `ring_road_corrected_0-001` | `/world_modeling/liso/odometry` | 50 ms median, 202 ms max | 1.0 | pass; no jumps |
| `may_30_ring_road_test_3_3` | `/world_modeling/slam/odometry` | 908 ms median, 5.8 s max — one pose per 5 m | 0.0 | **abort** |
| `recording_20261004_185825_1_with_eidos` (58 s; eidos run offline with `tools/eidos_offline_to_bag.sh`) | `/world_modeling/liso/odometry` | 50 ms median, 405 ms max; 39 gaps > 250 ms, most 252 ms | 0.92 / 0.73 / 0.72 (chunks 0000–0002) | **abort** (chunk 0001). The recording dropped `lidar_cc` scans (735 of ~1,160 at 19.8 Hz) and LISO can't pose a scan that isn't there; 4 of the 7 scans LISO missed are warmup, 3 are skipped registrations. Whole bag: 0.82 at 250 ms, 0.95 at 260 ms |
| `recording_20261004_185825_1_with_eidos` | `/world_modeling/slam/odometry` | ~710 ms (82 keyframes in 58 s) | — | **abort** (keyframe rate) |
| `recording_20260309_211754_0_with_eidos` (47 s, parked throughout — LISO and GNSS both move < 1 cm) | `/world_modeling/liso/odometry` | 50 ms median, 1,215 ms max; 2 gaps > 250 ms, both where `lidar_cc` itself has them (~27 s) | 0.950 / 0.987 (chunks 0000–0001; whole bag 0.969) | pass; 901 of 908 scans posed, the 7 unposed are warmup. Its cameras start 14.1 s in, so chunk 0000 drops 6,782 of 14,704 (sweep, camera) rows |
| `recording_20260309_211754_0_with_eidos` | `/world_modeling/slam/odometry` | 1 keyframe (parked) | — | **abort** (keyframe rate) |
| `ring_road_July_1-2` | `/novatel/oem7/odom` | 20 ms median after dropping held repeats (29,755 of 67,797 messages) | 1.0 | pass |
| `may_30_ring_road_test_3_3` | `/novatel/oem7/odom` | 2 ms median, but in bursts: fixes held up to 11.7 s, jumps up to 34 m (INS not converged) | 0.10 | **abort** |

## `max_bracket_ms`: how far apart two samples may be

Measured on `ring_road_corrected` by thinning `liso/odometry` and
interpolating the dropped samples back. For brackets of 200–250 ms, a point
30 m away is displaced by 3.7 cm median, 10.8 cm p95, 40 cm max; most of that
is rotation, not translation. At 350–500 ms it becomes 23 cm p95 and 1.4 m max,
about one 0.25 m voxel at p95. The reference is LISO itself, so its own jitter
is included.

## What the `may_30` abort prevents

Its 586 `slam/odometry` messages carry 40 distinct poses, 4.95–5.74 m apart. 39
of them are stamped exactly at a `lidar_cc` scan header, so the stamps are
correct; the stream is just sparse. The eidos code on `main` was unchanged
since the day before the recording, apart from `package.xml`, and the bag
behaves as that code does. Forcing the interpolation anyway, every `lidar_cc`
sweep between keyframes was located independently by point-to-plane ICP
against the keyframe scans placed at their eidos poses.

Displacement of a point 30 m away (821 sweeps; one clear ICP failure excluded):

| | Median | p95 | Max |
|---|---|---|---|
| All sweeps between keyframes | 0.46 m | 4.4 m | 5.8 m |
| Brackets longer than 1.5 s (the car slowing and stopping) | 3.5 m | 5.6 m | 5.8 m |
| Brackets of 1.5 s or less | 0.31 m | 1.1 m | 1.7 m |

- **Rotation dominates.** At the stop the car turned about 3° right, but
  interpolation swings it up to 7° left toward the next keyframe, which comes
  after a 37° left turn: 8.7° of heading error. Translation error peaks at
  1.2 m.
- **The last 309 sweeps get no pose at all.** They cover 15.6 s after the last
  keyframe, with the car stopped, and ingest never extrapolates.
- **How far to trust the method.** Registering each keyframe scan against its
  neighbours reproduces its own pose to 12 cm median, 34 cm max. The stop
  position is the same whether located against the earlier keyframe's scan or
  the later one's (within 13 cm).
- **The independent check only partly agrees.** eidos_transform's odom-frame
  EKF, which uses no LiDAR, gives the same right turn at the stop (−4.5° vs
  ICP's −3.1°) but places the stop 0.8 m further along; that disagreement is
  unresolved.

## How good are the poses (`ring_road_corrected`)

Passing the checks is necessary, not sufficient: they look at the pose stream
alone and can't see a mis-stamped stream or a biased attitude. Both occur on
the WATO recordings.

**1. What `slam/odometry` actually contains (`_0-001`; no assumptions).**
Every one of its 2,041 poses is bit-identical (0.0 mm, 0.000°) to a
`liso/odometry` pose, stamped 244–450 ms (median 310 ms, 90% within
266–357 ms) *after* that LISO pose's scan. Each stamp equals the message's own
recording time, i.e. it is the publish time, and 92% are 100 ms apart. The lag
is LISO's processing latency (239–337 ms, median 286 ms, from its own stamp to
its recording time) plus the wait for the next SLAM tick. Both vary per
message, so no constant shift removes it (section 2, last row). The LISO stamp
is the right one: it equals a `lidar_cc` scan header exactly, and,
independently, matching speed against the INS needs −20 ms for LISO but
−325 ms for `slam/odometry`. Between consecutive ticks the content advances by
0/1/2/3 LISO poses (12/1,166/859/3 times), which is why its implied speed jumps
4.7 → 18.6 → 4.9 → 14.7 m/s while the INS reads a steady 9.5 m/s. So in this run
`slam/odometry` is `liso/odometry` resampled at 10 Hz with late, varying
timestamps, and **no correction beyond LISO**. Its largest step (2.46 m in
100 ms) is three LISO poses spanning 252 ms at 9.7 m/s.

**2. Local motion vs the NovAtel INS (independent of the LiDAR).** Two
frame-free quantities per window — distance travelled and rotation angle — so
neither depends on how either sensor is mounted. The only fitted value is one
constant time shift per source. Assumes the INS is locally accurate (it is
IMU-driven).

| Stream (`_0-001`) | Best shift | 0.2 s: distance Δ median / p95 | 1 s: distance Δ | 1 s: rotation Δ | 5 s: distance Δ |
|---|---|---|---|---|---|
| `liso/odometry` as stamped | −20 ms | 1.9 / 8.5 cm | 4.8 / 17.1 cm | 0.06° / 0.31° | 11 / 34 cm |
| `slam/odometry` as stamped | −325 ms | 11.4 / 53.3 cm | 23.4 / 64.9 cm | 0.26° / 2.72° | 47 / 202 cm |
| `slam/odometry` shifted −325 ms | | 12.1 / 47.6 cm | 14.4 / 48.4 cm | 0.07° / 0.51° | 18 / 59 cm |

**3. Drift at a revisit (independent of GNSS).** The drive passes the same spot
twice, 115 m of driving apart (15 s and 40 s after LISO starts). Placed with
`liso/odometry`, the second pass needs 7.6–8.3 cm and ≤ 0.12° of ICP correction
to sit on the first. ICP's own floor, on a sweep 1 s later in the same pass, is
4.8 cm and 0.07°. This is the only revisit, so it bounds drift over 115 m only.

**4. Whole drive vs GNSS.** The NovAtel fix is autonomous only (NavSatStatus 0,
no RTK/SBAS), self-reported σ ≈ 1 m horizontal, 2 m vertical. After one rigid
2D alignment over the full 1.4 km, `liso/odometry` differs from it by 0.38 m
median, 1.0 m p95, 1.38 m max horizontally — within the GNSS noise. So LISO
drifts less than roughly 1–2 m over 1.4 km; GNSS this coarse can't resolve
smaller drift.

**5. LiDAR sweep overlap (not independent; it is what lidar_preprocessing
consumes).** The same `lidar_cc` sweeps (chunks 0003, 0006) projected with each
pose; overlap between sweeps 0.5 s and 2 s apart (nearest-neighbour distance,
later sweep → earlier). Absolute values are inflated by single-sweep sparsity;
compare rows. LISO registers exactly these scans, so this favours it by
construction, and a *constant* pose-time offset is nearly invisible here (the
metric mainly measures jitter). The `slam/odometry` row is from `_part1`'s run.

| Pose source | 0.5 s apart: within 10 cm | 2 s apart: median NN | 2 s apart: within 10 cm |
|---|---|---|---|
| `liso/odometry` | 15–18 % | 0.50–0.52 m | 7.6–8.1 % |
| `/novatel/oem7/odom`, constant attitude correction | 14–17 % | 0.50–0.53 m | 5.6–5.7 % |
| `slam/odometry` (`_part1`) | 8–10 % | 0.55–0.56 m | 3.7–4.4 % |
| `/novatel/oem7/odom` as recorded | 3–5 % | 0.81–0.87 m | 0.3–0.6 % |

**6. NovAtel attitude.** As recorded, NovAtel poses smear LiDAR badly over 2 s.
Timing isn't the cause: the best time shift gains nothing. One constant
rotation removes most of it (yaw +1.35°, pitch −1.93°, roll +0.34°), estimated
against LISO, so not independent evidence. The INS reports its own attitude σ
on this drive as 1.2° roll/pitch and 8.1° yaw, so the rotation could be INS
attitude error rather than a mounting offset; one recording can't tell them
apart, and a fixed correction is only valid if it's the mount. NovAtel's own
interpolation is fine: holding out every other sample and interpolating it back
errs by 0.3 cm median (7 cm p95).

**Not measured: eidos `main`'s `slam/odometry` against `liso/odometry`.** No
recording has both. On `main` those poses are keyframe poses after iSAM2 with
GPS and loop-closure factors. With ~1 m autonomous GNSS, GPS factors can move
keyframes by up to about a metre; whether that helps or hurts at sweep level is
unknown.

## eidos: `liso/odometry` vs `slam/odometry`

Both come from eidos (`wato_monorepo` `world_modeling/eidos`) and are map-frame
poses with child `base_footprint`.

| | `/world_modeling/liso/odometry` | `/world_modeling/slam/odometry` |
|---|---|---|
| What it is | LISO front end: each LiDAR scan GICP-registered against a submap of recent keyframe clouds, placed at their graph-optimized poses (IMU gives only the rotation initial guess) | Back end: iSAM2-optimized pose of the **newest keyframe** (LISO between-factors + GPS + IMU + loop closures) |
| When | every matched scan | every 10 Hz SLAM tick |
| Stamp | the scan's header time (all 2,894 in `_0-001` equal a scan header) | `main`: the keyframe's scan time. `_0-001`: ~310 ms after the scan |
| Distinct poses | one per matched scan | `main`: one per keyframe (≥ 5 m apart). `_0-001`: one per tick, each a LISO pose |
| Global corrections | follows them going forward (the submap is rebuilt from corrected keyframes); past poses are never revised | `main`: includes them for the newest keyframe (unmeasured here). `_0-001`: none — identical to LISO |
| Source | `liso_factor.cpp` `lidarCallback` | `eidos_node.cpp` `publishOdometry` |

- **Why LISO is dense.** It publishes from its LiDAR callback, once per scan
  GICP matches. Scans are skipped when GICP falls behind: 2,894 of ~4,000
  `lidar_cc` scans in `_0-001` (~13 Hz of 19.8 Hz).
- **Why `slam/odometry` is not.** LISO hands the graph a new state only after
  ≥ 5 m of travel (`min_scan_distance`, `liso_factor.cpp:476-486`); GNSS and IMU
  attach to existing states rather than creating them. `publishOdometry` then
  republishes the newest state's pose on every 10 Hz tick, so `may_30`'s 586
  messages carry 40 distinct poses.
- **Re-stamping `_0-001`'s `slam/odometry` doesn't help.** A message doesn't
  carry its scan's time and the lag varies by 200 ms; recovering it means
  finding the bit-identical `liso/odometry` pose, which gives back a 10 Hz
  subset of `liso/odometry` (2,029 of 2,894 poses).
- Neither is the final smoothed trajectory; that exists only in the saved
  `.map` file, and only at keyframes. Combining them — anchoring
  `liso/odometry` to the keyframe poses by interpolating the slowly varying
  keyframe correction — is not implemented.

## NovAtel `/novatel/oem7/odom`

- UTM coordinates (x ≈ 5×10⁵, y ≈ 4.8×10⁶ m), child `base_link`, 100 Hz
  messages with a 50 Hz INS position: every other message repeats the previous
  fix, which ingest's held-position filter drops.
- Not validated for labeling: as recorded its attitude smears LiDAR (section 6),
  and whether that's a mount offset or INS attitude error is undetermined.
- lidar_preprocessing's voxel keys are chunk-relative, so UTM magnitudes are
  fine there; anything that casts world coordinates to float32 loses ~0.5 m (the
  HTML viz does).
- The `ring_road_July` bags pass the pose checks on NovAtel but can't be
  ingested: they record no camera images (only `/camera_pano_nw/camera_info`)
  and no `lidar_nw` messages.

## Camera time vs sweep time

The car moves between a sweep and the nearest image (up to 44 ms apart on the
WATO rig). Projecting LiDAR into an image with the sweep's pose instead of the
pose at `camera_timestamp_ns` misplaces a point 30 m away by 17 cm median, up
to 87 cm.

## Bag observations

**`recording_20260309_211754_0_with_eidos`** (47.2 s, parked in the garage):

- Every camera topic (12 `image_rect` + 12 `camera_info`) starts 14.1–14.3 s in;
  the LiDARs, NovAtel and `/tf_static` start at 0. The raw recording has the
  same start, so the cameras weren't publishing when recording began.
  `trim-bag` keeps 14.261–46.525 s (set by `camera_lower_se/image_rect` and the
  0.67 Hz eidos visualization topics).
- Every topic faster than 1 Hz — all cameras, all three LiDARs, NovAtel,
  `oscc` — has a gap starting at 27.06–27.21 s, up to 1.37 s long; only the
  topics eidos computed offline, and `/tf`, have none. A gap shared by
  independent drivers points at the recorder (12 uncompressed 1280×1024 bgr8
  cameras at 12 Hz is ~570 MB/s).
- `camera_pano_ww`: header − log is −98 ms (`camera_info`) / −119 ms
  (`image_rect`); the other 11 cameras sit at −13 / −35 ms, so its stamps are
  ~85 ms (about one frame) earlier than its peers'. The bag alone can't explain
  it.
- Up to 9 images per camera have no `camera_info` with the same stamp (and up
  to 10 the other way). Ingest reads one `CameraInfo` per camera, so it's
  unaffected.

**Unindexed bags.** `ring_road_corrected_0-001.mcap` was never finalized and
has no index; it still came back in record-time order. Every ingest read of an
unindexed bag scans from the start to the chunk's end, because the storage
plugin can't skip unrequested topics; a full scan of that 19.5 GiB file takes
about 35 s.
