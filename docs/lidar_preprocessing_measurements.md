# lidar_preprocessing: measurements behind the defaults

The measurements that set lidar_preprocessing's defaults and limitations. The
behaviour itself is in the component README. "On-static %" is the share of a
cloud within one voxel of a static-map surface, scored with
`scripts/compare_seg_dynamic.py`; mover-recall is proxied by distance from the
static map. There are no labels, so these are proxies, not accuracy.

## Deskew on the WATO rig (`ring_road_corrected`)

- The Velodyne clouds carry a per-point `time` field, but every value is 0 (all
  153 sweeps sampled across `lidar_cc` / `lidar_ne` / `lidar_nw`), so deskew
  synthesizes times from azimuth.
- Uncompensated, the car moves ~42 cm (median) during one 50 ms sweep, so a
  point 30 m away is misplaced by 52 cm median, 83 cm p95.
- Each sweep is recorded 1.5 ms after its stamp, which a start-of-sweep stamp
  can't be. Together with the driver's `timestamp_first_packet: false`, that
  is why the WATO profile sets `header_stamp_at: sweep_end`.
- **VLP-32C seam points.** The azimuth anchor is the cloud's first point. The
  VLP-32C's lasers sit at slightly different horizontal angles, so a few points
  of the first firing land just behind the anchor and are timed one rotation
  late: ~0.5 % of `lidar_cc` points, ~40 cm off at WATO speeds. Unwrapping by
  array position wasn't done because nuScenes clouds are ordered differently
  (their last points wrap past the start angle) and it is unverified there.

## union: why the vetoes and the motion filter exist

- **Ground-height veto.** AW never calls road voxels static, so MF-MOS road
  false positives pass every voxel veto. On nuScenes scene-0061 they were half
  the union dynamic cloud, median height −0.36 m.
- **Why a motion filter.** After the AW-static and ground vetoes, ~58 % of the
  union dynamic cloud on real Velodyne data still sat within 25 cm of a static
  surface: structure the AW map covers sparsely (far walls, foliage,
  below-grade returns). MF-MOS is a SemanticKITTI (HDL-64E) model applied out
  of domain and over-fires on textured static surfaces.

**Persistence threshold** (WATO ring-road bag):

| `persistence_max_sweeps` | dynamic pts | on-static | mover-recall |
|---|---|---|---|
| (no filter) | 360.7K | 58.3% | 100% |
| 5 | 86.0K | 6.6% | 53% |
| 12 | 117.6K | 9.3% | 71% |
| **20 (default)** | **128.9K** | **12.6%** | **75%** |
| 28 | 143.3K | 16.0% | 80% |

Persistence 5 + coherence gave 3.9 % on-static at 34 % recall (over-cuts).
Leakage climbs faster than recall past ~24 sweeps. The recall ceiling is
~75–80 %: an extended mover (a 4.5 m car at 5 m/s dwells ~9 sweeps in a 0.5 m
voxel) is indistinguishable from structure by per-voxel occupancy alone.

**Rejected alternatives:**

- The coherence gate cuts ~20 % of real movers for < 1 % precision on sparse
  (32-/64-beam) LiDAR (distant movers don't cluster), so it is off by default.
- A velocity gate, and a translating-cluster *rescue* of persistent points,
  both tested worse: per-sweep visibility drifts a cluster's centroid as the ego
  passes structure, faking velocity. The rescue re-admitted ~77 % structure via
  wall-sliding.
- An `MF-MOS ∩ AW-dynamic` intersection raised leakage (17 % → 72 % on
  nuScenes): AW-dynamic voxels hug static surfaces.

## MF-MOS on the WATO rig (unaudited)

Both profiles default to `segmentation: aw` until MF-MOS mask quality on
`lidar_cc` is audited (`scripts/debug_mfmos_contribution.py`). MF-MOS also
skipped a large fraction of sweeps in two measured chunks (190/656 and
101/101), so its real contribution may be far smaller than "enabled" suggests.

## Step E — IWU (nuScenes scene-0061)

One 19 s chunk, `seg=union`, 191k map points, 77 sampled sweeps, 9 s runtime.

- The paper-literal rule (single-pixel seen-through test, nearest-only
  reinforcement) evicted 54 % of an AW-static map: ring gaps and origin error
  read as "seen through".
- The windowed hybrid rule evicts 7.0 %, and eviction is 2.0× more likely within
  0.5 m of the union dynamic cloud than elsewhere (the only precision proxy
  available without labels).

## Step F — motion proposals (same scene, 382 sweeps, 47 s)

- 5.4 % of points carry a bit; 21.6k clusters (57/frame), 404 with
  `motion_score > 1`, 12 box-filled tracks.
- On-static leakage above chance, against the IWU-refined map
  (`compare_seg_dynamic --proposals`): IWU_EVICTED +20.6, BOX_FILL +21.4, moving
  clusters +25.6, SEG_DYNAMIC +31.2, MF_MOS +55.7, AW_DYNAMIC +64.1,
  AW_AMBIGUOUS +72.7 pts. The leakage comes from the AW bits, so downstream
  should weight clusters by `n_sources`, `frac_persistent` and `motion_score`
  rather than trust any bit alone.
- **BOX_FILL's `frac_persistent < 0.5` condition.** Size-normalised displacement
  alone is fooled by a thin wall fragment whose visible window slides further
  than it is long: a 1.6×0.1 m fragment drifted 3 m with `frac_persistent` 1.0,
  while the clear movers sat at ≤ 0.11. The cost: a pedestrian-sized track at
  0.52 went unfilled (its row keeps its `motion_score`).
