# lidar_preprocessing: measurements behind the defaults

The measurements that set lidar_preprocessing's defaults and limitations. The
behaviour itself is in the component README. "On-static %" is the share of a
cloud within one voxel of a static-map surface, scored with
`scripts/compare_seg_dynamic.py`. There are no labels, so these are proxies, not
accuracy.

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

## Step B: keeping static structure out of the dynamic cloud

Measured on WATO `may_30_ring_road_test_5_2`. With plain carving, chunk 0000's
dynamic cloud was 3.2M points against 6.5M static, 27–31 % of non-ground points
on every scanner. 80 % of those points sat in voxels hit across ≥ 2 s, 97.6 %
lay within 0.5 m of a static-map point, and the median dynamic voxel had 96
traversals to 1 hit. Canopies, façades and edges dominated.

Scored with `compare_seg_dynamic.py --save-reference` / `--reference`, against
sets frozen from the run before these changes:

- **Persistent static:** dynamic points in voxels with dynamic hits across
  ≥ 5 s. Share still dynamic; lower is better.
- **Movers:** points of Step F tracks with `motion_score` > 2, `frac_persistent`
  < 0.3, `track_life` ≥ 5, ≥ 20 points (11 tracks on chunk 0000: five cars, a
  pedestrian...). Share kept; higher is better. These tracks were seeded by the
  old classifier, so the set leans towards what it already flagged.

| Chunk 0000 | dynamic pts | % of before | persistent static | movers |
|---|---|---|---|---|
| Before (clamped, no shield) | 3.20M | 100 | 100 % | 73.5 % |
| Clamp removed | 2.21M | 69.0 | 36.8 % | 79.5 % |
| Shield, same sweep | 1.74M | 54.5 | 47.4 % | 72.9 % |
| Shield, same frame | 1.63M | 50.9 | 44.0 % | 72.6 % |
| Shield, ±250 ms | 1.07M | 33.6 | 26.0 % | 71.2 % |
| **Same-frame shield + no clamp** | **886k** | **27.7** | **8.5 %** | **78.1 %** |
| + cluster ≥ 5 | 440k | 13.8 | 4.4 % | 77.3 % |
| **+ cluster ≥ 5 + persistence 0.5 s (default)** | **316k** | **9.9** | **2.2 %** | **76.9 %** |

Across chunks: 0002 is the busiest (59 mover tracks), 0004 the worst leak (79 %
of its dynamic cloud persistent).

| Chunk | Movers before | Shield + no clamp (size / persistent / movers) | + filter (size / persistent / movers) |
|---|---|---|---|
| 0000 | 73.5 % | 27.7 % / 8.5 % / 78.1 % | 9.9 % / 2.2 % / 76.9 % |
| 0002 | 71.6 % | 40.9 % / 21.8 % / 74.7 % | 17.1 % / 5.2 % / 73.5 % |
| 0004 | 67.9 % | 29.6 % / 11.7 % / 74.8 % | 8.9 % / 2.0 % / 71.5 % |

Both kernel changes are needed. Clamp removal alone leaves 60.0 % persistent on
0002 and 37.9 % on 0004; the shield alone leaves 47.3 % and 40.0 %, and costs
~0.5–0.9 points of movers. A voxel's clamped log-odds reflects only its last ~23
observations: processing chunk 0000's sweeps in timestamp rather than `sweep_id`
order (52 swaps) changed 386 sweeps' masks.

The shield adds ~15 % to ray casting: 184.9 s → 212.1 s for 400 sweeps of chunk
0000, run alone, two alternating runs each.

End to end on chunk 0000: static points 6.5M → 10.1M. Step F's clusters fall from
99,072 to 48,775 and its moving clusters from 2,175 to 1,378, with nothing lost
outright: of the frozen mover points, 98.8 % still carry a source bit, 98.1 % are
clustered and 84.6 % sit in a moving cluster. Six of the seven car-sized tracks
still pass the mover criteria. Small movers pay: the pedestrian's track fragments
(95.1 % clustered, 45.4 % in moving clusters, median `motion_score` 0.34).

**Filter thresholds** (cluster ≥ 5, on the three chunks):

- `persistence_max_s`: 0.5 s is 30 sweeps at this rig's 60 sweeps/s. 20 sweeps
  falls off a cliff (movers −8.5 / −4.6 / −6.2 points); 40 sweeps gains
  +0.3 / +0.3 / +1.1 points of movers for +0.5 / +2.3 / +0.7 of persistent.
- `min_cluster_pts`: 8 instead of 5 trims the cloud a further ~1–2 points for
  −0.2 / −0.2 / −0.6 points of movers. 5 matches Step F's `min_cluster_pts`.

**Rejected:**

- **Re-calibrating, or a grid per scanner.** `lidar_cc` alone keeps 23.7 % of its
  non-ground points dynamic, against 26.7 % with all three: cross-scanner carving
  is ~11 % of the leak.
- **Wider shield.** ±250 ms costs movers (71.2 % against 72.6 % for one frame,
  with the clamp on), likely by shielding a mover's own recent trail.
- **DUFOMap's "void once" rule on top of the shield.** Ever-carved voxels flood:
  235 % of the old cloud at k = 1, 199 % at k = 3. A single clean pass isn't
  motion evidence here. The paper's stricter void test (all 26 neighbours
  observed in the scan) was not tried.
- **Ground-height veto.** At 0.25 m it costs 3.1–3.7 points of movers for 0.8–1.5
  points of persistent, and would need Step C before the masks are written.
- **Hit-span gate** (drop voxels with dynamic hits across ≥ T). Traffic reuses
  voxels, so movers fall with it: at T = 3 s, 65.4 % / 53.8 % / 58.9 % kept.
- A **velocity gate**, a **translating-cluster rescue** of persistent points and
  a **coherence gate** all tested worse on the old fusion's candidates: per-sweep
  visibility drifts a cluster's centroid as the ego passes structure, faking
  velocity, and distant movers don't cluster on sparse LiDAR.

Not measured: anything beyond this one WATO bag. No nuScenes data was available,
so the filter defaults are unvalidated on `hdl32e`.

## Step E — IWU (nuScenes scene-0061)

One 19 s chunk, 191k map points, 77 sampled sweeps, 9 s runtime. Measured
before the Step B changes above; re-measure.

- The paper-literal rule (single-pixel seen-through test, nearest-only
  reinforcement) evicted 54 % of an AW-static map: ring gaps and origin error
  read as "seen through".
- The windowed hybrid rule evicts 7.0 %, and eviction is 2.0× more likely within
  0.5 m of the dynamic cloud than elsewhere (the only precision proxy available
  without labels).

## Step F — motion proposals (same scene, 382 sweeps, 47 s)

Measured before the Step B changes above; re-measure.

- 5.4 % of points carry a bit; 21.6k clusters (57/frame), 404 with
  `motion_score > 1`, 12 box-filled tracks.
- On-static leakage above chance, against the IWU-refined map
  (`compare_seg_dynamic --proposals`): IWU_EVICTED +20.6, BOX_FILL +21.4, moving
  clusters +25.6, SEG_DYNAMIC +31.2, AW_DYNAMIC +64.1,
  AW_AMBIGUOUS +72.7 pts. The leakage comes from the AW bits, so downstream
  should weight clusters by `n_sources`, `frac_persistent` and `motion_score`
  rather than trust any bit alone.
- **BOX_FILL's `frac_persistent < 0.5` condition.** Size-normalised displacement
  alone is fooled by a thin wall fragment whose visible window slides further
  than it is long: a 1.6×0.1 m fragment drifted 3 m with `frac_persistent` 1.0,
  while the clear movers sat at ≤ 0.11. The cost: a pedestrian-sized track at
  0.52 went unfilled (its row keeps its `motion_score`).
