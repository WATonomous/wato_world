# lidar_preprocessing: measurements behind the defaults

The measurements that set lidar_preprocessing's defaults and limitations. The
behaviour itself is in the component README. "On-static %" is the share of a
cloud within one voxel of a static-map surface, scored with
`scripts/compare_seg_dynamic.py`. There are no labels, so these are proxies, not
accuracy.

## Returns on the vehicle itself (2026-11-16)

Raw sweeps in the sensor frame, 20 per chunk:

| Rig | Inside the 1 m square (median / max per sweep) | Within 2.5 m | Of |
|---|---:|---:|---:|
| nuScenes scene-0061 LIDAR_TOP | 8,478 / 8,970 | 8,745 | 34,720 |
| nuScenes scene-1100 LIDAR_TOP | 7,801 / 9,037 | 7,948 | 34,720 |
| WATO July 7 `lidar_cc`, chunk 0000 | 100 / 109 | 100 | 22,830 |
| WATO July 7 `lidar_cc`, chunk 0008 | 90 / 98 | 90 | 21,427 |

Plotted, nuScenes' are the roof and sensor mount under the sensor (z −0.25 to
−0.9 m), the same shape every sweep; WATO's are two short roof-rack strips at
x ≈ ±0.8 m, z ≈ −0.4 m. Patchwork++ ignores returns under 1 m, so none is
ground: on nuScenes they were 44% of the non-ground points, a dense blob that
took a six-scale DBSCAN bank from 0.61 s to 3.8 s per frame. Hence
`remove_close_m` (README, Step A): 1.0 m on nuScenes, the devkit's default.
About 260–450 nuScenes returns per sweep remain between 1 and 2.5 m (roof
edges ahead of the square); the devkit keeps those too.

## Ground grid (2026-11-17)

A parked truck on nuScenes 0061 (GT 19319) had Step C cells 2–5 m above the
road under it, and label_refinement, which puts a moving track's box bottom
on the grid, lifted its box 3 m in some frames. The cells held the truck's
own returns, labelled ground by Patchwork++ in sweeps 18–60 m away: a
concentric-zone bin there held only raised structure, its lowest-point seeds
were one ring's line along the truck's side (a line fits a level plane), and
past its first rings Patchwork++ runs no height check (upstream's comment:
far patches are too sparse to judge); vertical-plane removal (R-VPF) runs in
its first zone only. No sweep sees the road under a parked vehicle, so those
returns were the cell's only ground, and the nearest-cell fill copied them
to the empty cells around.

Ground returns more than 0.5 m above a grid built from first-zone ground
(12.125 m at the shipped `min_range` 1 / `max_range` 90), by distance to
their sweep's sensor:

| Range | nuScenes 0061 | nuScenes 1100 |
|---|---:|---:|
| < 12.1 m (first zone) | 0.000 | 0.000 |
| 12.1–17.7 m | 0.039 | 0.023 |
| 17.7–30 m | 0.352 | 0.345 |
| 30–90 m | 0.635 | 0.826 |

Grid against the GT box bottoms (a box stands on the road), offset-free share
more than 1 m off:

| Grid | 0061 all / vehicles | 1100 all / vehicles |
|---|---|---|
| All ground (before) | 0.137 / 0.072 | 0.323 / 0.380 |
| First-zone ground (now) | 0.025 / 0.004 | 0.070 / 0.021 |

The new grid is no worse at any range (0–20, 20–40, 40 m+), above or below.
Raised cells (a populated cell over 1 m above the median of its 9×9
neighbourhood): 0061 7,734 → 23, 1100 2,956 → 10; under the truck 73 → 0.

Patchwork++'s own settings do not fix it. Re-run on 0061 with the library's
defaults (`uprightness_thr` 0.707 against the shipped 0.101, which the
monorepo took from upstream's ROS launch file; `th_seeds` / `th_dist` 0.125)
the raised cells fall to 4,305 but the truck keeps 71; of the returns it
turns from ground to non-ground over observed road, 34 % lie within 0.15 m
of it (it rejects whole patches). In its first zone both values give the
same grid.

Downstream on nuScenes (Step C, then Step F, proposal_generation, tracking,
label_refinement; IWU byte-identical): refined car IoU 0.31 → 0.34 (0061)
and 0.31 → 0.36 (1100), moving vehicles 0.38 → 0.39 and 0.32 → 0.36; 0061
tracks per object 1.42 → 1.33, false tracks 16 → 14. 0061's parked-vehicle
medians worsen (centre 0.47 → 0.72 m, yaw 4.2 → 9.4°) through one car (GT
22183): its lifted-only track now starts at sweep 142, not 233, and the one
parked box over the longer track is 1.5 m longer.

Cost: the grid holds direct observations for a third of the cells (0061
104,273 → 35,363; WATO July 7, 11 chunks, 4,351,442 → 1,166,132), the rest
filled from the nearest, so terrain seen only from afar comes out flat. On
WATO, where raised cells fall from 86,178 to 1,503, that is measured: on the
old grid's smooth cells (within 0.1 m of their 9×9 median, 78–90 % of them,
so not truck sides), old − new by distance from the drive path:

| From the path | Chunk 0000 (parked, 10 m driven) | Chunk 0005 (driving) |
|---|---|---|
| 0–12 m | +0.00 m (100 % within 0.3 m) | +0.00 m (98 %) |
| 12–25 m | +0.12 m (56 %) | +0.02 m (79 %) |
| 25–40 m | +0.20 m (30 %) | +0.14 m (44 %) |
| 40–60 m | +0.71 m (20 %) | −0.10 m (28 %) |
| 60 m+ | +1.30 m (12 %) | −0.68 m (8 %) |

Broad smooth surfaces 0.3–1.5 m above or below the road are terrain (lots,
lawns, slopes), so beyond about 25 m from the path the first-zone grid
loses it; nuScenes' GT objects are mostly near the road. GroundGrid does
not resolve it ("GroundGrid against Patchwork++", below).

## GroundGrid against Patchwork++ (2026-11-19)

GroundGrid (Steinke et al., RA-L 2024) builds a terrain map from every
return rather than from a segmenter's ground: per scan, a cell's 3×3 / 5×5
patch is ground when its height variance is small and it has enough returns
(relative to the expected count at that range); its height then joins the
cell's confidence-weighted estimate, other cells are only lowered to their
patch minimum, and a spiral pass from the vehicle outward fills low-confidence
cells from their neighbours. Ported line for line from the reference
(`dcmlr/groundgrid` master 7d6a5d7, the ROS1 code the paper evaluated) in
numba, offline: the reference's 120 m × 0.33 m window over a world-fixed
array, keeping cells the window leaves. Run with the authors' KITTI parameters
and their HeLiPR Velodyne set, with the horizontal step from `sensor_model.py`
(hdl32e 0.33°, vlp32c 0.40°); the two sets differ by little anywhere below,
so KITTI's are shown. At scan time its map is within 0.03 m of the
first-zone grid within 8 m of the vehicle.

Candidates: Patchwork++ ground at all ranges (`pw_all`, before), first zone
(`pw_zone`, now), GroundGrid's map (as the reference holds it after the last
scan; and each cell as of the scan GroundGrid was most confident in it), and
Step C's binning of GroundGrid's per-point ground. nuScenes 0061, grid under
each GT box centre against its bottom, offset-free share more than 1 m off
(all classes; by the box's distance from the drive path):

| Grid | all | 0–20 m | 20–40 m | 40 m+ | spike cells |
|---|---:|---:|---:|---:|---:|
| `pw_all` | 0.171 | 0.098 | 0.255 | 0.452 | 7,734 |
| `pw_zone` | **0.016** | **0.008** | **0.027** | **0.038** | **23** |
| GroundGrid map, after last scan | 0.347 | 0.312 | 0.381 | 0.493 | 1,808 |
| GroundGrid map, most confident scan | 0.145 | 0.070 | 0.252 | 0.391 | 9,779 |
| GroundGrid map at each GT keyframe (by range from the sensor) | 0.228 | 0.033 | 0.241 | 0.380 | |
| Step C binning of GroundGrid ground | 0.163 | 0.089 | 0.242 | 0.464 | 4,081 |

(spike cell: populated, > 1 m over its 9×9 median.) GroundGrid's misses are
mostly too high (64–77 %) and mostly not vehicles: past ~20 m a 32-beam
scanner lays one ring line per 0.33 m cell on a pedestrian or a car, its
height variance is near zero, and it passes GroundGrid's flatness test — the
failure that let Patchwork++ beyond its first zone take truck sides for
ground. Per point, at GT keyframes, GroundGrid labels as many GT-object
returns (0.3 m above the box bottom) ground as Patchwork++ (0061: 4.7 % against
4.5 %; beyond 20 m 30 % against 28 %) and more of the road beside GT boxes
(beyond 20 m 0.95 against 0.83). Scene 1100 does not score terrain: its ego
poses have z = 0 on a slope while its GT bottoms fall from +0.44 m near the
path to −1.51 m beyond 40 m, so every grid misses its far boxes by 1.7–2.7 m.

On WATO, GroundGrid keeps far open terrain: on `pw_all`'s smooth cells it is
within 0.3 m for 82–98 % of them out to 60 m from the path (chunks 0000 /
0005; `pw_zone` 20–56 %), but has 1,081 / 3,945 spike cells against
`pw_zone`'s 1 / 141.

The grid is read under objects (box bottoms, Step F's floor), where no
GroundGrid variant is better than `pw_zone` at any range, so Step C keeps
first-zone Patchwork++. Far lots and lawns stay flattened to the nearest
near-road cell.

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

## Step F against ground truth (nuScenes scene-0061)

`scripts/eval_motion_proposals.py`, default config, 37 annotated instants
(every keyframe sweep matched at 0 ms). A non-ground point is moving iff it is
inside an annotated box whose object moves faster than 0.5 m/s at that instant
(`wato_common.eval.motion.instant_speeds`); 8,785 such points, mostly cars
(6,860) and pedestrians (1,788).

| Verdict | IoU_MOS | Recall | Precision |
|---|---|---|---|
| AW_DYNAMIC | 0.27 | 0.52 | 0.35 |
| AW_AMBIGUOUS | 0.01 | 0.01 | 0.01 |
| IWU_EVICTED | 0.31 | 0.48 | 0.47 |
| SEG_DYNAMIC | 0.29 | 0.32 | 0.73 |
| UNMAPPED | 0.21 | 0.38 | 0.33 |
| any bit | 0.25 | 0.78 | 0.27 |
| clustered | 0.35 | 0.77 | 0.39 |
| clusters with motion_score > 1 | 0.22 | 0.23 | 0.93 |
| gated (+ n_sources ≥ 2, frac_persistent < 0.5) | 0.22 | 0.22 | 0.94 |
| BOX_FILL | 0.22 | 0.22 | 0.94 |
| `dynamic_mask` (Step B) | 0.24 | 0.32 | 0.49 |

Gated recall: 0.25 on cars, 0.12 on pedestrians, 0 on the one moving truck;
0.23 / 0.25 / 0 / 0 by range band (0–15 / 15–30 / 30–50 / 50+ m).

- **The candidates are fine; the motion test drops most movers.** Clustering
  keeps 77% of moving points, motion_score > 1 keeps 23%. The other two gates
  cost almost nothing.
- **Why:** of the 4,754 moving points in clusters scored ≤ 1 (161 clusters,
  point-weighted medians), the track lasted 4 frames (p25 1) and the object
  really moved 0.41 m in that time (GT speed × duration), against a cluster
  whose longest side was 3.5 m (p75 8 m: clusters merge neighbours).
  "Further than its own size" can't pass on a 0.2 s track. 58% of those
  points' clusters did move ≥ 0.5 m net. 80% of them are on cars.
- What passes is precise (0.94): the gated rows are a clean seed.

One scene with 5 moving cars and 42 moving pedestrians; WATO not measured
against ground truth yet.
