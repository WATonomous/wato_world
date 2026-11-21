# Camera-only boxes: verifying lifted objects in LiDAR

Researched and measured 2026-11-16 on the rotfix v2 run of
`sensors_2026_07_07_23_01_24_2_with_eidos_mcap` (calibration
`calibration_rotfix_v2.json`) and on the two nuScenes GT scenes, before any
pipeline change. Figures marked *measured* come from this
repository's data with the prototype under "Reproduce"; everything else is
the cited source's own statement. The design was built on 2026-11-17 and
is on by default after gate A; its nuScenes and WATO results are under
"Design".

## Summary

Most false boxes on WATO are a camera mask attached to the wrong LiDAR
surface: a car behind a hedge boxed on the hedge, a far parking-lot car
boxed on the tree line or building behind it, a person on a poster boxed on
the wall, a passenger boxed inside the car, a pedestrian boxed on the façade
behind them. The pipeline turns "a camera mask covers these pixels" into
"the first visible LiDAR returns along these rays are that object", and
nothing afterwards asks whether those returns form a distinct object inside
the mask. Published box-producing lifters make that check: the object's
LiDAR segment is grown over the whole scan and must lie inside the camera
frustum (FGR), and must match the lifted points (SAL). Published
auto-labellers then confirm a track only after enough confident frames
(MS3D++ retroactive labelling). A prototype of both, on existing artifacts:
on WATO the check separates the hand-checked false tracks from the real
ones, and track confirmation drops two thirds of camera-only tracks while
keeping every track LiDAR motion confirms; on nuScenes ground truth it
raises box precision from 0.72 to 0.79 on the scene with objects that
aren't there, for 0.6 points of recall. As a per-frame filter it would cost
a third of true boxes, so it belongs at the track level. False pedestrians
are only partly addressed.

## What the rotfix v2 run shows (measured)

- **73% of tracks rest on camera lifting alone.** Of 1,160 tracks, 844 have
  a lifted member in > 50% of their frames and neither a `lidar_mos` nor a
  `fusion_detector` member. 360 of them are "moving" cars: a box that jumps
  between the surfaces its mask happens to cover reads as motion.
- **Most of them are wrong.** Of 16 tracks sampled at random (14
  camera-only, 2 camera + detector), 12 are false or boxed on the wrong
  surface, all 12 camera-only. The patterns: car behind a hedge or flower
  bed → box on the vegetation in front (5); far car → box on the trees or
  building behind it (3); person mask → box on the façade behind, in a tree
  canopy, or on the wall a poster hangs on (3); passenger in a car →
  pedestrian box inside the car (1). Overlays: `overlays/` under
  "Reproduce".
- **Nothing downstream can rank them.** `refined_labels.confidence` is 1.0
  on every sampled track (median proposal score, a placeholder). SLF's
  silhouette residual is worse on camera-only tracks (median 0.70, against
  0.42–0.49 when lidar_mos or the detector agrees) but only chooses between
  geometries; it never questions the track. It can't by itself: a box on a
  hedge 1–2 m in front of the car projects onto nearly the same pixels
  (0008-227: residual 0.38, 0005-355: 0.43, both false).
- **Tracking promotes them.** A lone lifted box is *strong* with confidence
  ≥ 0.5 and ≥ 20 seed points; a track is dropped only if short (< 5 frames)
  **and** never strong (`tracking.yaml`). Confidence is class agreement
  between cameras, not evidence that the points are the object.

## Why it happens

Lifting establishes visibility, not ownership. A return is labelled when it
projects into the mask and nothing nearer hides it (UniLiPs Eq. 1,
`semantic_lifting/visibility.py`). When the object itself has no returns —
behind a hedge at LiDAR height, too far for a 32-beam scanner, between
beams — the first surface along those rays passes: the occluder in front,
or the background behind through a gap. `largest_group` then keeps the
largest 2 m-connected part of whatever was claimed, and any ≥ 5 points make
a box. The opt-in `scan_clusters` path keeps the original seed when no
cluster matches (SAL's rule, written for segmentation), so it boxes the same
points. The earlier diagnosis of foreground cars labelling background
(`docs/lifting_background_diagnosis.md`) found the same mechanism behind a
calibration error; with calibration fixed, occlusion and range produce it
on their own.

## How published pipelines handle it

| Source | Output | What stops a mask from boxing the wrong surface |
|---|---|---|
| [UniLiPs](https://arxiv.org/html/2601.05105) (3DV 2026) | per-point semantics; boxes | Boxes **only for moving objects**, from LiDAR points with no match in the static map (HDBSCAN, then a constant-velocity Kalman filter). Camera masks only label points, and static points keep per-point (label, count) histograms over scans. |
| [SAL](https://arxiv.org/html/2403.13129#A1.SS1) (ECCV 2024) | point segmentation | Lifted mask replaced by the best DBSCAN cluster of the full non-ground scan at IoU > 0.5; **otherwise kept** — acceptable for segmentation, where a leaked mask mislabels a few points. |
| [SAL-4D](https://arxiv.org/html/2504.00848) (2025) | 4D segmentation | Same DBSCAN replacement per scan, IoM > 0.5 suppression by temporal volume, window linking. Its method section describes no occlusion or bleeding handling. |
| [FGR](https://ar5iv.labs.arxiv.org/html/2105.07647) (ICRA 2021) | 3D boxes from 2D boxes | Region growing over the **whole** cloud at radii 0.1–0.7 m; a component is kept only if ≥ **0.8** of it lies inside the 2D box's frustum, "otherwise discarded as belonging to another object"; nearest vehicles first, their points removed. |
| [Find n' Propagate](https://arxiv.org/html/2403.13556v2) (ECCV 2024) | 3D boxes from 2D boxes | Frustum search over depth / yaw / class-anchor sizes; best by point density plus projected-box IoU with the 2D box. |
| [SLF](https://arxiv.org/html/2407.11382v2) (ECCV 2024) | shapes from 2D prompts | Prompts come from GT, so false objects aren't addressed; its score is the fitted mask's IoU with the target mask. |
| [MaskClustering](https://arxiv.org/html/2401.07745v2) (CVPR 2024) | 3D instances from 2D masks | A mask whose points scatter over different masks (none holding ≥ 0.8) in > 0.2 of the frames that see it is under-segmented and dropped; masks merge at view consensus ≥ 0.9. |
| [MS3D++](https://arxiv.org/html/2308.05988) (2023) and [its code](https://github.com/darrenjkt/MS3D/blob/main/docs/MS3D_PARAMETERS.md) | 3D boxes from a detector ensemble | **Retroactive labelling**: a track becomes a label only with ≥ 7 detections above the positive threshold (static vehicles 3, pedestrians 1), then labels its whole life. Static pedestrian tracks excluded ("pole-like false positives become distinguishable once they move"). |
| [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/loaders.py) | benchmark | GT boxes with zero LiDAR and radar points are removed before evaluation; this repo's `wato_common.eval` already treats unseen GT as ignore. |

The common thread: in every method that outputs boxes, an object's 3D
geometry comes from LiDAR segmentation, and the camera only says which
segment and what class. A camera mask alone never makes a box.

## Prototype (measured)

`fgr_check.py` scores existing lifted proposals without rerunning anything.
For each one: L = the frame's non-ground points inside its box; for each FGR
radius 0.1…0.7 m, the connected component of the **whole** non-ground scan
holding most of L. It is *verified* when, at some radius, that component
lies ≥ 0.8 inside the frustum of its masks' 2D boxes (FGR θ_seg) **and**
matches L at IoU > 0.5 (SAL). Neither test alone separates (FGR's alone
passes tiny fragments at 0.1 m; SAL's alone passes a hedge strip that is
mostly the lifted points).

**WATO, per track** (sampled frames, 90 tracks over all chunks):

| Tracks | Median share of frames verified | ≥ 50% verified |
|---|---:|---:|
| Camera-only (24 random) | 0.15 | 17% |
| Camera + detector (25) | 0.83 | 84% |
| Camera + lidar_mos + detector (25) | 0.92 | 100% |
| Hand-checked false (12) | 0.00–0.33 | 0 of 12 |
| Hand-checked plausible (4) | 0.82, 0.83, 1.0; 0.0 (a car at 77 m) | 3 of 4 |

**WATO, track confirmation** (chunk 0008, every lifted proposal checked).
A frame is confident when its fused row has a verified lifted member, a
lidar_mos member, or a detector score ≥ MS3D++'s positive threshold (0.7
vehicles, 0.6 pedestrians). Confirmation uses MS3D++'s counts (vehicles ≥ 7
confident frames, parked vehicles ≥ 3, other classes ≥ 1) without its
static-pedestrian exclusion (see nuScenes below). Camera-only: 44 kept, 86
dropped (vehicles 39 / 81). Camera + detector: 39 kept, 7 dropped. Every
track with lidar_mos kept. Five random kept camera-only tracks are all real
objects (cars at 24–57 m, two of them in a parking lot; a turning car; a
pedestrian). Of the seven dropped camera + detector tracks, the four viewed
are cars behind hedges: one boxed on the hedge (false), two ambiguous, one
real car lost. One hand-checked false car survives (0008-227, behind a
hedge, verified on exactly 3 frames) — the case the multi-view step below
targets — and so do most false pedestrians: one verified frame suffices.

**nuScenes, per frame** (lifted proposals at GT instants, 2 m class-agnostic
matching, `wato_common.eval.boxes`):

| Scene | All lifted: P / R | Verified only: P / R | FPs rejected | TPs rejected |
|---|---|---|---:|---:|
| 0061 | 0.708 / 0.323 | 0.805 / 0.216 | 85 of 145 | 118 of 351 |
| 1100 | 0.809 / 0.598 | 0.830 / 0.404 | 36 of 87 | 120 of 369 |

A single frame's verdict is too noisy to delete boxes by: a real object
fails it in about a third of frames (crowds, adjacent cars, partial
returns). That is why the design below uses it as evidence a track
accumulates, not as a per-frame filter.

**nuScenes, track confirmation** (every lifted proposal checked; owned
track boxes at GT instants, same matching; a track is *true* when most of
its matched instants are hits):

| Scene | All tracks: P / R | MS3D++ rule as published: P / R | Without static-pedestrian exclusion: P / R | Vehicles only: P / R |
|---|---|---|---|---|
| 0061 | 0.723 / 0.319 | 0.704 / 0.199 | **0.785 / 0.313** | 0.785 / 0.316 |
| 1100 | 0.813 / 0.598 | 0.813 / 0.374 | **0.822 / 0.593** | 0.816 / 0.596 |

- The published rule's recall loss is almost all MS3D++'s
  `USE_STATIC_PED_TRACKS: false`: it drops 31 true standing-pedestrian
  tracks across the two scenes. MS3D++ uses it only for its first
  self-training rounds; final labels need standing people, so it is left
  out below.
- Without it, confirmation removes 40 of 133 false boxes on 0061 for 7 of
  347 true ones. On 0061, 76 of the 133 false boxes are ≥ 8 m from any GT
  (objects that aren't there — the WATO failure). On 1100, 64 of 85 are
  within 4 m of a GT box (a real object boxed badly), which existence checks
  don't address; precision barely moves there, as expected.
- On 0061, true cars verify in a median 86–95% of their frames and false
  cars in 34–39%; true pedestrians verify in only 41–68% (both scenes).
  The check is strongest for vehicles, which is what FGR was designed for.

These scenes' artifacts predate independent masks (Nov 3); they measure the
check, not the current pipeline. There is no WATO ground truth: the WATO
figures are inspection, not precision/recall.

## Design

One principle, applied at each stage: **the camera proposes, LiDAR
confirms.** A mask says what an object is and roughly where it lies (a
bundle of rays); the object's existence must come from LiDAR evidence that a
wrong association can't easily fake.

**Built (2026-11-17), both on by default after gate A:**

1. **Per-frame LiDAR verification** (semantic_lifting + proposal_generation,
   `lifted.verification`). Lifting records each instance's 2D box frustum
   (every non-ground return in its masks' rectangles, visible or not) beside
   its claims in `lifted_candidates`. proposal_generation checks every
   seeded instance against the frame's existing SAL DBSCAN bank: at each
   scale, the cluster with the best IoU against the seed must match it at
   IoU > 0.5 (SAL) and lie ≥ 0.8 inside the frustum (FGR θ_seg); any scale
   passing verifies it (FGR keeps the best component over its radii). The
   verdict is `lidar_verified` on lifted rows. **Geometry is unchanged**:
   `largest_group` still boxes the seed, and an unverified lifted row is
   still a normal detection that can start a track.
2. **Track confirmation** (tracking, `confirmation`): MS3D++'s retroactive
   labelling. A detection is confident with a lidar_mos member, a verified
   lift, or as a detector-only box at the existing ByteTrack birth score
   (0.6); a matched detector never adds to a lift. A linked track is kept
   only with ≥ 7 confident instants (parked vehicles 3, other classes 1),
   then keeps its whole life. MS3D++'s static-pedestrian exclusion is left
   out (on nuScenes it cost 31 true standing-pedestrian tracks). The
   `strong` rule is unchanged. `track_confirmation.parquet` records every
   judged track's evidence and decision.

Differences from the prototype measured above, each chosen by a
measurement during planning: the bank is SAL's DBSCAN (six radii,
`min_samples` 5) rather than FGR's connected components (nuScenes 0061: 0.69
of true and 0.34 of false lifted boxes verify, against 0.65 and 0.38); the
seed is the instance's candidate claims rather than the points inside its
box; and the frustum comes from lifting on its rounded pixels. Counting
matched detector boxes as confident changed nothing on WATO chunk 0008, and
requiring 7 for "unknown" (too short to judge) vehicles was as good as any
smaller count on nuScenes GT.

**Results on nuScenes GT** (rerun from lidar_preprocessing, confirmation
off → on): frame precision 0.74 → 0.80 (0061) and 0.82 → 0.84 (1100) for
recall 0.34 → 0.33 and 0.62 → 0.61; false tracks 37 → 16 and 12 → 6, true
tracks 51 → 48 and 30 → 29; HOTA 0.38 → 0.38 and 0.64 → 0.65; refined labels
0.76 → 0.82 and 0.84 → 0.86 precision. Tables:
`docs/tracking_measurements.md` ("Confirmation"),
`docs/proposal_generation_measurements.md` ("LiDAR verification").

**Results on WATO** (no GT; rerun from semantic_lifting, confirmation off →
on; `docs/tracking_measurements.md`, "Confirmation on WATO"):
- Tracks go from 1,160 to 565.
- Of camera-only tracks, 271 stay and 573 go. Tracks with detector or
  LiDAR-motion evidence almost all stay (294 kept, 21 dropped).
- 9 of the 12 hand-checked false tracks go; 3 of the 4 plausible ones stay.
- Near the car (< 20 m), the camera-only tracks that go are almost all
  false: hedges, signs, trees, the car's own body.
- Two costs:
  - real cars beyond about 50 m with only a few returns;
  - real people standing against a hedge or storefront. Their returns
    merge with it at every SAL scale, and their seeds also hold returns from
    metres behind them, so no cluster matches the seed (IoU 0.1–0.3).

The pipeline's verdict matches an independent recomputation from the mask
PNGs on 954 of 958 sampled rows; the 4 others are half-pixel rectangle
edges (`docs/proposal_generation_measurements.md`).

**Tried after A, each behind its own measurement:**

3. **Box the verified segment** (proposal_generation,
   `support_method: verified_clusters`, opt-in): a verified instance's
   support becomes the cluster that verified it, at SAL's best-IoU scale
   (SAL replacement); unverified keeps `largest_group`. On nuScenes the
   cluster holds more of the object than `largest_group` at the same
   precision, and 0061's boxes and ID switches improve. It loses on sparse
   objects: a car or truck with under about 20 returns splits into pieces
   at every DBSCAN scale and is boxed on the largest piece (1100's one
   night car at 22 m, 0061's moving truck at 34–43 m). Not the default
   (`docs/proposal_generation_measurements.md`, "Boxing the verified
   segment"; a first run that took the coarsest passing scale is there
   too).
4. **Multi-view consistency for parked tracks**: MaskClustering's
   under-segmentation test, with its thresholds, on a parked track's
   per-frame masks, judged per image as in the paper. An image observes a
   mask when ≥ 0.3 of its claimed points project into it unoccluded
   (lifting's LiDAR occlusion test); the mask is filtered when, in > 0.2 of
   the observing images, < 0.8 of those points fall in the track's own mask.
   A filtered frame stops counting as confident. Read-only study only
   (`eval_runs/2026-11-17_gate_a/phase_c_maskclustering_v2.py`; the first
   version counted points outside a camera's view as misses), not built:
   - **nuScenes:** it drops 1 false and 1 true parked vehicle track and
     keeps 5 false ones.
   - **WATO:** it keeps the hedge car 0008-227 and drops 12 of 30 sampled
     kept parked tracks, real-looking cars among them (0006-240).
   - **Why:** the test needs each mask's points to be mostly its object.
     On nuScenes, 25–35 % of the lifted seeds that match a true object have
     more than 20 % of their returns outside its GT box grown by 0.25 m
     (`seed_purity.py`), so true parked cars fail many views with good
     calibration too. MaskClustering back-projects each mask's own pixels
     from a depth image registered to the colour image; a lifted seed is the
     LiDAR returns, seen from another viewpoint, that pass a sparse occlusion
     test, and it carries background.
5. **False pedestrians** (study, `phase_d_pedestrians.py`):
   - **Detector distance:** on nuScenes, true pedestrian tracks lie 0.2 m
     from a BEVFusion pedestrian (median) and false ones 3.5 m. But those
     scenes are BEVFusion's training data. On WATO, the one checked true
     person is 1.8 m from the nearest BEVFusion pedestrian and the van case
     3.0 m.
   - **Verified share:** the true person verifies on 0 of 16 frames; the
     poster on 0.46 of its frames.
   - **Box height:** the poster's box is 5.3 m tall, but the true person's
     is 0.9 m (partly hidden).
   - **Ground contact:** every case still kept has its box bottom on the
     ground; only the canopy, already dropped, does not.

   No measured feature separates true from false pedestrians on WATO, so no
   change is proposed. Labelled WATO pedestrians would be needed first.
6. **A confidence that means something** (label_refinement): the
   confirmed-instant share and, where SLF ran, SLF's own mask-IoU score,
   calibrated on the GT scenes.

What this deliberately does not do: invent a range cut, a vegetation
filter, a per-class point floor or a tuned score. Every threshold above is
the cited source's; MS3D++'s counts were set for its own frame rates and
detectors, so they are starting points to re-measure on the GT scenes, not
WATO-validated values.

## Limits and open questions

- A car behind a hedge with no LiDAR return on it gets no box. For a LiDAR
  student that is the right label: nuScenes evaluation drops such boxes.
  Ignore regions for camera-only objects (KITTI's DontCare) are an option
  if the student's loss supports them; nothing does yet.
- **The check is weak for people, both ways.** On WATO the façade and
  canopy cases go, but the poster and the pedestrian placed by a passing
  van stay. By eye, 5 of 16 sampled dropped pedestrian tracks are real
  people in front of a hedge or storefront. Their returns merge with what
  they stand against at every SAL scale, so verification fails on IoU;
  mask alignment is not the cause (`docs/tracking_measurements.md`).
  - On nuScenes, by contrast, confirmation keeps 51 of 53 true non-vehicle
    tracks.
  - A passenger inside a car's component is an ownership question (SAL-4D's
    IoM suppression against the car's support).

  Open.
- Semantic errors survive verification: a road sign boxed as a truck
  (0010-98) is a distinct LiDAR object inside its mask. Class-shape
  plausibility (class size anchors as in Find n' Propagate; ground contact)
  is a separate step.
- MS3D++'s counts assume its detectors' scores and frame rates; here
  confidence is a binary check at 10 Hz (WATO) or 20 Hz (nuScenes). They
  are the defaults on gate A's evidence; a sweep of them on the GT scenes
  (precision, recall, `eval_tracks.py` fragmentation) is still open.
- Badly placed boxes on real objects (most of nuScenes 1100's false
  positives) are a geometry problem for refinement, not an existence one.
- **The segment bank is one scan.** It is DBSCAN over the frame's own
  non-ground returns. A sparse object (far, dark, or seen at night) splits
  into pieces or falls below `min_samples`, so it fails verification or,
  under `verified_clusters`, is boxed on one piece. CPD (CVPR 2024, §3.1)
  and UNION (NeurIPS 2024, §3.3) cluster several scans aggregated in the
  world frame instead, after removing moving points from the other scans.
  That densifies static objects only; a moving one (1100's night car) gains
  nothing. On the nuScenes GT scenes real parked vehicles already verify in
  0.83–0.93 of their frames at every range (`verify_fail_real.py`), so the
  measurable gain there is small; WATO's far parked cars are the case.
- label_refinement places a moving track's box bottom on the chunk's ground
  grid at its centre. Under a large parked truck the grid had single cells
  2–4 m above the road (Patchwork++ ground on the truck's returns, labelled
  from beyond its first zone), which lifted its box by 3 m in some frames
  (nuScenes 0061, GT 19319). Step C now builds the grid from Patchwork++'s
  first zone only (`docs/lidar_preprocessing_measurements.md`, "Ground
  grid").
- Side finding: nuScenes world sweeps kept ~8,000 returns per sweep off the
  ego roof and sensor mount, all within the devkit's 1 m `remove_close`
  square and none at exactly zero range (WATO: ~90–100 roof-rack returns).
  lidar_preprocessing now drops them with `remove_close_m` (1.0 m on
  nuScenes, off on WATO; `docs/lidar_preprocessing_measurements.md`,
  "Returns on the vehicle itself").
- WATO's results above are judged by eye. Labelled frames
  (SUSTechPOINTS, `src/ingest/scripts/`) in a parking lot and along hedged
  streets are needed to measure them.

## Reproduce

Scripts and results: `data/artifacts/eval_runs/2026-11-16_lifted_fp/`
(git-ignored), run from the repository root with
`ARTIFACT_ROOT_URI=file://$PWD/data/artifacts PYTHONPATH=<dir>:src/common/src`
(`remap.py` maps the manifests' `/data/...` paths to the host).

- `sources.py` — per-track evidence sources (`tracks.json`).
- `overlay.py <out> <track_id>...` — mask, LiDAR and box per supporting
  camera at a track's middle frame (`overlays/`).
- `fgr_check.py <bag> <chunk> <out.jsonl> [@ids]` — the check; `GT_ONLY=1`
  restricts to GT instants, `SHARD=i/n` splits by sweep. Returns within
  2.5 m of the sensor are left out of the component bank; the WATO results
  (`results/res2`, `results/full`) predate that and include WATO's ~90 such
  points per sweep, the nuScenes ones (`results/nfull`) exclude them.
- `confirm.py` / `confirm2.py` — MS3D++ confirmation on one WATO chunk, with
  / without the static-pedestrian exclusion.
- `eval_nusc.py` (per frame), `eval_tracks_confirm.py` (published rule),
  `eval_tracks_confirm2.py` (`KEEP_STATIC_PED=1`, `VEH_ONLY=1` variants),
  `nusc_breakdown.py` (per class / motion, false-box distances) — nuScenes
  scoring.

Gate studies after the build: `data/artifacts/eval_runs/2026-11-17_gate_a/`
(git-ignored), same invocation; those importing proposal_generation also
need `src/proposal_generation/src` on `PYTHONPATH`.

- `gate_b_compare.py <scene> <tag>...` — tracks and refined labels per run.
- `gate_b2_iou.py <scene> <tagA> <tagB> [moving|parked]` — refined boxes per
  GT vehicle, instants matched in both runs.
- `verified_cluster_points.py [scene]` — verifying cluster against
  `largest_group` support, by the GT object's returns.
- `verify_fail_real.py` — verification on real vehicles by motion and range.
- `seed_purity.py`, `phase_c_maskclustering_v2.py`, `phase_d_*.py` — items
  4 and 5.
