# perception_2d: measurements behind the defaults

The measurements behind perception_2d's defaults. The behaviour
itself is in the component README. "The debug clip" is 181 frames of one WATO
pano camera at 1280×1024; timings are on an RTX 5070 Ti.

## Detector backends

- YOLO26-x takes ~30 ms per keyframe at `imgsz: 1280`.
- Ultralytics treats a numpy image as BGR. Passing the pipeline's RGB frames
  straight through measurably changed detections on a WATO frame (a fire hydrant
  and a stop sign were lost, every score shifted), so `detect()` converts.
- COCO semantics, not bugs: the campus shuttle comes out `truck`, and a cyclist
  is a `person` (→ pedestrian) plus a separate `bicycle`.
- Ego-vehicle false positive: PANO_NN (front) sees the hood along the bottom
  ~15 px (x ≈ 450–850 at 1280×1024). YOLO scores it `car` at 0.26–0.35 and SAM2
  tracks it for the rest of the clip (a 161-frame masklet on the debug clip).
  GroundingDINO did not fire on it.

## Mask propagation backend: stock SAM2 vs DAM4SAM (2026-11-05)

Both nuScenes GT scenes run end to end with each `segmentation.backend`,
everything else the shipped defaults and the same code (CUDA hole filling on
for both), each into its own artifact root from the same ingest and
lidar_preprocessing outputs, on an RTX 4090. Scored by the boxes downstream
(`eval_proposals.py`, `eval_tracks.py`, `eval_refined.py`, 2 m matching):
there is no mask ground truth.

| scene | backend | masklets | fused hits / preds | true / false tracks | objects in one track | refined recall | car IoU | car yaw (°) |
|---|---|---|---|---|---|---|---|---|
| 0061 | sam2 | 693 | 352 / 495 | 60 / 46 | 0.45 | 0.34 | 0.36 | 2.9 |
| 0061 | dam4sam | 710 | 356 / 497 | 62 / 45 | 0.50 | 0.35 | 0.36 | 2.5 |
| 1100 | sam2 | 222 | 372 / 461 | 37 / 19 | 0.56 | 0.60 | 0.30 | 2.9 |
| 1100 | dam4sam | 233 | 368 / 458 | 35 / 22 | 0.56 | 0.60 | 0.30 | 3.1 |

- **No measurable difference.** Every gap is a few matches in ~350 or two or
  three tracks, and its sign flips between the scenes; identity switches
  (0.06 → 0.04 on scene-0061, 0 on scene-1100) and motion-state accuracy
  (0.78, 0.88 for both) agree too. Boxes here come from LiDAR points, so a
  mask change matters only where it changes which points an instance claims.
- **The masks barely differ either:** 51.1k vs 50.6k and 43.1k vs 43.1k
  masks, empty-mask share 2.1 / 2.3% and 0.6 / 0.6%, median threshold
  stability 0.936 / 0.936 and 0.930 / 0.932. DAM4SAM's masklets are slightly
  more and shorter (median 44 vs 45, 80 vs 88 images).
- **Same cost:** the perception pass (depth included) took 13 min on
  scene-0061 with either and 12.8 / 14.2 min on scene-1100.
- DAM4SAM stays the default: nothing measured here favours either, and its
  published case (distractors over long clips) is WATO's panoramic clips,
  which have no ground truth. These scenes are 20 s; whether it helps on
  WATO is unmeasured.

## Birth gate, back-fill and DAM4SAM memory (2026-11-07)

`sensors_2026_07_07-23_01_24_2` chunk 0005, PANO_SS, all 679 images, through
`scripts/diagnose_masks.py` (YOLO + DAM4SAM, no depth or DINOv2) on an RTX
4090, the same code both times. "Old" turns the four changes off
(`birth_gate: none`, `backfill_frames: 0`, `dam4sam.update_delta: 5`,
`memory_min_predicted_iou: 0`); "new" is the shipped `perception_2d.wato.yaml`.
A second "new" run reproduced the first exactly.

| policy | masklets | masks | births refused | back-filled masks | predicted IoU ≤ 0.5 | time |
|---|---|---|---|---|---|---|
| old | 208 | 14,833 | – | – | 21.8% of masks | 364 s |
| new | 122 | 8,769 | 279 of 401 | 267 (74 tracks) | 13.8% of masks | 274 s |

- **Refused births are duplicates.** By detector score: 135 / 191 refused at
  0.25-0.4, 100 / 123 at 0.4-0.6, 44 / 63 at 0.6-0.8, none of 24 at ≥ 0.8.
  All 16 of a random sample were drivers boxed as `pedestrian` inside a
  tracked car, a pedestrian already tracked, or a box on a car whose mask
  already covers it.
- **What the old run had that the new one lacks.** Matching both ways at
  pixel IoU 0.5 on every 5th image (as `compare` does), 2,198 old masks
  (14.8%, 90 masklets) have no new mask. The 16 largest losses (~1,540 of
  those masks) are drivers and windshields (8 masklets, 1,088 masks: the
  `pedestrian` / `car` boxes inside cars that ran for 48-248 images), a second
  mask over the top of the car behind the ego vehicle (a 0.82 box, 173
  masks), a part box on a car, and a cluster of distant cars both runs mask
  but split differently. None was a real object the new run lost. The other
  74 masklets are unreviewed.
- **Back-fill** gained 74 of 112 later births a median of 2 images (stops:
  53 lost mask, 56 already tracked, 3 at the 20-image span). Long back-fills
  extend the same object, junk tracks included (a price sign as `truck`, a
  hydrant as `pedestrian`). One drifted: a bus followed back behind a hedge
  latched onto a post for ~8 images, with predicted IoU 0.71-0.83 throughout,
  so neither SAMURAI's 0.5 gate nor Seg2Track's ≤ 0.1 removal would stop it.
- **Each change alone**, on the same camera (old policy plus that one change),
  matched against "old" both ways at pixel IoU 0.5 on every 5th image:

  | change alone | masklets / masks | old masks unmatched | its masks unmatched |
  |---|---|---|---|
  | birth gate | 121 / 7,958 | 2,105 (14.2%) | 46 (0.6%) |
  | back-fill | 208 / 15,081 | 0 | 245 (1.6%) |
  | `update_delta: 2` | 198 / 13,687 | 501 (3.4%) | 245 (1.8%) |
  | memory gate | 203 / 14,910 | 640 (4.3%) | 337 (2.3%) |

  The gate makes the whole reduction and back-fill only adds; without the
  gate, 138 of its 197 back-fills stop on the track they duplicate. Δ = 2 and
  the memory gate reshape masks both ways; the largest differences either way
  are junk or duplicate masks (windshields, drivers, a second mask on the car
  behind), no real object gained or lost among those viewed. Δ = 2 lowers the
  share of masks with predicted IoU ≤ 0.5 to 18.0%; the memory gate alone
  raises it to 23.5% (it refused 3,423 frames).

### Downstream, on nuScenes and a WATO chunk

Each arm runs perception through label_refinement in its own artifact root,
linked to the same ingest and lidar_preprocessing outputs, with the default
configs; "new" reuses "old"'s depth maps, so the mask policy is the only
difference. "+gate" is old plus the birth gate; "−back-fill" is new without
back-fill. Scored at 2 m (`eval_proposals.py`, `eval_tracks.py`,
`eval_refined.py`). Roots: `data/artifacts/eval_runs/2026-11-07_mask_policy/`
(the two-chunk scene's ingest and LiDAR outputs are in its `nusc1100_new`
root, whose absolute URIs they hold).

| scene | arm | masklets | fused hits / preds | true / false tracks | switched | motion | refined R / P | car IoU | car yaw (°) |
|---|---|---|---|---|---|---|---|---|---|
| 0061 | old | 694 | 362 / 495 | 61 / 54 | 0.082 | 0.76 | 0.356 / 0.796 | 0.36 | 2.5 |
| 0061 | +gate | 575 | 361 / 491 | 58 / 51 | 0.070 | 0.74 | 0.355 / 0.794 | 0.36 | 2.5 |
| 0061 | −back-fill | 573 | 360 / 498 | 60 / 49 | 0.055 | 0.76 | 0.354 / 0.783 | 0.36 | 2.9 |
| 0061 | new | 573 | 367 / 518 | 63 / 53 | 0.056 | 0.74 | 0.361 / 0.761 | 0.33 | 6.2 |
| 1100 | old | 226 | 376 / 461 | 39 / 19 | 0.049 | 0.92 | 0.611 / 0.838 | 0.30 | 5.4 |
| 1100 | −back-fill | 168 | 378 / 460 | 34 / 17 | 0.054 | 0.92 | 0.614 / 0.844 | 0.30 | 3.0 |
| 1100 | new | 168 | 385 / 476 | 42 / 19 | 0.045 | 0.96 | 0.624 / 0.828 | 0.30 | 3.0 |
| 0061 ×2 | old | 969 | 362 / 496 | 66 / 48 | 0.066 | 0.81 | 0.349 / 0.780 | 0.37 | 4.8 |
| 0061 ×2 | new | 820 | 368 / 518 | 67 / 50 | 0.051 | 0.83 | 0.363 / 0.770 | 0.37 | 9.5 |

"switched": share of tracks switching objects; "motion": motion-state
accuracy when judged. The two-chunk seam keeps 0.31 of 13 crossing objects in
one track either way.

- **The gate removes masklets without touching the boxes**: 17-26% fewer,
  identical refined recall, precision, car IoU and heading on 0061, and fewer
  identity switches. It refused 30-42% of the boxes no track matched.
- **The memory changes** (Δ = 2, SAMURAI's gate) are within noise on 0061
  and lower car heading error on 1100 (5.4° → 3.0°).
- **Back-fill** adds 6-9 fused hits per scene (about +2% recall) for 16-22
  more boxes, so precision falls 1-2 points; on 1100 it adds 8 true tracks
  and lifts motion-state accuracy to 0.96. On 0061, in both chunkings, it
  also extends one parked car (GT 22183) so that its box absorbs extra
  points: as one chunk its two old tracks become one whose centre wanders
  1.9 m, it is judged moving and sized from percentiles (+0.38 m wide,
  +1.09 m long); as two chunks it stays parked but grows +0.42 m / +1.25 m.
  That one car, with about nine GT cars in the scene, is the whole car IoU
  and heading regression. A car only the new policy finds (GT 2143) adds a
  poor box (1.65 m off).
- **WATO chunk 0005** (all 8 cameras, same depth, no ground truth): masklets
  1,081 → 831, lifted instances 798 → 593, labelled points 513k → 521k,
  lifted rows 8,586 → 8,888, fused rows 10,024 → 10,322. proposal_generation
  discards 33% fewer 3D candidates (14,925 → 10,051) while emitting more:
  the duplicates the gate refuses were reaching 3D only to be thrown away.
- **Full bag** (the 2026-11-07 rerun against the 2026-11-06 run, which also
  predates independent masks): 8,302 → 6,105 masklets, 5,331 → 4,400 lifted
  instances, 88,435 → 90,800 fused proposals, 2,669 → 2,757 tracks; the gate
  refused 6,653 of 12,763 unmatched boxes.

## Masks on a full bag (2026-10-31)

`sensors_2026_07_07-23_01_24_2` (eidos poses), 11 chunks, 8522 masklets;
flags from 8 evenly spaced masks per masklet:

- Ego body: the rear and side cameras (PANO_SS / SE / SW, once NW) see the
  vehicle's rack along the bottom of the image. YOLO boxes it as `truck` at
  0.25–0.42 and SAM2 tracks it for the whole chunk: 44 masklets (0.5%) holding
  31% of all mask pixels. Some of these masks then spread over the road or the
  sky (a 0.31 `truck` covering the whole sky in 0004 / PANO_SW).
- Drift: a pedestrian prompted at the image edge became a 341×58 px smear over
  parked cars and a hedge within 40 frames and was recorded for 160 more. 79 of
  878 pedestrian / bicycle / motorcycle masklets are 1.6× wider than tall.
- Parasites: 5.9% of masklets are speckle (median fill < 0.2 or < 60 px), e.g. a
  pedestrian track left as 6–80 px on a car's hood for 313 frames.
- 37.5% of masklets are seeded by a detection scoring < 0.35.

The ego mask (`perception_2d.wato.yaml`) was traced from the per-pixel median of
every 20th image of the bag: the scene blurs out while driving and the rack
stays sharp. The masks of the 44 rack masklets fit it closely; PANO_SS also
sees an antenna above the rack that no masklet covered, and EE / WW see no part
of the car (WW's median shows only a faint shadow, which YOLO never fired on).
Polygons are the union, dilated 6 px, simplified to within 2 px.

`ego-mask` automates that (`ego_mask.py`). Pixel statistics could not:
over the drive the rack's grey-level std (median 27–51 inside the polygons) is
no lower than the scene's, and the median frame-to-frame difference is 0–2
grey levels on sky and road as on the rack, so neither thresholds to the body
(IoU ≤ 0.5 with the polygons at any threshold). Run on chunks 0006–0010 of the
run without an ego mask, its body test picked 19 masklets on the six cameras
that see the car (none on EE / WW); at most 1.9% of their pixels fall outside
the traced polygons, and its draft polygons match them at IoU 0.95–0.99. What
the draft lacks is the PANO_SS antenna (2.7% of that camera's polygons), which
no masklet covered.

Detector support (`support_keyframes`, `support_min_area`, `birth_score`) on
four clips: 0009 / PANO_SS, 0006 / PANO_EE and two PANO_SW clips. The detector was replayed on every keyframe against the
masks of a run without the support rules (417 masklets, 9150 keyframe-masks),
and the tracks it removes were labelled by eye: 11 junk (the sky / road / hood
`truck`s, the pedestrian smear, a billboard `truck`, blobs on hedges and signs,
a person behind a shop window) and 15 real (cars parked behind a hedge or in
lots, 20–60 px wide). Real distant cars are boxed once at 0.25–0.4 and then at
3–40% of keyframes; junk at 0–13%, so support alone doesn't separate them:

- SAM 3's rules as such: confirmation (support at the next 1–2 keyframes, or
  SAM 3's hotstart, at most half the first 6–8 missed) kept 26–48% of the
  real keyframe-masks; an ungated keep-alive of 15 kept 76% of real and 16% of
  junk, of 50 kept 97% / 40%.
- By size: masks over 0.3% of the image, other than junk, are boxed at 89–97% of
  keyframes (under 0.1%: 63%); junk over 0.3% at 0–2%. Counting only keyframes
  whose mask bbox covers ≥ 1% of the image, keep-alive 5 kept every real
  keyframe-mask and cut two unlabelled masklets besides the junk: the road
  then hood of 0006 / PANO_EE (`car` 0.25, 458 frames) and a minivan masked
  over its rear half only. Junk kept: 45% — the small static false positives
  (billboard, hedge blobs), which only a better detector removes.
- Mask growth since the last supported keyframe (> 3–6×) cut 3–6 masklets
  and only the smear among the junk: the sky and road masks are large from
  the first frame (the sky `truck`'s seed box was 774×527).
- Re-prompting fired 0–3 times per clip and changed the number of masks
  written by at most 0.2%; whether it improves them is unmeasured.

Run with those defaults (keep-alive 5 over 1%, no confirmation), the four clips
lost 1770 of 18098 masks (10%), all from 6 tracks: the smear (from where its
mask passed 1% of the image), both 0004 / PANO_SW `truck`s (698 masks, 29% of
that clip), the 0001 road `car` and half-masked minivan, the 0006 road / hood
`car` (458 of that clip's 846). Every other mask of the run without the rules
is still covered at IoU ≥ 0.5.

Tracks born touching the left, right or top image edge (1690, 20% of masklets)
are not worse: 1.2% fragments against 7.1% for tracks born inside, and the same
0.9% of wide pedestrian / bicycle / motorcycle masks. They only grow more than
8× between samples more often (16% vs 4%), as objects entering the view do. So
births at the border are not suppressed.

Traffic cones and barriers, the two taxonomy classes COCO lacks: the pinned
GroundingDINO prompted with only `traffic cone . barrier .` over 1992 sampled
images (every 30th per camera) gave 252 cone and 1065 barrier detections ≥ 0.25.
The top barrier hits were sound walls, fences and the ego rack; no temporary
road barrier appeared. Cone hits mixed a few real cones and construction drums
with fire hydrants, yellow bollards and poles. Neither is usable as labels, so
both are left out of the taxonomy.

## SAM2 track ending

When an object leaves the image or is covered, this SAM2 build keeps its id
alive and within 0–30 frames usually reports it on a different, similar object
nearby. On the debug clip every such resume was wrong: one pedestrian's masklet
walked out of view and reappeared on a car, three other pedestrians and a parked
car; an oncoming car's masklet jumped onto a parked car in the frame it left the
image. That is why tracks end (`drop_lost_after_frames`).

The cost: a genuinely occluded object comes back as a new masklet, and an
object the detector never finds is no longer covered by a hijacked track (5
frames of a dark figure GroundingDINO never detected were previously masked
under another pedestrian's id as `motorcycle`).

## Speed (debug clip, full tracking pass)

| setting | time | peak VRAM | masklets |
|---|---|---|---|
| float32, never end tracks | 283 s | 7.3 GB | 25, four hopping between objects |
| bfloat16, never end tracks | 63 s | 7.3 GB | 26 |
| bfloat16 + current defaults | 27 s | 4.6 GB | 26, none hopping or duplicated |

Runs are bit-for-bit repeatable. bfloat16 moves masks no more than feeding SAM2
the source JPEG instead of a re-encode does (per-frame IoU vs the float32 path
0.981 vs 0.980 mean). SAM2's `vos_optimized=True` (torch.compile) is unusable
with this per-object tracker: 224 s of compile, then a CUDA-graphs buffer-reuse
error in memory attention.
