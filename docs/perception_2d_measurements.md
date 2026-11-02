# perception_2d: measurements behind the defaults

The measurements and past fixes behind perception_2d's defaults. The behaviour
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
  pedestrian track left as 6–80 px on a car's hood for 313 frames: it latched
  onto the car, which owns the contested pixels, and its raw mask's IoU with
  the car's stays under `duplicate_mask_iou` because it covers only part of it.
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

Collapse threshold (`collapse_fraction`, `collapse_after_frames`), from the
share of its own mask each contested track owned after ownership on
0009 / PANO_SS (280 tracks): the hood parasites owned 0–3% (8 specks); real
distant cars whose neighbours' masks bled over them owned 15–80% at the median
but dipped lower now and then. Ending on any frame under 0.2 ended 83 tracks, 52
of them with a median above 0.3 (real objects), and lost 3.5% of the old run's
well-formed masks; 3 frames in a row under 0.05 ends 23, 4 of them real.

Containment (`contained_classes`, `container_classes`) on 0009 / PANO_SS and
0006 / PANO_EE: with pedestrians contained in car / truck / bus, 34 pedestrian
tracks were removed, and all 34 were drivers, passengers or window interiors
(one labelled a windshield `pedestrian` for 200 frames); no car changed.
Containing cars as well removed one cabin track but cut the old run's
well-formed masks still covered on 0009 / PANO_SS from 97.2% to 94.4%: distant
cars inside a neighbour's bleeding or smeared mask, which also ends above
theirs.

The ego mask, island, collapse and pedestrian containment rules together
against the old run, per clip: ego share of mask pixels 5-24% -> 0; flagged
fragments 3.8-8.3% -> 0-2.1%; old well-formed masks still covered 97-99% on
0009 / PANO_SS and 0006 / PANO_EE (the loss is the drivers), 82-89% on the two
PANO_SW clips, where the loss is rack and roadside `truck` masklets the old
flags missed. Still there: a pedestrian mask drifting onto parked cars, and
`truck` masks born from 0.26-0.40 boxes over the road beside the rack, one
spreading over the whole sky in 0004 / PANO_SW.

Detector support (`support_keyframes`, `support_min_area`, `birth_score`) on
the same four clips. The detector was replayed on every keyframe against the
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

All of these rules, detector support included, against the pipeline before
them (`compare`, 2026-11-01): commit 4866411 rerun on chunks 0004 and 0010 into its
own artifact root and compared with the current run, every 5th image, IoU 0.5.
176 old masklets lost masks (~10,800 masks), all reviewed: 6,649 inside the ego
mask (14 masklets: the rack and hood `truck`s, a rack `bicycle`), 1,850 road
or sky (4), 1,276 people inside vehicles (39), 139 specks, signs and a mask
drifted onto pavement (22), 801 on objects a current mask still covers at
IoU < 0.5 (88), 75 unsure (7 distant cars in clusters the current run masks,
identity unclear), and 10 real: a pedestrian beside another, unmasked for its
first 5 images, and a distant car boxed at 0.74 whose old masklet lasted 5
images. Real-object masks lost: car 5 of 36,804 (0.2% counting the unsure),
pedestrian 5 of 5,092 (0.1%), truck, bus and bicycle none (the old run had no
motorcycle). The current run has 410 masks the old one lacks (car 270,
pedestrian 110), not reviewed. Both runs share the detector, so this bounds
what the rules removed, not what the detector misses.

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
image. That is why tracks end (`drop_lost_after_frames`, `duplicate_mask_iou`).

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

## One entry per image (2026-10-15)

Before 2026-10-15, `frame_index` rows were fed straight through: each WATO image
went through Depth Anything ~5× (the last row's single sweep anchored the
surviving file), SAM2 held ~5× the frames in host RAM (chunk 0001 of
`recording_20260309_211754_0_with_eidos_trimmed` was SIGKILLed at 1,154 frames
per camera for ~238 images), and `frames_present` repeated each `camera_seq`
~5×. The frame-counted defaults were then 10 / 5 / 5 rows
(`redetect_every_k` / `drop_lost_after_frames` / `embeddings.every_k_frames`);
2 / 1 / 1 images keeps that behaviour. On that bag's chunk 0000 (parked in the
garage), 2 / 1 / 1 gave 63 masklets over 6,647 masked images against the old
60 over 6,545, while 10 / 5 / 5 as images found 49 over 6,018 (10 fewer
pedestrians). Outputs written before then aren't comparable.

## Depth fit (2026-10-11)

The original fit modelled depth as an affine of disparity, which is the wrong
shape (the true slope is negative and hyperbolic). Those files carry
`affine_a` / `affine_b` instead of `disp_scale` / `disp_shift`, and the reader
rejects them with `ArtifactFormatError`.

## Model registry (2026-09-21)

Florence-2 was referenced by `discovery.model` but missing from the registry
until 2026-09-21, so it was never pre-fetched and enabling discovery on a clean
host failed at load time.
