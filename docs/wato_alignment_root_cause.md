# WATO alignment: stage isolation and independent calibration evidence

Measured 2026-11-03 on `sensors_2026_07_07_23_01_24_2_with_eidos_mcap`.
The supplied screenshot is **chunk 0005, PANO_NN image 50, reference sweep 98**.

**The strongest root-cause evidence for this screenshot is incorrect NN camera
orientation relative to the sensor/ego geometry.** It misprojects real moving
vehicle returns below their image and makes lifting associate a car mask with
an unrelated foreground surface. Correcting the orientation estimated from
independent image/ego-motion data changes that association in the actual
lifter, without changing the masks or proposal-fitting parameters.

This supersedes the earlier inference that partial-box construction alone
explained the screenshot. Ground extension explains how its wireframes were
constructed; it does not establish why the selected observations and image
were inconsistent. The viewer time-selection fix is also a separate bug:
this particular image has only one paired proposal time.

## Follow-up: rerunning the original box-alignment audits

The original audits were rerun with the existing NN-only and all-camera
candidates, without fitting new parameters. Results and an interactive
comparison are in `data/artifacts/box_alignment_audit/extrinsics_rerun/`
(`README.md`, `index.html`, `summary.json`).

The user's later screenshot is **chunk 0004, NN image 451, reference sweep
894**. On its seven original car targets, actual relifting and proposal
fitting changes median projected-box/mask-rectangle IoU **0.300 → 0.693**
with the NN candidate, including one missing proposal as zero overlap.
Across the 40 original sampled NN image times, 165 fixed car-target views
change **0.346 → 0.634**. Eleven candidate proposals are missing and remain
in the denominator. The all-camera experiment gives 0.665 on those NN
views, with eighteen missing. Other cameras have only projection rerenders
in this follow-up; the broader original projection-only cohort still has
substantial regressions.

The recorded-calibration replay reproduces stored point labels, source
geometry and fusion exactly. Every original input hash and all 320
baseline viewer JPEGs match. Candidates add 640 projection rerenders and
82 NN relifting images, plus 41 baseline relifting images. Depth maps and
the chunk-wide masklet merge remain frozen to isolate the calibration's
effect on lifting, voting and proposal fitting; tracking and refinement
were not rerun.

## Is motion fusion introducing the cyan displacement?

No, for the three inspected boxes. Running `lifted.proposals()` with all
accepted motion-cluster members, then with **every motion cluster removed**,
produces identical proposal dictionaries for the SUV, red car and thin
right-side column. Each is a singleton lifted member in the fused output.
The orange proposal is a separate motion source.

The orange track is not well explained as a sliding patch of static scenery.
Track hint 42 has 82 instants over 4.94 s and moves 28.81 m. In a camera-free
check, its old returns were queried against complete later/earlier non-ground
sweeps, both at their original world locations and translated by the track's
measured displacement:

| Pairs with displacement >4 m | Original world positions | Translated with track |
|---|---:|---:|
| Median nearest-return distance, 15 pairs | 2.62 m | 0.259 m |
| Median fraction within 10 cm | 0% | 5.6% |

Sparse scanning and changing visible surfaces prevent exact point identity.
This supports genuine motion; it is not a rigidity or semantic annotation.
Nevertheless, the moving returns themselves project onto empty grass below
the visible SUV in six inspected images. Box fitting cannot explain a
misprojection of the raw member points.

## Calibration estimate independent of masks and LiDAR labels

`image_motion_calibration.py` matches SIFT image features, applies a free
essential-matrix RANSAC to remove inconsistent matches, and compares them to
the epipolar geometry predicted by ego motion at the two image timestamps.
Only a **single constant camera rotation** is fitted, with robust loss.
Recorded camera translation and intrinsics stay fixed. Neither mask pixels,
semantic labels, motion clusters, box centres, class sizes nor ground heights
are inputs to this fit.

For NN, 60 image pairs across chunks 0001–0010 yield 8,400 correspondences.
Fitting chunks 0001–0007 and withholding **all of chunks 0008–0010** gives:

- Local optical rotation vector: **[−2.6665°, +2.0208°, −0.0167°]**.
- Held-out pairs: 19.
- Median absolute epipolar residual: **1.242 → 0.532 px**.
- 90th percentile: **4.018 → 1.444 px**.
- Fitting alternate disjoint image-pair subsets gives optical x corrections
  −2.78° / −2.66° and y +1.86° / +2.18°. Roll is less stable.

The convention is `ego_T_cam_new.R = ego_T_cam.R @ Exp(rotvec)`;
optical x is right, y down, z forward. These are not vehicle-frame Euler angles.
This updates the physical projection model rather than moving individual boxes.
The bag's recorded camera TF rotations are ideal nominal ring orientations.

The same procedure was run for all eight cameras. The table reports models
fitted before chunk 0008 and evaluated only on chunks 0008–0010:

| Camera | Held-out pairs | Median epipolar error, recorded → experiment | Optical rotation vector x / y / z (degrees) |
|---|---:|---:|---|
| NN | 19 | 1.242 → 0.532 px | −2.667 / +2.021 / −0.017 |
| NE | 15 | 2.733 → 0.505 px | −2.907 / +1.520 / +0.081 |
| EE | 14 | 2.975 → 0.696 px | −0.568 / +2.640 / −2.075 |
| SE | 15 | 1.563 → 0.543 px | +1.821 / +1.869 / −0.098 |
| SS | 16 | 0.770 → 0.460 px | +1.383 / +0.782 / −0.010 |
| SW | 15 | 0.889 → 0.641 px | +1.400 / +1.112 / +1.165 |
| WW | 15 | 2.951 → 0.576 px | +0.921 / +0.044 / +2.140 |
| NW | 15 | 0.862 → 0.535 px | −2.456 / +1.426 / −0.123 |

**These are experimental estimates, not eight certified calibrations.**
Epipolar agreement depends on ego poses and exposure timestamps; it does not
independently certify camera translation or LiDAR extrinsics. Near-straight
motion provides weaker rotation constraints. A nuScenes scene 0061 control
also preferred a spurious-looking roll adjustment, with inconsistent held-out
median improvement, so this procedure alone must not automatically deploy
all fitted rotations. NN has an additional, independent LiDAR/image check
below; the other cameras have not received that check.

## Does the NN estimate recover the actual object upstream?

Track 42's moving returns were selected entirely by LiDAR preprocessing.
Its correspondence to the black SUV mask `9bb1d277-451` was inspected across
the image sequence. The masks follow the visible SUV, including its occlusion
by the signpost; they do not outline the grass or the foreground column.

Without fitting anything to that correspondence, applying the **image-motion
estimate above** changes mean point-in-mask containment on 14 interleaved
validation views from **0.7% to 89.2%**. The original count denominator is held
fixed, including points leaving the image. The other 15 views change from
0.6% to 87.3%. These are repeated views of one vehicle, not 29 independent
objects.

Next, the actual `_ChunkLifter.sweep()` and `lifted.proposals()` were replayed
on both frame-98 scanners with NN alone, before/after the independent rotation.
All masks, depth artifacts, world points, ground grid and fitting settings
were held fixed. Cross-camera merge membership was frozen to the existing
instance table; this is a controlled NN-stage experiment, not a full rerun.

| Right-hand car mask, NN50 | Recorded calibration | Independent NN estimate |
|---|---:|---:|
| Claimed returns from independently moving SUV | 0 | 16 |
| Claimed returns closer than 15 m | 38 | 7 |
| Centre-LiDAR claimed-point median depth | 10.22 m | 32.02 m |
| Fitted component footprint | 0.634 × 0.165 m | 3.651 × 1.171 m |
| Fitted height | 2.249 m | 0.839 m |

The latter height is still a partial observation, not a validated complete
vehicle height. The key result is **different, correct-depth point ownership**,
not the shorter box. The old 10 m foreground column is replaced by the
independently moving SUV's component. No thin-box rejection or size prior
was added.

For the near cars, the same replay removes the >20 m background claims:
SUV 65 → 0 and red car 37 → 0 across the two scanners. Their fitted observed
lengths change 2.976 → 4.267 m and 1.868 → 2.827 m. These dimensions are not
3D ground truth; they demonstrate that projection changes which surfaces
enter the existing fitter.

`nn50_replay_calibration.jpg` separates two effects: the top row changes only
projection of **unchanged stored boxes**; the bottom row replays actual NN-only
lifting and box fitting. The calibration experiment does not merely redraw
an already corrupted proposal at a better-looking pixel position.

## Why the earlier mask-based calibration audit underestimated the error

Its point/mask correspondences came from tracks and labels generated using
the original calibration. In NN50 that selects foreground points for the car
and omits the actual car's returns. A fitted rotation is rewarded for preserving
that wrong association. Holding these correspondences fixed prevents gaming
the denominator but does not make the associations correct.

Indeed, the independent NN rotation **worsens** the original 69-view held-out
track/mask proxy (median 0.421 → 0.088), despite recovering the independently
selected SUV returns and improving held-out image epipolar geometry. This
contradiction is reported rather than discarded. It prevents using the old
label-derived containment metric alone to accept or reject a physical
calibration, and warrants more independently identified objects before
replacing the bag calibration.

## Additional lifting and proposal limitations

- NN50's metric depth fit has only 1,238 / 4,320 inliers. Near-car mask depths
  are biased, and no per-pixel uncertainty reaches lifting. Background at
  22–25 m receives the red-car label where the pooled map is just beyond
  the hard 10 m trust cutoff. A threshold increase could hide this example
  while introducing other errors; it was not adopted.
- The visibility tests are one-sided occlusion tests. A foreground surface
  can be visible but still belong to a different object. Calibration errors
  therefore propagate directly into incorrect point ownership.
- Proposal generation boxes the retained observed component. Even a correct
  calibration does not make every observed extent a complete amodal car.
  A separate nuScenes GT control confirms losses at/before lifting and
  incomplete shape coverage; increasing clustering radius is not justified.
- Grounding creates the lower wireframe extension, but local current-sweep
  road fits agree with the ground grid to about 2 cm for the near cars.
  Raising ground by 0.55–0.59 m would conceal the symptom with an unsupported
  geometry change.

The bag-wide **2.24 m SLF median is not established as fixed**. A calibration
change requires depth refitting and regeneration of semantic labels, proposals,
tracks and refinement, followed by independent quality checks. Existing
calibration, masks, point labels and source proposals have not been overwritten.

## Frozen-input SLF trial does not validate a full-bag fix

The same five originally accepted moving-car tracks from the earlier audit
were refit with the all-camera experiment, retaining their original masks,
point sets, selected views, Phase 3 initialization and every optimizer/gate
setting. Only one of five candidate fits passes the existing acceptance
rules, versus five of five baselines. A lower median fitted height cannot
therefore be reported as a successful refinement result.

| Track | Height, original → experiment | Silhouette residual, original → experiment | Candidate accepted |
|---|---:|---:|---|
| 0005-122 | 2.590 → 1.433 m | 0.654 → 0.299 | no |
| 0009-589 | 2.465 → 1.451 m | 0.555 → 0.316 | no |
| 0009-178 | 2.769 → 1.309 m | 0.618 → 0.687 | yes |
| 0004-348 | 2.869 → 1.350 m | 0.491 → 0.943 | no |
| 0004-372 | 2.428 → 1.204 m | 0.529 → 0.856 | no |

The first two rejections are not evidence of worse absolute silhouettes:
their new residuals improve substantially, but their already better initial
residuals (0.384 / 0.400) leave only 0.085 / 0.084 improvement. Both miss the
unchanged **minimum gain of 0.15**. The last two also exceed the absolute
residual limit; the one accepted candidate has an implausible 3.73 m width.
No acceptance gate was relaxed. These mixed results require regeneration
of upstream labels/associations and further validation; they do not justify
applying all eight candidate rotations or claiming the 2.24 m height issue
resolved. Results: `slf_check_image_motion.json`.

## Implementation scope and a platform-agnostic fix

Implementation update: the skew/inverse-ray and missing lens-model gaps below
are now fixed through one shared `CameraCalibration.model` used by depth
anchors, lifting, SLF and overlays. Unsupported image/calibration combinations
fail explicitly. SLF also now samples the nearest mask pixel, preserves
precision at large map coordinates, starts from Phase 3's smoothed positions
and exports exactly the positions its quality gates evaluated. No calibration
values, size priors or acceptance thresholds changed. Contracts and reference
checks: `docs/camera_geometry_contract.md`; paper comparison:
`docs/research/slf_projection_implementation_review.md`.

The three-track recorded-calibration replay in
`data/artifacts/slf_camera_fix/wato_refit.json` still gives heights 2.611,
2.459 and 2.758 m (previous stored values 2.590, 2.465, 2.769 m). All three
pass the existing gates. This is evidence that the implementation fixes alone
do not resolve the WATO height issue; it is not a full-bag rerun.

The calibration experiment is not a certification that every projection and
SLF path is implementation-complete. The initial inspection below records
the issues that motivated these fixes and the remaining constraints:

- The checked WATO rectified-pinhole lifting path uses the intended transform
  order and image-time pose. The earlier multiple-proposal-times viewer bug
  was real and was fixed separately.
- Before the shared-model fix, `wato_label_refinement.views.project()` used a separate simplified
  projection. It uses fx/fy/cx/cy and ignores intrinsic skew, whereas the
  shared forward projector multiplies by the full K. A direct synthetic
  comparison gives 0 px disagreement for zero skew and **6 px** when
  K[0,1] is 40 px. SLF ray construction likewise assumes zero skew. WATO's
  recorded skew is zero, so this does not explain the WATO result, but it
  is a demonstrated portability gap. The shared `lift_pixel_to_world()`
  also uses the simplified inverse and needs the same contract review.
- Before the shared-model fix, the forward projector and SLF did not model lens distortion.
  Rectified WATO images satisfy that assumption; accepting raw/distorted
  or other camera models requires explicit normalization/model support or
  rejection, rather than silently using their K alone.
- Current SLF optimizes track size and a shared horizontal offset, while
  heading and bottom height are inherited from Phase 3. It is therefore a
  constrained refinement, not a solver for erroneous calibration or
  arbitrary pose/ground errors. Size can compensate for inconsistent cues;
  a lower silhouette loss alone does not prove correct 3D dimensions.
- One-sided visibility is not an object-ownership test. The depth trust cutoff
  and lack of propagated depth uncertainty are additional implementation/design
  limitations. A fixed range or box-size adjustment is not a general solution.
- The minimum silhouette-gain gate can reject an absolutely better fit when
  its initialization is already good. The measured rejected fits and the
  accepted excessively wide fit justify reviewing acceptance criteria;
  they do not justify lowering a threshold to accept this bag's results.

The general remedy is a single tested camera-model projection/unprojection
contract shared by depth anchoring, lifting, SLF and viewers; independently
validated sensor transforms, exposure timing and ego poses supplied as data
per sensor setup; ownership/uncertainty checks; and SLF pose/shape constraints
validated against known geometry. Unsupported models and underconstrained
calibration estimates should produce an explicit failure or low-confidence
result. A measured rig-specific calibration belongs in input artifacts;
this bag's fitted rotation does not belong in generic algorithm defaults.

Validation needs independent synthetic end-to-end geometry tests through
`views.problem()` as well as the optimizer, followed by held-out objects and
annotated data from multiple sensor setups. The new adapter tests exercise
`views.problem()` with skew, nonidentity camera/ego poses, Brown/rational
distortion and fisheye. Independent reference observations validate forward
and inverse geometry. Placement and map-translation regressions cover the
optimizer/export handoff and numeric precision. These verify supported
geometry, not arbitrary pose errors or physical calibration accuracy.
Full depth, instance-merge, lifting, proposal, tracking and SLF regeneration
is still required before claiming the WATO height problem fixed.

## Reproduce

Diagnostics and isolated calibration candidates are under
`data/artifacts/root_cause_audit/`. From the repository root:

```bash
export ARTIFACT_ROOT_URI="file://$PWD/data/artifacts"
export PYTHONPATH=src/common/src:src/semantic_lifting/src:src/proposal_generation/src
export OPENBLAS_NUM_THREADS=1
venv/bin/python data/artifacts/root_cause_audit/trace_wato_frame.py
venv/bin/python data/artifacts/root_cause_audit/image_motion_calibration.py
venv/bin/python data/artifacts/root_cause_audit/independent_motion_projection.py \
  --rotation-report data/artifacts/root_cause_audit/image_motion_calibration.json
venv/bin/python data/artifacts/root_cause_audit/replay_nn50_calibration.py \
  --check-calibration recorded
# FAIL: the car-mask proposal contains no independently moving SUV returns.
venv/bin/python data/artifacts/root_cause_audit/replay_nn50_calibration.py \
  --check-calibration image_motion_candidate
# PASS: the fitted component recovers independently selected SUV returns.
```

`image_motion_calibration.py --camera PANO_NE --pairs 40` selects another
camera. It uses the installed OpenCV SIFT implementation and caches numeric
feature correspondences. `motion_world_consistency.py` checks world motion;
`oracle_stage_control.py` runs the nuScenes GT stage control. Diagnostic
NPZs contain numeric arrays and JSON metadata, not pickled Python objects.

`calibration_candidate_NN_image_motion.json` is the isolated NN hypothesis;
`calibration_candidate_all_image_motion.json` contains the less validated
all-camera experiment. Neither is the bag's active calibration.

Validation: the synthetic check recovers an independent [1.4°, −1.1°, +0.7°]
rotation from 1,800 generated correspondences using the diagnostic's actual
epipolar residual, to numerical precision. The real NN50 ownership check
fails for recorded calibration and passes for the candidate (16 independently
selected SUV returns in the fitted component). Proposal-generation and
semantic-lifting tests: **87 passed**. Diagnostic Ruff checks and
`git diff --check` pass.
