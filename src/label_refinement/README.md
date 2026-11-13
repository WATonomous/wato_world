# label_refinement

Per-track refinement of tracking's tracks: one box per parked vehicle from its
points over every frame, one size, a corner anchor and a smoothed path per
moving track (Phase 3, training-free), then SLF: a box shape fit to the
track's camera masks and LiDAR points, which replaces the size where it
explains the masks better. SLF runs in PyTorch on the GPU (a CPU path for
tests and GPU-less hosts); a learned refiner (LabelFormer) and a car mesh
prior for SLF come later. Measurements: `docs/label_refinement_measurements.md`.

## How a track is refined

What a track is comes from tracking (`track_summaries.parquet`): parked,
moving or unknown, and its class. A track linked across chunks is refined
whole, over one frame per instant (`tracks.per_instant`: the owned frame where
both chunks of a shared span saw it), each frame against its own chunk's world
sweeps, poses and ground grid (a parked box: its chunk with the most frames).

- **Parked vehicle** (car, truck, bus; 3D Auto Labeling's static path). Every
  frame sees the car from another side, and it doesn't move, so its points over
  all frames are one cloud. They are gathered from each frame's non-ground
  points inside the rectangle enclosing the track's boxes (those centred within
  1 m of their median centre, grown by 0.3 m), deduplicated on a 5 cm grid,
  and the largest group chained within 0.5 m is the car. One box is fit to it,
  lowered to the ground, and used on every frame. With fewer than 30 points it
  is refined like a moving track.
- **Every other track** (moving, unknown, parked pedestrians and two-wheelers:
  a standing person still moves).
  - One size: the 90th percentile of l, w, h over its denser half of frames
    (far frames see one face, so their boxes are small); the median for
    pedestrians, bicycles and motorcycles, whose high percentile is bleed.
  - Heading from the velocity of a local linear fit over 1 s where the track
    moves ≥ 0.5 m/s and that speed is ≥ 3 standard errors of the fit's
    velocity, which also gives it a front (`heading_directed`); otherwise the
    mean of its denser frames' headings, mod π. The second test rejects a
    parked car's jitter: as its visible faces change, its box centre moves
    back and forth faster than 0.5 m/s, giving headings tens of degrees off.
  - Each frame's box is placed on the observed box's corner nearest the ego
    and extends away from it (OYSTER, CPD); along an axis the observed box
    already spans at full size, its centre is kept. A pedestrian's box keeps
    its observed centre: a person's returns surround it.
  - Centres are smoothed by the same 1 s local linear fit;
    `residual_smoothness` is the RMS distance it moved them. A mover's points
    are not stacked: 3DAL found aligning them by noisy boxes hurt.

Every row of a track has one size. `confidence` is the median score of the
track's proposals, an uncalibrated ranking until labels calibrate it.

## SLF: a shape fit per track

An adaptation of Segment, Lift and Fit (Li et al., ECCV 2024), with a rounded box for the shape, on
tracks of `slf.classes` (all six classes):

- **Views.** The images in which one of the track's masklets has a mask, at
  the camera's pose at the image's own time; up to 16 spread over the track's
  life (the largest projected box in each stretch), at least 3. Each is a
  48 × 48 grid of rays over the box's and the mask's bounding rectangle. Pixels
  of a nearer tracked object's mask are left out (the track may continue
  behind it); a mover's box moves by its velocity to the image's time.
  Projection and inverse rays use the shared full-intrinsic/lens model:
  [camera contract](../../docs/camera_geometry_contract.md).
- **Shape and pose.** One length, width and height for the track. Each
  frame starts at Phase 3's smoothed position and follows size changes from a vehicle's
  near corner (the far faces are what the LiDAR doesn't see and the
  silhouette does), a parked box's or a pedestrian's centre. One small offset
  along the box's axes is shared by the track; the bottom is Phase 3's.
- **Heading.** Phase 3's plus a fitted correction (≤ 30°), for cars, trucks,
  buses and two-wheelers: with the heading fixed, a box turned 10° fits the
  silhouette only by growing wider. A parked vehicle's one box gets one
  angle; any other track a correction linear in time between knots ≤ 2 s
  apart. Weak priors pull each knot to Phase 3's heading and neighbouring
  knots together. The near-corner anchors' extents follow the corrected
  axes; which end is nearer the ego stays Phase 3's, or the box would jump
  sideways as the heading crossed it. A pedestrian's near-square footprint
  says nothing about its heading, which stays Phase 3's.
  Fitting uses a local origin per track before float32 conversion, so large
  map coordinates retain centimetre precision.
- **Fit.** Adam over log size and the offset, 150 steps at lr 0.1 (SLF's),
  and the heading knots at 0.02 rad, 32 tracks per fit, each on its own loss
  and keeping its lowest-loss step (Adam at a fixed rate doesn't settle) (the GPU's cost is launching each
  step, so a batch takes about as long as one track), minimising: Dice between each view's soft silhouette (the SDF's minimum
  along each ray) and its mask; a robust |SDF| at the LiDAR returns on the
  observed boxes (they lie on a vehicle's surface; a pedestrian's need only be
  inside its box); a weak pull to the class's mean size (nuScenes', as
  CenterPoint's anchors) and of the offset to zero.
- **Gates.** The fit replaces Phase 3's geometry (`size_source` `slf`, the
  exact fitted size and evaluated poses for every frame) only if it lowers the silhouette
  residual (mean 1 − IoU over the views) by ≥ 0.15: where Phase 3's box
  already matches the masks the fit mostly shrinks it to masks drawn tight. A
  pedestrian's fit is kept whatever the gain: its masks are a few pixels wide
  and its returns a sliver, and the prior gives the depth and width neither
  shows. Every kept fit needs a silhouette residual ≤ 0.8 and a median |SDF| ≤
  0.3 m. `residual_silhouette` and `residual_lidar_fit` are the written box's
  on the views, kept or not; null where SLF didn't run.

Accepted positions and headings are written directly: another smoother or
ground lookup would change the geometry the quality gates evaluated. The
bottom remains Phase 3's constraint. The learned shape space, a free
per-frame pose and the first-intersection LiDAR term of the paper are not
implemented; the
[primary-source review](../../docs/research/slf_projection_implementation_review.md)
separates these adaptations from implementation bugs.

## Run

```bash
./watod run label_refinement --bag <bag_id> [--force] \
    [--config /ws/src/label_refinement/config/label_refinement.wato.yaml]
python /ws/src/label_refinement/scripts/eval_refined.py <bag_id> [--canonical-lidar lidar_cc]
```

tracking must have run. The bag is skipped if `refined_labels.parquet` exists
unless `--force`. Each chunk gets `manifest_label_refinement.json`.
`eval_refined.py` scores the owned refined boxes and the track boxes they came
from against `gt_boxes.parquet`, side by side, each labelled instant once, in
the chunk that owns it.

## Inputs and output

Reads `tracks.parquet` + `track_summaries.parquet`
(`wato_common.artifacts.tracks`), the world sweeps
(`wato_common.artifacts.lidar_sweeps`), `ground.npz` and `poses.parquet`, and
for SLF perception_2d's masklets and masks (`wato_common.artifacts.masklets`),
`frame_index.parquet` (`FrameIndex`) and `calibration.json`.
The config's `frames.canonical_lidar` must be the one lidar_preprocessing ran
with (`lidar_cc` in `.wato.yaml`): a track frame's points are its frame's
sweeps.

Writes bag-level `refined_labels.parquet` through
`wato_common.artifacts.refined_labels`: one `RefinedTrackRow` per
tracks.parquet row, with `owned` carried from it; the two rows of an instant
both chunks saw get the same box. `size_source` is `aggregate` (parked, from
its points), `percentile` or `slf`.

Configuration: `config/label_refinement.yaml` and `.wato.yaml` (the canonical
LiDAR only; a test fails on any other drift); every field is documented in
[`config.py`](src/wato_label_refinement/config.py). `slf.enabled: false` gives
Phase 3 alone; `slf.device` picks the GPU (`auto`) or the CPU.

## Camera calibration source

`calibration_source: recorded` is the default. `validated` requests the
camera-calibration stage's accepted result and fails on missing, stale,
unqualified or failed-camera evidence. Keep this choice consistent across
camera-dependent stages. Output/cache identities reject mixing calibrations;
rerun depth and downstream processing with `--force` after a change. See
[the calibration module](../camera_calibration/README.md).
