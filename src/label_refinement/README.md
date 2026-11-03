# label_refinement

Training-free refinement of tracking's tracks: one box per parked vehicle from
its points over every frame, and one size, a corner anchor and a smoothed path
per moving track. CPU only. SLF (fitting a shape to the camera masks per
track) and a learned refiner (LabelFormer) come later. Measurements:
`docs/label_refinement_measurements.md`.

## How a track is refined

What a track is comes from tracking (`track_summaries.parquet`): parked,
moving or unknown, and its class.

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
    moves ≥ 0.5 m/s, which also gives it a front (`heading_directed`);
    slower, the mean of its denser frames' headings, mod π.
  - Each frame's box is placed on the observed box's corner nearest the ego
    and extends away from it (OYSTER, CPD); along an axis the observed box
    already spans at full size, its centre is kept.
  - Centres are smoothed by the same 1 s local linear fit;
    `residual_smoothness` is the RMS distance it moved them. A mover's points
    are not stacked: 3DAL found aligning them by noisy boxes hurt.

Every row of a track has one size. `confidence` is the median score of the
track's proposals, an uncalibrated ranking until labels calibrate it.

## Run

```bash
./watod run label_refinement --bag <bag_id> [--force] \
    [--config /ws/src/label_refinement/config/label_refinement.wato.yaml]
python /ws/src/label_refinement/scripts/eval_refined.py <bag_id> [--canonical-lidar lidar_cc]
```

tracking must have run. The bag is skipped if `refined_labels.parquet` exists
unless `--force`. Each chunk gets `manifest_label_refinement.json`.
`eval_refined.py` scores the owned refined boxes and the track boxes they came
from against `gt_boxes.parquet`, side by side.

## Inputs and output

Reads `tracks.parquet` + `track_summaries.parquet`
(`wato_common.artifacts.tracks`), the world sweeps
(`wato_common.artifacts.lidar_sweeps`), `ground.npz` and `poses.parquet`.
The config's `frames.canonical_lidar` must be the one lidar_preprocessing ran
with (`lidar_cc` in `.wato.yaml`): a track frame's points are its frame's
sweeps.

Writes bag-level `refined_labels.parquet` through
`wato_common.artifacts.refined_labels`: one `RefinedTrackRow` per (track,
frame), with `owned` carried from tracks.parquet. `size_source` is
`aggregate` (parked, from its points) or `percentile`.

Configuration: `config/label_refinement.yaml` and `.wato.yaml` (the canonical
LiDAR only; a test fails on any other drift); every field is documented in
[`config.py`](src/wato_label_refinement/config.py).
