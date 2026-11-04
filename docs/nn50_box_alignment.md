# Exact screenshot diagnosis: chunk 0005, PANO_NN image 50

This is the supplied image with the red Honda ahead, black SUV to the left,
and a thin cyan "car" beside the signpost. It is the existing file
`raw/sensors_2026_07_07_23_01_24_2_with_eidos_mcap/chunks/0005/viz/proposals_cam_sources/PANO_NN/000050.jpg`.

**Updated root-cause finding:** an independently estimated NN camera rotation
recovers actual moving-SUV returns instead of the foreground column, using
unchanged masks and the actual lifting/fitting code. See
[stage isolation and independent calibration evidence](wato_alignment_root_cause.md).
The measurements below explain the original wireframe construction; they do
not establish partial shape coverage as its sole upstream cause. The viewer
has only one proposal time here, and ground-height adjustments are unsupported.

## Exact reproduction

There is one paired proposal frame, reference sweep **98**, **12.979 ms
before** the image. Rebuilding both near-car proposals from their existing
point labels, connected components, ground grid and configuration reproduces
the stored centres and heights exactly. This rules out stale proposal
geometry or a mismatched sweep/index in this replay.

| Near object | Fitted returns | Inside its own visible mask | Visible-mask top / bottom | Projected proposal top / bottom |
|---|---:|---:|---|---|
| Black SUV, `009ba003-901` | 287 | 93.7% | 546 / 718 px | 593 / 794 px |
| Red car, `102d7f93-d7a` | 173 | 100% | 584 / 783 px | 634 / 868 px |

The proposal tops are **47–50 px below the visible roofs**. Their bottoms
extend **76–85 px below the masks**. This is not evidence of a single rigid
pixel translation: the fitted point sets omit upper visible shape, and the
box-building step separately extends the lower bounds.

## Which operation creates the lower extension?

`lifted.proposals()` keeps the largest connected group of claimed returns,
fits `fit_bev_box()`, then calls `to_ground()`. The latter preserves the top
and BEV footprint, lowers the bottom to the grid under the centre, and
recomputes height and centre z.

| Near object | Raw point z extent | Ground-extended height | Bottom lowering | Projected bottom, raw bounds → final |
|---|---:|---:|---:|---|
| Black SUV | 1.096 m | 1.650 m | 0.553 m | 731 → 794 px |
| Red car | 0.809 m | 1.403 m | 0.594 m | 778 → 868 px |

The red car's raw bounds already end near its visible lower edge; the ground
extension introduces about **90 px** of additional projected downward extent.
The SUV's extension adds **63 px**. `box_construction.png` holds the image,
calibration and point membership fixed and shows observations, raw bounds,
and the final proposal side by side.

**Removing ground extension is a diagnostic, not a validated fix.** It would
leave boxes above the road and would not restore their missing upper shape.
A cuboid's nearest ground corner also need not coincide with a visible wheel:
body overhang, the box footprint's empty corners and occlusion can make a
correct amodal cuboid extend below a foreground mask. Mask-edge mismatch
alone is not proof that the ground grid or projection is wrong.

## Is the ground grid simply too low?

Not according to the LiDAR evidence in this frame. The grid agrees with
nearby aggregate ground returns. Separately, robust planes fit to the
**current** sweeps' Patchwork ground within 3 m of each box centre differ
from the grid by:

| Scanner | Under SUV | Under red car |
|---|---:|---:|
| lidar_cc | +1.35 cm | −1.34 cm |
| lidar_ne | −2.03 cm | −0.36 cm |

These differences do not support raising the ground grid by the 0.55–0.59 m
needed to remove the extension. This check is internal LiDAR consistency;
it does not independently validate the camera extrinsics or the road under
every unobserved part of a vehicle.

## Many returns still provide incomplete shape coverage

The retained components mostly cover lower/rear body surfaces. In a 1 m
neighbourhood of each observed box, the original raw returns projecting into
the car mask are essentially all preserved by lifting. None of those
near-car returns is marked ground. There is no wholesale roof-point loss
from Patchwork++ or the lifting visibility stage in this replay.

Some labelled returns do project onto apparent roof pixels but have a
completely different depth. On the red car, body returns are around 7–9 m;
roof-band background claims are around 22–25 m. Including those to fill the
mask would put another surface into the car. The largest-group filter
correctly avoids that kind of completion.

A radius sweep illustrates why a generic parameter change is insufficient:

- The red car keeps the same 173 returns and the same displaced-looking
  bounds for radii from **1 to 4 m**.
- Raising the SUV's radius from 2 to 2.5 m adds seven returns and moves its
  projected top from 593 to 561 px, but increases its inferred width from
  **1.68 to 2.75 m**. At 4 m its length reaches **16.2 m**. Better overlap
  alone would reward contamination.

The relevant failure is **coverage and ownership**, not merely the count of
points. Hundreds of returns from one observed portion do not identify the
full object's roof, footprint or centre.

## The signpost-area "car" is another failure mode

`lifted-9bb1d277-451-000098` has 42 claimed returns. Its observed group is
only **0.634 × 0.165 m** in BEV and **0.593 m** in raw height; grounding adds
**1.656 m**, producing a **2.249 m** vertical box. Its association with a
car mask does not make this thin near-surface group a full car. This is a point-ownership problem. The later independent camera-rotation
experiment does change its identity: actual moving-SUV returns replace the
foreground component. The original inference that camera calibration could
not resolve it was too strong.

## Consequence for a fix

The initial box-construction experiments did not determine the physical
calibration. The later independent image/ego-motion check and actual lifting
replay localize a substantial error upstream, in NN camera orientation and
therefore point ownership. See the linked root-cause report for the measured
rotation, held-out checks, limitations and isolated candidate. Shape completion
remains a separate limitation after projection and ownership are corrected.

## Artifacts and replay

Diagnostics live in `data/artifacts/box_alignment_audit/nn50/`:

- `box_construction.png` / `.json`: exact observations → raw bounds → grounding.
- `point_support.png` / `.json`: raw, ground and own-label returns.
- `probe.json`: stored versus rebuilt proposals and per-object measurements.
- `local_road_check.json`: current-sweep road fits.
- `radius_sweep.json`: isolated clustering-radius experiment.
- `road_intensity_rotations.png`, `reflectivity_height_check.png`: exploratory
  independent-feature overlays, not fitted or accepted calibration changes.

From the repository root:

```bash
export ARTIFACT_ROOT_URI="file://$PWD/data/artifacts"
export PYTHONPATH=src/common/src:src/proposal_generation/src:src/semantic_lifting/src
export OPENBLAS_NUM_THREADS=1
venv/bin/python data/artifacts/box_alignment_audit/nn50/probe.py
```

The last command deliberately fails when a near-car proposal extends more
than 35 px below its own visible mask. It reproduces the screenshot symptom;
that threshold is a diagnostic signal, not a proposed 3D-label acceptance
criterion. No production calibration, ground grid, point labels or proposals
were altered for this analysis.
