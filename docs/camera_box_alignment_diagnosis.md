# Why proposal boxes look displaced in WATO camera images

**Updated:** [independent calibration and stage-isolation results](wato_alignment_root_cause.md)
identify an NN orientation error that changes lifting ownership in the supplied
screenshot. The measurements below preceded that check and must not be read
as ruling out calibration or attributing the screenshot solely to partial geometry.

Measured 2026-11-03 on
`sensors_2026_07_07_23_01_24_2_with_eidos_mcap`, chunks 0000, 0004 and 0009.
The reported symptom is boxes shifted away from the visible object in camera
images. The subsequently supplied exact screenshot is chunk 0005 / PANO_NN
image 50. Its near cars have 287/173 returns and only one proposal time:
[the exact-frame diagnosis](nn50_box_alignment.md) traces their low fitted
tops and ground-extended bottoms. The time-mixing and sparse-count findings
below describe separate measured phenomena, not a complete explanation of
that particular image. Two issues need separating: an actual camera-viewer bug and the
limited object geometry available to early lifted proposals.

**The concrete viewer bug was mixing proposal times.** It is fixed in
`wato_proposal_generation/viz/camera.py`. **The underlying lifted boxes also
frequently describe a small set of observed returns rather than the complete
car.** Calibration tuning alone cannot recover that missing geometry or
correct a return assigned to the wrong object. The measurements do not identify
one physical calibration error as the cause of every remaining offset.

## A camera image was showing more than one LiDAR instant

Ingest pairs each LiDAR sweep with its nearest camera image. The inverse
mapping has several sweeps per image. On WATO, those sweeps can belong to two
canonical proposal frames: the centre sweep nearest an image and the corner
sweep nearest it need not belong to the same frame-sync tick.

The old `_boxes_for()` collected **all** those frames' proposals and drew
them together on the image. Using the ego pose at the image time is correct,
but does not move a moving object from one timestamp to another. One object
therefore appeared at multiple positions on the same image.

A deterministic replay reached the real selection function:

```text
PANO_NN, chunk 0000, image 11
before: 14 fused boxes from two proposal instants, −46.234 and +4.307 ms
fixed:   7 fused boxes from the single nearest instant, +4.307 ms
```

Across all cameras with trusted image poses:

| Chunk | Images paired to a proposal frame | Images mixing proposal instants before | After |
|---|---:|---:|---:|
| 0000 | 5,068 | 2,161 | 0 |
| 0004 | 5,434 | 2,232 | 0 |
| 0009 | 5,437 | 2,149 | 0 |
| Total | 15,939 | 6,542 (41.0%) | 0 |

Sampling every tenth image, as the viewer does, produced 1,742 repeated
visible tracked-object appearances from multiple times. Their projected
box-centre separations were median **5.4 px**, p90 **21.8 px**, p99 **89.1 px**.
These are tracked associations, not independently annotated object identities.
The explicit examples in `overlay_before_after.png` are large-separation
moving-car cases with locally consistent track motion, not typical-case
estimates. The image and calibration are identical before and after selection.

The fix selects the nearest canonical reference timestamp **among the paired
frames**. Earlier time wins a tie. A nearest frame with no proposals stays
empty; it does not fall back to an older populated frame. No proposal,
calibration, mask or point-label artifact is changed.

Existing generated JPEGs retain their old drawings until regenerated. A new
set of 320 overlays from the actual fixed `write_camera()` is saved under
`data/artifacts/box_alignment_audit/fixed_camera/`, every fiftieth image of
all cameras in the three measured chunks.

## Object motion is a residual, not the dominant aggregate improvement here

Both lifting and the proposal camera viewer compensate ego motion using the
image's pose, but leave dynamic-object motion uncompensated. A return also
has its own firing time within the sweep, so moving a box by velocity times
the image-minus-reference timestamp is only an approximation.

For a controlled check, the nearest frame was fixed first, then the same
box was translated by a local track velocity times the image/reference time
difference. Velocity was fit over ±0.3 s, requiring at least five instants,
0.15 s of span, centre-fit RMS ≤0.15 m, and speed ≤30 m/s. Boxes needed at
least 30 returns and an associated car mask. Size, heading, calibration and
image were held fixed. This is a diagnostic using downstream tracks; it is
not added as a dependency of proposal generation.

| Cohort | Views | Median projected box-rectangle / mask-rectangle IoU, wrong sign / none / correct sign |
|---|---:|---:|
| All eligible | 654 | 0.618 / 0.620 / 0.623 |
| World speed >2 m/s | 179 | 0.597 / 0.614 / 0.614 |

The sign matters in some cases, but the typical aggregate change is small.
This does not justify treating an arbitrary camera timestamp shift as the
main fix. It also does not rule out motion error on fast or very near objects.

## The early lifted box is a box around observed surfaces

Semantic lifting writes point labels, not boxes. Proposal generation takes
each lifted instance's claimed returns, keeps its largest radius-connected
group, applies the car height cap, fits a minimum-area BEV rectangle and z
extent, then extends the bottom down to the ground grid. There is no full
object-shape completion in this source. A sparse side, roof strip or edge
can become the whole proposal, with its centre on that observed fragment.
A nearby surface incorrectly claimed through the mask can be boxed instead.

After fixing frame selection, 3,761 lifted-car camera views with a supporting
mask were measured, every tenth image of the same chunks. Boxes are clipped
to the image; the comparison is the projected 3D box's bounding rectangle
against the visible mask's bounding rectangle.

| LiDAR returns used by box | Views | Median l × w × h (m) | Median rectangle IoU | Median normalized centre offset |
|---|---:|---|---:|---:|
| 5–19 | 1,798 | 1.54 × 0.38 × 1.16 | 0.248 | 0.403 |
| 20–49 | 739 | 3.49 × 1.13 × 1.57 | 0.426 | 0.272 |
| 50+ | 1,224 | 3.70 × 1.55 × 1.50 | 0.622 | 0.191 |

Offset is the norm of the projected-rectangle centre minus mask centroid,
normalized separately by mask width and height. It exceeds 0.5 for **39.3%**
of the 5–19-point views versus **8.4%** of the 50+-point views. Point count,
range, visibility, mask quality and occlusion are confounded; these are proxy
measurements, not an estimate of the fraction of wrong 3D boxes.

The examples in `sparse_box_examples.png` replay the actual selected point
sets; reconstructed point counts match the stored proposals. They show why
moving a camera rotation cannot turn seven returns into the missing shape
of a car. Some may also have the wrong depth/surface association.

A 3D cuboid legitimately projects beyond a visible object mask: hidden body
parts and perspective are not represented by the foreground mask. In
particular, extending a roof-only observation down to the ground can draw
below the visible mask of an occluded car. Such a picture alone does **not**
prove a projection error or a bad ground grid. These proxies cannot replace
3D ground truth.

## Visibility does not establish object ownership

Lifting checks whether a point is visible and inside a mask. Its depth tests
reject returns **behind** a nearer surface. A foreground return landing on a
background car's mask edge is itself the nearest surface, so it can pass.
Neither the LiDAR occluder nor the one-sided depth test establishes that the
return belongs to that car. The dense depth map only occludes up to 10 m,
because its fitted depths become unreliable at longer ranges.

Among 1,024 measured masks whose median depth-map value is 5–20 m, 10.7% of
box centres are closer than 0.65 times that depth. This is consistent with
some foreground associations, but is not an independent correctness verdict.
Farther masks are not used for this inference: their depth fits can be badly
wrong. The existing occlusion investigation in
`proposal_generation_measurements.md` documents examples of hidden objects
boxed on their occluders. Increasing the clustering radius or changing a
camera yaw cannot resolve that ownership ambiguity.

## What the evidence supports

1. **Fixed:** the camera viewer mixes time samples. It now selects one nearest
   proposal frame. Regression tests reproduce the actual writer call, including
   scanners belonging to different ticks, timestamp ties and empty frames.
2. **Main structural limitation of the lifted source:** sparse/partial or
   incorrectly associated returns are treated as complete object geometry.
   The mechanism is explicit in the source and the sparse-box dimensions.
3. **Residual motion and calibration:** both can contribute. The controlled
   motion experiment gives a small aggregate improvement, and the prior
   calibration sweep improved the original containment check only from 69.8%
   to 72.5% (`camera_projection_measurements.md`). Neither experiment proves
   them absent or identifies a rig-wide calibration correction.

The next geometry work should evaluate track-consistent shape completion
anchored to the observed near surface, and object-depth/ownership checks for
weak lifted instances, against labelled WATO examples. Using class dimensions
or a far-range depth threshold without that evaluation can simply replace
one biased box with another. SLF's 2.24 m car-height issue is downstream and
is not fixed by the viewer change.

## Replays and validation

All diagnostic scripts and JSON results are under
`data/artifacts/box_alignment_audit/`. From the repository root:

```bash
export ARTIFACT_ROOT_URI="file://$PWD/data/artifacts"
export PYTHONPATH=src/common/src:src/proposal_generation/src:src/semantic_lifting/src
export OPENBLAS_NUM_THREADS=1
venv/bin/python data/artifacts/box_alignment_audit/reproduce_overlay.py
venv/bin/python data/artifacts/box_alignment_audit/verify_selection.py
venv/bin/python data/artifacts/box_alignment_audit/analyze_overlay.py
venv/bin/python data/artifacts/box_alignment_audit/analyze_geometry.py
```

`analyze_overlay.py` explicitly retains the historical selector to reproduce
the old comparisons after the fix. `plot_examples.py` and
`plot_sparse_geometry.py` produce the figures (matplotlib required).
`render_fixed_camera.py` remaps Docker image paths in memory for host replay
and calls the real writer; source artifacts are untouched.

Proposal-generation and semantic-lifting tests: **87 passed**. The regression
suite failed on the original viewer and passed with nearest-frame selection.
The real-bag replay also changed from two instants to one, and the full
15,939-image selection check passes.
