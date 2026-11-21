# Background points receiving car masklet labels

Investigated 2026-11-09 using the saved July 7 WATO bag. The strongest
explanation in the reproduced scene is camera orientation error, compounded
by a depth visibility test that cannot reliably reject the wrong surface.
No production calibration, configuration, masks or labels were changed.

## Real-scene reproduction

Bag `sensors_2026_07_07_23_01_24_2_with_eidos_mcap`, chunk `0005`,
`PANO_NN` image `50`, paired sweeps `98` (`lidar_cc`) and `99` (`lidar_ne`).
The current calibration SHA256 is
`83afb0469640f976c2f9136d48f43739b6912200b58dfcd312d2fd2d2e862acf`;
it matches the calibration input recorded by this chunk's lifting manifest.
All eight camera orientations remain the nominal ring rotations: roll
−90°, pitch 0°, and yaw in multiples of 45°.

Replayed the production `_ChunkLifter.sweep()` with the saved world points,
image-time ego pose, masks, depth and default config. Then changed one input
at a time:

- **Rotation only:** right-multiply NN's `ego_T_cam` rotation by
  `Exp(deg2rad([-2.667, +2.021, -0.017]))`, in optical coordinates. This is
  the experimental independent estimate already recorded in
  [the earlier measurements](semantic_lifting_measurements.md#wato-camera-alignment-july-7-bag-2026-11-03),
  not a newly fitted or certified calibration. Depth remains frozen.
- **Depth range only:** use the existing metric depth map at all finite
  ranges instead of only below 10 m. Projection remains unchanged.

Counts below are point-mask claims summed over the two sweeps. They are not
unique physical points across scanners. Depths are optical z, in metres.

| Target | Recorded calibration | Rotation only | Depth range only |
|---|---|---|---|
| Left nearby car, `ecd7eb90-22d` | 65 claims beyond 20 m; 278 below 15 m | 0 beyond 20 m; 364 below 15 m | 0 beyond 20 m; 276 below 15 m |
| Front nearby car, `22d8682c-09a` | 37 claims beyond 20 m; 174 below 15 m | 0 beyond 20 m; 240 below 15 m | 0 beyond 20 m; 174 below 15 m |
| Farther centre car, `bbc842dd-43c` | Median claimed z 39.37; p10–p90 38.12–48.60 | Median 22.85; p10–p90 22.42–24.76 | Same as recorded |

The image overlay shows the near cars' scan lines below their silhouettes
under the recorded calibration, with distant returns taking their roof
pixels. The experimental rotation aligns the lines with the cars and removes
the distant claims. It also shifts the farther car from background surfaces
to a coherent nearer surface. This reproduces the earlier documented result
on current artifacts. The >20 m threshold identifies background for the two
near cars only; it is not an error criterion for the farther car.

## Where the failure enters

1. `semantic_lifting/projection.py:25` computes
   `inv(ego_T_cam) @ inv(world_T_ego_at_image_time)`. This order is correct
   for the world-registered points. The pipeline passes the stored image's
   camera model; rectified cameras use their rectified intrinsics. The
   reproduction supports a problem with the supplied physical alignment,
   rather than an inverted transform or an unused extrinsic.
2. `semantic_lifting/pipeline.py:180` applies visibility before handing every
   surviving point to its covering masks. It uses actual mask pixels, not
   filled bounding rectangles.
3. `semantic_lifting/visibility.py:93–94` deliberately bypasses the depth-map
   check whenever the pooled map depth is **at least 10 m**. The threshold
   is on the predicted occluding surface, not on the candidate point's range.
   Every one of the 65 + 37 + 95 background claims examined in these three
   masks bypasses that check.
4. The remaining LiDAR test uses only the current sweep's non-ground returns,
   in 8 px cells pooled over a 3 × 3 cell window. If no nearer return falls
   there, a background return can serve as its own nearest surface and pass.
   Even with a nearer return, the allowed difference is `0.5 + 0.05*z` m.
   The test establishes absence of a local occluder; it does not establish
   that the point belongs to the object in the image.

The map is loaded and indexed; it is neither absent nor globally unused.
This frame reports fit status `OK`, 4,320 anchors, 1,238 inliers, and
inlier RMSE 0.215 m. That inlier statistic does not establish accuracy over
the car masks:

| Mask | Median saved map depth | Median claimed z after rotation only |
|---|---:|---:|
| Left nearby car | 13.69 | 9.74 |
| Front nearby car | 9.09 | 7.20 |
| Farther centre car | 70.00 | 22.85 |

These are summaries over different sample sets, not ground-truth depth
errors. Nevertheless, the farther car's saved map also fails the direct
ablation: trusting it at all ranges leaves its background claims unchanged.

The visibility inequality is one-sided: `point_z <= surface_z + tolerance`.
An overestimated map depth admits background in front of that estimate, and
an unrelated nearer point also passes. A symmetric depth band would require
reliable object depth first; the current maps do not provide that here.

## Independent minimal reproduction

The local diagnostic `synthetic_repro.py` uses the existing full-pipeline
test fixture: perfect camera and pose, an exact dense foreground/car depth,
and a sparse sweep hitting only background at 40 m through the car mask.
It asserts that the background receives no label.

From the repository root:

```bash
PYTHONPATH=src/common/src:src/semantic_lifting/src python3 \
  data/artifacts/eval_runs/lifting_background_diagnostic/synthetic_repro.py
```

Observed output, exit 1:

```text
exact car depth=8m; background=40m; background indices=[]
exact car depth=20m; background=40m; background indices=[0, 1, 2]
AssertionError: car mask incorrectly labels background with perfect camera and depth
```

Adding `--trust-all-depth` makes both cases pass, exit 0. This isolates the
10 m bypass independently of calibration. It is a deliberately failing
diagnostic, outside the normal test suite. It does not justify trusting all
depths in production: the real-scene map ablation above shows why.

## Artifacts and validation

Local, git-ignored diagnostic directory:
`data/artifacts/eval_runs/lifting_background_diagnostic/`.

- `replay.py`: fixed-scene production replay and four-way ablation.
- `overlay.png`: recorded / rotation-only / all-depth / combined comparison;
  yellow outlines are masks, points are coloured by optical depth. For
  readability it displays sweep 98; the table aggregates both sweeps.
- `results.json`, `replay.log`: per-mask claim counts and depth summaries.
- `gate_counts.json`: inspected background claims and gate counts.
- `synthetic_repro.py`: deterministic expected-failure probe above.

Recreate the real-scene outputs from the repository root:

```bash
PYTHONPATH=src/common/src:src/semantic_lifting/src python3 \
  data/artifacts/eval_runs/lifting_background_diagnostic/replay.py
```

Focused existing tests passed: 31 semantic-lifting projection, visibility and
pipeline tests, plus 11 perception depth-alignment and image-time-pose tests.
Run the component suites separately because both use the Python package name
`tests`.

## Exact user screenshot: 19:10:57.945 EDT

The user subsequently identified the screenshot as `/wato_world/refined_labels`
in `../wato_monorepo/bags/sensors_2026_07_07-23_01_24_2_with_eidos_debug.mcap`
at **2026-07-07 19:10:57.945 EDT** (23:10:57.945 UTC). Reading the MCAP's
actual MarkerArray gives the last message at `1783465857908281600` ns,
36.7184 ms before the requested time: chunk `0000`, reference sweep `1145`,
frame-sync frame `567`, paired sweeps `(1145, 1146)`.

The screenshot's **left box is track `0000-139`; the right box is
`0000-169`**. Recreating their BEV footprints and the point cloud matches the
image. Their MCAP dimensions match the current refined artifacts exactly.
Both boxes already exist as lifted-only proposals: neither has a motion
proposal or detector member supporting it at this frame.

| Track | Source proposal | Original seeds → largest group | Source l × w × h | Refined l × w × h |
|---|---|---|---|---|
| `0000-139` | `lifted-0d82a798-5f0-001145` | 106 → 106 | 6.95 × 2.49 × 2.17 m | 7.26 × 3.58 × 2.50 m |
| `0000-169` | `lifted-03eef08d-8a1-001145` | 64 → 49 | 8.09 × 2.27 × 2.19 m | 10.28 × 2.91 × 2.53 m |

Replayed `_ChunkLifter.sweep()` against this frame, followed by the actual
`lifted.refine_support()` and `lifted.proposals()` calls. The recorded replay
asserts that its point-key sets equal the saved candidate sets and that all
seven source box geometry fields match the saved proposals within 1e-6.
Both assertions pass. This locks the diagnosis to the real source boxes,
rather than only to a nearby scene or a synthetic example.

Relevant current masks and image numbers:

| Track | Camera / image | Masklet | Recorded claimed z p10 / median / p90 | Rotation-only claimed z p10 / median / p90 | Median saved map depth |
|---|---|---|---|---|---|
| `0000-139` | NN / 578 | `0d82a798-5f0` | 45.00 / 46.24 / 48.12 m | 33.16 / 33.51 / 33.90 m | 138.88 m |
| `0000-139` | NE / 578 | `25da8b3d-57a` | 49.99 / 50.54 / 51.12 m | 35.93 / 36.43 / 37.02 m | 84.69 m |
| `0000-169` | NE / 578 | `03eef08d-8a1` | 30.58 / 58.35 / 60.61 m | 30.13 / 30.40 / 30.98 m | 53.62 m |

Different cameras have different optical z for the same surface. The
experimental NN and NE rotations are the ones already documented in the
earlier measurements; no parameters were fitted to this screenshot.
Masks, ego poses, points, depth maps and the existing instance grouping stay
fixed. Refit source box centres move **14.42 m** (`0000-139`) and **29.73 m**
(`0000-169`) in XY, toward the foreground cars. The rotation-only source
sizes are 4.10 × 0.77 × 1.11 m and 2.47 × 1.55 × 0.72 m respectively:
partial visible-surface fits, not validated amodal car dimensions. The
replay stops at source proposals; it does not rerun tracking/refinement or
claim that their final boxes are fixed.

This exact case exposes three successive failures:

1. **Projection selects the background.** The masks visibly cover the two
   foreground cars, but recorded camera geometry projects distant returns
   into them. The experimental rotation moves the accepted support from
   background depths of roughly 45–61 m to car surfaces around 30–37 m.
2. **Depth cannot reliably rescue it.** All 209 per-camera claims from these
   three masks bypass the 10 m depth-map gate. Trusting the maps at every
   finite range leaves `0000-139`'s 106 candidate points and fitted box
   completely unchanged: its map depths are beyond even the background.
   It reduces `0000-169` to 15 foreground seeds, of which 14 survive cleanup.
3. **Largest-group selection reinforces the mistake.** For `0000-169`, the
   recorded seed set contains 15 foreground returns and 49 background
   returns. The 2 m connectivity cleanup keeps the larger background group
   and discards the actual car. Tracking carries the resulting lifted-only
   detections as strong, and refinement's `percentile` path enlarges them.
   Neither final box is an accepted SLF fit. The displayed `1.00` confidence
   comes from proposal scores/class agreement, not verified surface
   ownership or metric accuracy.

The local diagnostic directory now includes `screenshot/`:

- `markers.json`, `extract_markers.py`: decoded debug-bag message, timestamp
  and exact dimension checks against current artifacts.
- `matched_bev.png`: reconstructed screenshot with readable track IDs.
- `camera_comparison.png`: NN and NE masks and accepted returns, under the
  recorded geometry, rotation-only change, and depth-range-only change.
- `trace.py`, `trace.json`: complete track/source lineage, depth summaries,
  candidate-key equality checks and source box replays.

From the repository root, this command reproduces the real code path and
then fails on the reported association (about 6 s):

```bash
PYTHONPATH=src/common/src:src/semantic_lifting/src:src/proposal_generation/src \
  python3 data/artifacts/eval_runs/lifting_background_diagnostic/screenshot/trace.py \
  --check-foreground recorded
```

```text
CHECK recorded 0000-139 claimed optical depth p90 48.12 must be <40m
CHECK recorded 0000-169 claimed optical depth p90 60.61 must be <40m
AssertionError: foreground car masks label background: 0000-139, 0000-169
```

Replacing `recorded` with `rotation_only` passes, with p90 depths 33.90 and
30.98 m. The 40 m separator is specific to the two foreground surfaces and
the distinct background surfaces observed here; it is not a proposed
production range limit or ground-truth certification of the rotation.

## Correction order and limits

First validate the rig calibration with independent static image/LiDAR
correspondences across the field of view and cameras. A constant rotation
improving this frozen scene is strong evidence of alignment error, but does
not uniquely separate mount rotation from pose/timing bias. The earlier
independent epipolar result supports the rotation explanation; this replay
does not re-estimate it. Other cameras and moving objects still need checks.

Then regenerate metric depth using the validated calibration, before rerunning
semantic lifting and downstream stages. The same extrinsics project the
LiDAR anchors used to fit depth (`perception_2d/pipeline.py:236–244`), so
correcting lifting alone leaves depth fitted with the old associations.
Validate depth accuracy by range before increasing its trusted range.

Finally measure residual background leakage and object-level support after
geometric refinement. A sparse, one-sided visibility gate cannot guarantee
object ownership even with correct calibration. Mask drift and dynamic-object
timing remain possible additional causes; this diagnosis does not establish
that every reported case has the same source.
