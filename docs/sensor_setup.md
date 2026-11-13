# WATO sensor selection and BEVFusion influence

## Current profiles

`--profile wato` selects exactly eight top cameras (`PANO_NN`, `PANO_NE`,
`PANO_EE`, `PANO_SE`, `PANO_SS`, `PANO_SW`, `PANO_WW`, `PANO_NW`) and
`/lidar_cc/velodyne_points`. Lower cameras, corner LiDARs and merged clouds
are excluded. The pose source remains `/world_modeling/liso/odometry`.
`wato_no_lidar_nw` is a compatibility alias with the same selection.
The legacy `wato_6cam` ingest profile explicitly selects six cameras for
recordings that lack EE/WW; it does not satisfy the eight-camera contract.

BEVFusion uses the same eight cameras, their individual calibration and
image-time poses, and only centre-LiDAR points. Up to nine earlier `lidar_cc`
sweeps within 0.5 s accompany the current sweep. Frames missing a required
camera within 200 ms are skipped, with the camera IDs recorded.

The eight-camera export specializes the shared encoder's input/output shapes
and five Reshape targets; learned tensors and quantization scales are
unchanged. Camera geometry, depth, normalization and BEV pooling run with
eight views in one native pass. The other four graphs are unchanged.
The source and exported camera-backbone SHA256 values are respectively
`9d08cd562600fa73d1131d65b60c20645c6f2781338cd232185a1b235aae46da` and
`2fb57dd27265ffb9582e8e6076c426cb8136e5f44040f50869a7d8716fbda411`.
See [export instructions](../src/detection_3d/README.md#eight-camera-export).

## Existing July bag artifacts

The existing `sensors_2026_07_07_23_01_24_2_with_eidos_mcap` output directory
has **not** been regenerated under this sensor policy. Its ingest manifest
and calibration record eight top cameras plus `lidar_cc` and `lidar_ne`.
Its earlier BEVFusion run used six cameras, omitting EE/WW, with centre-LiDAR
points only. Changing YAML does not change these saved outputs.

On chunk 0000 before this change, 4,814 of 8,608 fused proposals included a
BEVFusion detection and took its geometry; 1,223 were detector-only. Only
eight of those detector-only rows reached score 0.6. These counts describe
the old run and policy, rather than quality estimates or predictions for
eight-camera inference.

## Supplemental detector policy

Motion clusters and camera-lifted instances are produced before reading
BEVFusion proposals. Detection priors do not feed motion classification,
depth, mask tracking, lifting or lifted support refinement.

- Matched detections add source provenance. The existing object's box,
  class, score, point count and camera support remain unchanged.
- Matched detector scores do not make weak independent evidence strong.
  Detector and camera lifting are never counted as two independent sources.
- Unmatched detections add recall only at score ≥ 0.6 and with at least five
  current-frame non-ground points. These are the existing detector birth
  confidence and lifted support minimum, not calibrated WATO thresholds.
- Duplicate suppression and track association give motion/lifted observations
  priority over detector-only additions, including weak independent boxes.
- Raw detector boxes and each source's proposals remain available for
  separate evaluation. Refinement still uses the resulting tracks, masks and
  observed points, so detector-only additions can create new label tracks.

This bounds detector influence without claiming accuracy gains. WATO ground
truth evaluation and a frozen-input enabled/disabled comparison remain
necessary to quantify the eight-view detector's added recall and false positives.

## GPU input audit (2026-11-11)

The rebuilt detection image and TensorRT 10.16 engines ran three real frames
of the July bag on an RTX 4090. Each call received `(8, 900, 1600, 3)` images,
eight calibration matrices and eight timestamps, with only `lidar_cc` input
sweep IDs. The frames used 17,983 / 35,933 / 53,911 points from 1 / 2 / 3
centre sweeps and produced 200 raw boxes each.

The isolated camera encoder returned `(8, 32, 88, 80)` features. Repeated
identical inference had zero feature difference. Ablating each view changed
that view's features (maximum absolute delta 2.69–5.03) and changed the other
seven views by exactly zero. All eight full-model ablations exceeded five
times the repeat score variation in the saved report. Full-model inference
includes stochastic LiDAR voxelization; an earlier SW repeat fell below that
conservative threshold. The deterministic encoder probe establishes per-view
processing independently of final detection changes on a particular frame.

Machine-readable report:
`data/artifacts/eval_runs/bevfusion_8cam_cc_inputs.json` (local, untracked).
Reproduce inside the detection image after ingest and LiDAR preprocessing:

```bash
python3 /ws/src/detection_3d/scripts/audit_inputs.py \
  --bag sensors_2026_07_07_23_01_24_2_with_eidos_mcap \
  --config /ws/src/detection_3d/config/detection_3d.wato.yaml \
  --frames 3 --ablate-cameras \
  --out /data/artifacts/eval_runs/bevfusion_8cam_cc_inputs.json
```

## Regenerating saved labels

Ingest rejects reuse when completed chunks contain different camera/LiDAR
topics. Detection rejects reuse when its resolved inference configuration
differs or is absent from an older manifest. Rebuild the changed components
before using deploy images; development services bind the current sources.
Sensor changes require ingest and every downstream stage to rerun:

```bash
./watod pipeline \
  --bag /data/bags/sensors_2026_07_07-23_01_24_2_with_eidos.mcap \
  --profile wato --force ingest
```

Use a new `--bag-id` if the old output set must remain available. A full bag
relabeling run was not part of the GPU input audit above.
