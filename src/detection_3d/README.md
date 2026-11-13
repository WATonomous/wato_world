# detection_3d

Learned 3D boxes from BEVFusion, the camera + LiDAR detector the car runs
(`wato_monorepo` `src/perception/bevfusion`): the same int8 model, through the
same CUDA-BEVFusion fork, on every frame of a bag, offline. proposal_generation
turns the boxes it keeps into `fusion_detector` proposals. GPU only.
Measurements: `docs/detection_3d_measurements.md`.

The model predicts each object's whole box — the sides LiDAR never hit, the
extent a partial mask can't show — where the other sources fit boxes to the
points they see. On nuScenes scene-0061 (a scene the model was trained on, so
an upper bound) it finds 91% of objects at score ≥ 0.1 with precision 0.85
and lengths/widths within 5 cm.

## What offline adds over the car

Per frame — lidar_preprocessing's frame_sync tick, anchored on the canonical
LiDAR — the model sees:

- **Its training frame.** BEVFusion learned nuScenes' LIDAR_TOP frame: x
  right, y forward, 1.84 m above the ground. `input_frame.kind:
  nuscenes_lidar` rebuilds it on any rig (the canonical LiDAR's x/y, ego
  axes, the training height); the car feeds `lidar_cc`'s own frame, x forward,
  a 90° turn. nuScenes bags use `sensor`: their LIDAR_TOP *is* that frame.
- **Earlier sweeps.** The nuScenes config accumulates the current sweep and 9
  earlier ones, each point tagged with its sweep's age (the 5th feature); the
  car feeds one sweep, age 0. Here: up to `points.past_sweeps` earlier sweeps
  no older than `points.max_time_lag_s` (the ages it was trained on), from the
  deskewed world sweeps, close points of earlier sweeps removed
  (LoadPointsFromMultiSweeps' `remove_close`).
- **Images at nuScenes geometry.** The model's camera input is a 1600×900
  image it resizes by 0.48 and crops from the bottom. The car resizes WATO's
  1280×1024 by 0.55, so the crop starts at row 558 — at or below every
  camera's principal point: almost nothing above the horizon reaches the
  network. Here each image is scaled to 1600 wide (×1.25 on WATO: focal ≈
  1280 px against nuScenes' ≈ 1266, the same ~64° field of view) and a
  900-high band is cut with its principal point on nuScenes' row
  (`image.principal_row_px`); K follows.
- **Each image at its own time.** The camera-to-LiDAR matrix composes the ego
  pose at the image's time with the one at the frame's, as nuScenes' info
  converter does for BEVFusion's training data. With that, each camera's
  nearest image within the car's sync tolerance (`camera_sync.max_offset_ms`,
  200 ms) is used, so a sweep between two 12 Hz images still runs.
- **Every box.** All 200 boxes per frame are kept, with scores, classes and
  velocity; thresholds are proposal_generation's, so they change without a
  GPU re-run.

On WATO (no ground truth yet), the rebuilt frame and the earlier sweeps each
raise agreement with camera-lifted objects (0.46 against 0.40 in the car's
frame and 0.32 with one sweep); adding the VLP-16 `lidar_ne` collapses it
(0.04), so `points.lidars` stays the canonical LiDAR.

## The box conventions

The vendor decoder returns (x, y, z), (w, l, h) and a rotation in
mmdetection3d v0.x's LiDAR convention (`geometry.py`): the box faces
`-rot - pi/2`, `l` is along that, and z is the box's **centre** — the decoder
subtracts h/2 as if from a centre to a bottom, but MIT BEVFusion's loader read
nuScenes' centres as bottoms when it built the targets, so what comes out is
the centre. Checked against nuScenes ground truth (directed heading error
4.8° median, 6.6% flipped; height error −0.17 m once z is the centre, +0.77 m
as the bottom). The car publishes z as the centre too, which is right.

## Run

`ingest` and `lidar_preprocessing` first, and the model files installed (below):

```bash
./watod run detection_3d --bag <bag_id> [--config /ws/src/detection_3d/config/detection_3d.wato.yaml] [--force] [--chunk 0000]
```

`watod pipeline` runs it after lidar_preprocessing. A chunk whose
`detections_3d.parquet` exists is skipped. The first run on a GPU builds the
TensorRT engines (~1.5 min on an RTX 4090) into `data/models/.cache/tensorrt`;
a chunk then takes about 0.1 s per frame.

## Model files

The car's `resnet50_trt11_int8` (ResNet-50 camera backbone, int8 LiDAR
backbone), from the "Bevfusion Models" export (`resnet50int8_trt11/`):

```bash
D=data/models/bevfusion/resnet50_trt11_int8 && mkdir -p $D
for f in camera.backbone.onnx camera.vtransform.onnx fuser.onnx head.bbox.onnx lidar.backbone.xyz.onnx default.yaml; do
  unzip -p "Bevfusion Models-...-001.zip" "Bevfusion Models/resnet50int8_trt11/$f" > $D/$f
done
```

`model.sha256` pins every ONNX file; a different file stops the run. The zip's
prebuilt `.plan` engines are not used: an engine only runs on the TensorRT
version and GPU that built it.

## Image

`docker/detection_3d.Dockerfile` compiles WATonomous's CUDA-BEVFusion fork at a
pinned commit (`libbevfusion_core`, the LayerNorm plugin, and its pybind11
module `libpybev`) against TensorRT 10.16 (the car installs TensorRT 10), for
`CUDASM` 89 (Ada, the car's default; set 86 for an RTX 30-series). Two
departures from the car, both needed for it to work here:

- pybind11 is moved from the fork's 2.11 to 2.13.6: the image's NumPy 2 changed
  the C ABI, and libpybev built on 2.11 returns garbage.
- Engines get INT8 only for `camera.backbone` and `fuser`, whose ONNX carry
  quantization scales (NVIDIA's `build_trt_engine.sh`). The car's builder sets
  it for all four, which TensorRT 10 refuses for `camera.vtransform`.

Inference isn't bit-exact: two runs on one frame differ by ~1 cm.

## Output

Per chunk, through `wato_common.artifacts.detections_3d`:

- `detections_3d_frames.parquet`: one row per frame — `ok`, or skipped and why
  (`no_pose`, `missing_camera`, `no_points`) — with the sweeps and image times
  that went in and `world_T_model`, the model's frame at the reference time.
- `detections_3d.parquet`: every box of every `ok` frame, in the world
  (centre, `l` along a *directed* heading, velocity) and as the model returned
  it (`model_*`), with the model's class (`CLASSES`, nuScenes' ten) and score.

`manifest_detection_3d.json` records the model files' hashes, the TensorRT
version, GPU, fork commit, camera-count ABI, engine directory and full resolved
inference configuration. Existing detections with a different or unrecorded
configuration require `--force` and a downstream rerun.

## Configuration

`config/detection_3d.yaml` (nuScenes) and `config/detection_3d.wato.yaml`
(`canonical_lidar: lidar_cc`, `points.lidars: [lidar_cc]`, all eight top cameras
`PANO_NN, NE, EE, SE, SS, SW, WW, NW`). The WATO profile uses
`resnet50_trt11_int8_8cam`, with `model.num_cameras: 8`; configuration, ONNX
shapes, CUDA normalization and geometry must agree. Each view carries its own
intrinsics and pose-adjusted extrinsics; pooling uses all eight in one pass.
`detection_3d.wato_6cam.yaml` explicitly retains the original six-view export
for older recordings without EE/WW; it is selected only by that legacy profile.
Every field is documented in
[`config.py`](src/wato_detection_3d/config.py).

## Eight-camera export

Install the six-camera source model above first, then install `onnx==1.17.0`
in the environment running the export. In the detection image's paths:

```bash
python3 -m wato_detection_3d.export_cameras \
  --config /ws/src/detection_3d/config/detection_3d.yaml \
  --output-dir /data/models/bevfusion/resnet50_trt11_int8_8cam --num-cameras 8
```

The source hashes are verified. The export specializes five Reshape targets
and camera-axis annotations; learned tensors and quantization scales stay
unchanged, and other model graphs are copied unchanged. ONNX shape inference
and full checking validate the result; `camera_export.json` records both sets
of hashes. The WATO YAML pins the resulting graph. Rebuild the detection image:
its checked-in vendor patch sizes Python/CUDA buffers from the camera count
and rejects mismatched engine/calibration/image shapes. This changes sensor
coverage without retraining; WATO detection accuracy still needs ground truth.

`scripts/audit_inputs.py --bag <id> --config <wato.yaml> --out <report.json>
--ablate-cameras` records actual image/sweep inputs and probes every camera
independently through TensorRT. Full-detector ablations are reported alongside
repeat noise; deterministic encoder ablations establish per-view processing.
See [sensor setup](../../docs/sensor_setup.md) for the measured audit.

## Camera calibration source

`calibration_source: recorded` is the default. `validated` requests the
camera-calibration stage's accepted result and fails on missing, stale,
unqualified or failed-camera evidence. Keep this choice consistent across
camera-dependent stages. Output/cache identities reject mixing calibrations;
rerun depth and downstream processing with `--force` after a change. See
[the calibration module](../camera_calibration/README.md).
