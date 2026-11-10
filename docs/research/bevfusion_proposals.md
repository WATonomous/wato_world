# Learned 3D proposals from the monorepo BEVFusion implementation

Research date: 2026-11-09. This is source inspection and an integration assessment, not an inference benchmark.

**Status (2026-11-09): implemented** as the `detection_3d` component, with
proposal_generation's `fusion_detector` source and tracking's detector evidence
(`src/detection_3d/README.md`; measurements in
`docs/detection_3d_measurements.md`). Checking against nuScenes ground truth
corrected one statement below: the vendor z is the box's *centre* (MIT's
loader read nuScenes centres as bottoms), so the car's ROS output, which
publishes it as the centre, is right; "z-center versus bottom-center" is
settled. BEVFusion is now on wato_monorepo `main` (#474).

## Scope and source revision

The adjacent checkout is `/home/watonomous/brianzheng/wato_monorepo`, currently on `bzheng/eidos_vio`; that checkout has no `src/perception/bevfusion` directory. Its existing local remote-tracking branch `origin/ashish/bevfusion` contains the implementation inspected here at commit `9edb8b334fcb53d9434aa4f3aa728a2a8f09b095`. Inspection used `git show` without fetching, switching branches, or changing that repository. Links below identify that exact revision rather than nonexistent working-tree files. The older local `ashish/bevfusion` branch differs materially, including its treatment of the fifth LiDAR feature. No applicable `AGENTS.md` was found in the inspected repository trees or workspace ancestors.

## Finding

**Recommendation for wato_world: a learned detector is worth an isolated proposal-source experiment.** The strongest expected benefits are completing partial boxes and adding objects the current sources miss. Prefer the existing BEVFusion core when a working fusion runtime and trustworthy six-camera calibration are already available; prefer a separately trained LiDAR-only backend when the aim is geometric evidence independent of cameras. This is an engineering judgment from the evidence below, not a measured gain or a recommendation to replace the current sources.

**The existing node supplies learned 3D detections that could become proposal priors, but it is a camera-plus-LiDAR detector, not a ready-to-use LiDAR-only backend.** Its core returns position, dimensions, yaw, score, class ID, and planar velocity. Its ROS publisher exposes geometry, class, and score but drops velocity. Reusing those outputs is a plausible way to introduce learned object geometry; whether it improves this pipeline must be measured. [Core output][core-output], [ROS output][ros-output].

## Verified implementation behavior

### Modalities and runtime inputs

- The default model is `resnet50_trt11_int8`, with six configured panoramic camera frames. The runtime requires a nonempty image batch whose size equals `camera_names.size()`, as well as an initialized core and camera calibration. Missing images cause the frame to be skipped. [Parameters][params], [frame guards][frame-guards].
- The node synchronizes `deep_msgs/MultiImage` or `deep_msgs/MultiImageCompressed` with `sensor_msgs/PointCloud2`; the raw/compressed switch does not disable camera inference. Activation creates an `ApproximateTime` synchronizer over both streams. [Subscriptions][subscriptions].
- Images are reordered by their `header.frame_id` according to `camera_names`; compressed images are decoded and converted to RGB. Both callbacks pass the **image-batch header** into `processFrame`, which supplies its timestamp to the published detections. Thus a recorded detection timestamp is not necessarily the LiDAR sweep timestamp. [Image callbacks][callbacks], [publication][publication].
- Calibration comes from `deep_msgs/MultiCameraInfo` and TF. The node pads each `CameraInfo.k` into a row-major 4×4 intrinsic matrix, looks up camera-to-LiDAR extrinsics, computes `lidar2image = K * inverse(camera2lidar)`, and constructs a resize/crop augmentation matrix. A dataset adapter must supply matrices matching its actual image geometry. [Calibration][calibration].
- LiDAR input requires FLOAT32 `x,y,z,intensity` fields. Nonfinite points are skipped. The flat core input is `[x,y,z,intensity,time_lag]`; the current ROS path sets time lag to `0.0` for a single sweep. **Do not copy the older branch's ring-ID-as-feature behavior.** [Point conversion][point-conversion], [core input contract][core-input].
- The core converts the flat point buffer to FP16 before invoking the vendor pipeline. No ground removal, moving-object filtering, or object clustering occurs in this input path. Consequently the practical detector input should be the full compatible sensor cloud, before proposal-specific pruning. The last sentence is an adapter recommendation inferred from the code path, not a measured requirement on this dataset. [Core inference][core-infer].

### Predictions, classes, and filtering

| Item | Verified behavior |
| --- | --- |
| Core prediction | `position.{x,y,z}`, `size.{w,l,h}`, `velocity.{vx,vy}`, `z_rotation`, `score`, `id`. [Core output][core-output] |
| Published prediction | `vision_msgs/Detection3DArray`; bounding-box pose and dimensions, numeric class encoded as a string, and score. No velocity is copied into the ROS message. [ROS output][ros-output] |
| Coordinates | Core boxes are in the configured LiDAR frame. The node transforms their pose into `target_frame` (default `base_link`), negates vendor yaw, and publishes size.x=`w`, size.y=`l`, size.z=`h`. [ROS output][ros-output] |
| Classes | Constants label IDs 0 car, 1 truck, 2 construction vehicle, 3 bus, 4 trailer, 5 barrier, 6 motorcycle, 7 bicycle, 8 pedestrian, 9 traffic cone. These IDs are semantic classes, not persistent track identities. [Class constants][classes], [hypothesis creation][ros-output] |
| Score | Package YAML threshold `0.08`; threshold is passed to the vendor decoder. This is a deployment choice, not a calibrated probability or an evaluated threshold for this pipeline. [Parameters][params], [decoder configuration][core-init] |
| Additional suppression | The node drops box centers inside an XY radius of 2.5 m from the LiDAR origin by default. TF lookup failure also produces an empty output array. [Parameters][params], [ROS output][ros-output] |
| Range and caps | Voxelization x/y ±54 m, z −5 to 3 m; voxels 0.075×0.075×0.2 m; 300,000 input points, 160,000 voxels, 10 points/voxel. The node truncates excess points. Post-decoder center bounds are ±61.2 m in x/y, ±10 m z. These are different filters; increasing decoder limits alone does not extend the trained input representation. [Parameters][params], [truncation][truncation] |
| Synchronization | Package YAML allows up to 200 ms difference, queue 5. Use this as a deployment fact, not an acceptable matching tolerance for offline evaluation. [Parameters][params] |

The output looks useful as an object-shape prior because geometry comes from a learned detector rather than a bounding box fitted only to visible returns. This is an inference about its role in the pipeline. It does not establish amodal accuracy, yaw reliability, calibration, or recall on the user's scenes.

### Model and deployment requirements

The node resolves four TensorRT plans in `<model_dir>/build`: `camera.backbone.plan`, `camera.vtransform.plan`, `fuser.plan`, and `head.bbox.plan`, plus `<model_dir>/lidar.backbone.xyz.onnx`. Missing plans are built from the corresponding ONNX files. A `.pth` checkpoint alone is insufficient for this C++ path. Precision is selected from whether the directory name contains `int8`. [Model paths][model-paths], [artifact checks and builds][model-build].

Initialization loads `libcustom_layernorm.so` and creates camera preprocessing, geometry, LiDAR sparse convolution, fusion, and detection components. The source requires TensorRT major version ≥10; the vendor CMake configuration supplies CUDA/TensorRT paths and architecture defaults. Existing plans should be validated or rebuilt for the actual runtime rather than assumed portable. [Core initialization][core-init], [TensorRT assertion][trt-version], [vendor build][vendor-build].

The standalone launch loads package parameters and remaps output detections to `/perception/detections_3d_bev`. The node is a ROS lifecycle node: configure initializes the core; activate creates subscribers, synchronization, and publishers. An offline adapter that calls `BEVFusionCore` directly could avoid ROS message construction while retaining the same model/calibration requirements. This is an implementation option inferred from the exposed core API. [Launch][launch], [lifecycle activation][subscriptions], [core API][core-input].

## LiDAR-only distinction, checked against upstream primary sources

MIT's upstream repository publishes a **separate LiDAR-only TransFusion-L baseline**, checkpoint, and `configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml` configuration. That is evidence that a learned LiDAR-only model is available in the model family; it is not evidence that the deployed fusion engines can run without cameras. The upstream stack also documents older Python/PyTorch dependencies, so reproducibility needs an isolated compatible environment. [MIT model table and usage](https://github.com/mit-han-lab/bevfusion#results), [LiDAR-only configuration](https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml).

NVIDIA's CUDA implementation advertises fusion model artifacts and six-camera example data. Its actual forward path executes the camera backbone, BEV pooling, camera view transform, fusion with LiDAR features, and detection head. This supports the local source finding that removing image subscription or providing an empty batch does not turn the existing engine into a LiDAR-only detector. Black placeholder images would be an unvalidated input substitution; a LiDAR-only checkpoint needs a compatible inference graph/backend. [NVIDIA model documentation](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/README.md), [NVIDIA forward implementation](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/src/bevfusion/bevfusion.cpp).

## Practical adapter choices — recommendations, not implemented changes

| Option | When useful | Work needed |
| --- | --- | --- |
| Import recorded BEVFusion detections | Compatible detections already exist for the same drive | Decode ROS boxes into pipeline proposals; reconcile timestamps, frames, class IDs, axes, and filters. Avoid loading model/runtime dependencies. Velocity cannot be recovered from the current published message. |
| Run the current fusion core offline | Six usable calibrated images are available with each cloud | Build a small runner around the ROS-free core, feed RGB and full LiDAR inputs plus matching calibration, and write cached predictions at reference sweep times. Export velocity and model metadata before the ROS publisher discards them. |
| Add a true LiDAR-only detector | Independent geometric prior or camera-free operation is required | Use a LiDAR-only checkpoint with its matching config and backend; export a common proposal representation. The MIT TransFusion-L model is an evidenced candidate, not a drop-in replacement for these fusion plans. |

For any option, preserve detector center, dimensions, and yaw through proposal creation; associate observed points as supporting evidence rather than immediately refitting the detector box to those points. Keep source model, checkpoint/config identity, score, class, timestamp, frame, and optional velocity in the cache. Any confidence weighting, size prior, and birth policy should be evaluated by source/class/distance rather than treating score as calibrated certainty.

Before evaluating quality, verify a few boxes by comparing **all eight corners** after conversion. The node's width/length assignment and yaw negation make a blind `(w,l,h)`→`(l,w,h)` copy unsafe; the appropriate conversion depends on the coordinate definition of the consumed output. Also check z-center versus bottom-center convention against the exact decoder/checkpoint, intensity scaling, single- versus multi-sweep time lag, rectified camera intrinsics, image resize/crop, ego suppression, and sweep timestamp. These are verification items, not claims that a particular correction is always needed. [ROS conversion][ros-output], [calibration][calibration], [point contract][core-input].

No BEVFusion inference or accuracy experiment was run for this note. The adjacent checkout's `models/` directory contains YOLO artifacts, not the configured BEVFusion directory. Model names elsewhere on the host do not establish checkpoint compatibility or engine availability. Improvement claims remain untested.

## Why a detector fits this pipeline's measured weaknesses

The following are existing repository measurements, not new experiments and not BEVFusion performance numbers. They describe their documented configurations; a fresh comparison must rerun the current working-tree baseline.

| Existing evidence | Implication for a learned proposal source (inference) |
| --- | --- |
| On nuScenes scene-0061, fused proposal recall 0.33, precision 0.70, median matched 3D IoU 0.12. Recall is 0.08 at 30–50 m. Lifted and fused results are almost identical. | Another box source has room to improve both recall and shape; more tracking sophistication cannot supply observations never proposed. |
| Matched lifted cars are about 0.64 m short and 0.38 m narrow; pedestrian IoU is 0.07 despite small center error. | Predicted full-object dimensions and centers address missing surfaces more directly than smoothing point-fitted boxes. |
| Step F has recall 0.02 and precision 0.89 in that scene; it targets motion. | A detector can complement motion evidence with stationary objects and missed movers. |
| The tracking research note attributes 39% of 152 breaks to shape gates and 38% to duplicates, over two nuScenes scenes. | Stable detector geometry could reduce shape changes, but simply appending detector boxes risks creating further duplicates. |

Sources: [proposal measurements](../proposal_generation_measurements.md#against-ground-truth-nuscenes-scene-0061), [tracking break analysis](tracking_improvements.md#starting-point). These small nuScenes samples do not establish WATO transfer quality. WATO's sparse Velodyne returns and different cameras remain a domain-transfer question; the node's single-sweep input is also different from a multi-sweep training input. Do not equate published benchmark AP with the recall/precision above, which use this repository's matching and visibility rules.

The expected benefit is strongest for represented classes and partly observed objects within the model's input range. It does not extend the model's ±54 m input grid, guarantee recovery behind complete occlusion, or solve pose errors. Fusion still uses camera geometry, so it cannot be assumed to provide independent corroboration of lifting when both share a miscalibrated camera. The WATO measurements already identify camera orientation as a material failure source. [Range parameters][params], [WATO camera alignment measurements](../semantic_lifting_measurements.md), [camera geometry contract](../camera_geometry_contract.md).

## Integration changes required in wato_world

The artifact architecture already anticipates `lidar_detector`: it is an accepted source in `wato_common.artifacts.proposals`. However, the actual orchestrator produces only `lidar_mos` and `lifted`, and the old `config/pipeline.yaml` detector list is not the active component configuration. An accepted provenance value does not implement a detector. [Artifact source enum](../../src/common/src/wato_common/artifacts/proposals.py), [orchestrator](../../src/proposal_generation/src/wato_proposal_generation/pipeline.py), [active component configuration](../../src/proposal_generation/config/proposal_generation.wato.yaml).

Recommended first implementation, subject to the experiment below:

1. **Cache detector output in a separate GPU stage.** Follow the file-only component boundary and put any shared cache reader/writer in `wato_common.artifacts`. Save raw model identity, configuration/checkpoint hashes, sensor times, local frame, geometry, class, score, and optional velocity. Preserve enough metadata to distinguish multimodal BEVFusion from a LiDAR-only model even if both eventually use the current `lidar_detector` provenance. Proposal generation can remain CPU-only. [Repository component contracts](../../CLAUDE.md), [current proposal reader/writer](../../src/common/src/wato_common/artifacts/proposals.py).
2. **Feed a compatible full cloud in a local reference frame.** The existing processed sweep preserves intensity, and the raw sweep is also available. If using world sweeps, transform full XYZ back to the chosen reference-time LiDAR frame before inference; preserve intensity and the model's time-lag feature. Begin with the configured center LiDAR to reproduce the node's input, rather than assuming merged corner clouds improve a pretrained model. Include ground and stationary returns. Retain actual camera and LiDAR times and transform predicted boxes to world at their documented reference time. Adding temporal sweeps or other LiDARs is a separate ablation. [Sweep artifact contracts](../../src/common/src/wato_common/artifacts/lidar_sweeps.py), [point conversion][point-conversion], [parameters][params].
3. **Preserve the predicted geometry when associating sources.** Current `fusion.fuse` fits a new minimum-area box to the union of motion/lifted support and then lowers its bottom to ground. Reusing that operation on detector support would discard the very missing-surface prior being added. Use support overlap/containment, compatible centers, and class evidence to associate partial observations to predicted objects, with explicit handling of one detector claiming multiple real objects. Retain detector geometry as an alternative and preserve matched masklet IDs for downstream views. Keep unmatched current proposals and evaluate detector-only additions. Association thresholds and geometry selection need measurement; a low IoU between a sliver and its full box alone is insufficient reason to reject the match. [Current fusion](../../src/proposal_generation/src/wato_proposal_generation/fusion.py), [ground adjustment](../../src/common/src/wato_common/geometry/objects.py).
4. **Teach tracking what detector evidence means.** `detections.detection` recognizes only motion and lifted members: a detector-only fused row currently has no instance key and is always weak. Strong means motion-plus-lift or sufficient camera seed evidence. Add measured detector confidence/birth rules without pretending a fusion detector and a camera mask are independent sensors. Source scores are currently placeholders and averaging them does not calibrate reliability. Preserve motion evidence separately: a detected object is not necessarily moving. The Kalman filter's fixed noise targets cluster boxes, so source-specific uncertainty is a later option after measuring residuals, not the first integration step. [Evidence classification](../../src/tracking/src/wato_tracking/detections.py), [proposal scoring](../../src/proposal_generation/README.md), [Kalman noise](../../src/common/src/wato_common/tracking/kalman.py).
5. **Evaluate refinement as well as tracking.** Track artifacts currently write the observed proposal box, not the Kalman-filtered box. Label refinement then refits parked vehicles to accumulated returns, derives other tracks' sizes from observed boxes, and can replace geometry through SLF. A detector's improved dimensions may survive the percentile path, but can be lost through refitting. Compare retaining accepted detector geometry against the current refiner, and carry enough provenance to support that choice. A detector-only track has no matched masklets, so SLF cannot be assumed to supply missing views. [Track output](../../src/tracking/src/wato_tracking/pipeline.py), [refinement implementation](../../src/label_refinement/src/wato_label_refinement/pipeline.py), [refinement behavior](../../src/label_refinement/README.md).

The working tree also contains an optional circle-NMS path that ranks detections by strong status and point count. If enabled, detector-only boxes would currently be weak and could lose to a partial lifted box. Include this ranking in the detector-evidence integration and freeze its setting for comparisons. [Duplicate suppression](../../src/tracking/src/wato_tracking/detections.py).

Class conversion should explicitly retain the six current classes: IDs 0, 1, 3, 6, 7, 8 map to car, truck, bus, motorcycle, bicycle, pedestrian. Other detector classes should remain available in the raw cache and be excluded from the fixed-taxonomy comparison unless the taxonomy is deliberately extended. In particular, do not silently relabel trailers or construction vehicles as trucks. Detector-provided classes also require updating the current schema documentation, which describes `cls` as coming from lifting. [Class IDs][classes], [current proposal schema](../../src/common/src/wato_common/schemas.py).

Before any quality comparison, audit the image contract: the monorepo node reads `CameraInfo.k`, whereas this pipeline uses the intrinsic block of `P` for rectified images. Supply intrinsics describing the actual stored RGB images and their exact resize/crop; this is a concrete compatibility check, not proof that every recorded BEVFusion box is wrong. Similarly, preserve the sweep association instead of matching image-stamped detections to sweeps with the node's 200 ms tolerance. [Calibration][calibration], [camera contract](../camera_geometry_contract.md#stored-image-versus-raw-metadata), [publication][publication].

## Smallest experiment that answers whether it is worth keeping

Freeze the baseline config, code state, inputs, and outputs in a separate artifact root so the experiment does not overwrite the current labels. Use the two available nuScenes scenes for adapter checks, then a manually labelled WATO subset spanning near/far, parked/moving, occlusion, and pedestrians/two-wheelers. Audit checkpoint training-scene overlap before interpreting nuScenes results as held-out accuracy. WATO checks must include the known camera calibration problem; otherwise a fusion-model failure may be a calibration failure rather than evidence against learned LiDAR proposals.

| Run | Question answered |
| --- | --- |
| Current pipeline, freshly rerun | What is the actual current baseline? |
| Detector alone | Does the adapter/model produce useful geometry on this rig at all? |
| Current pipeline + detector geometry on matched objects only | Does shape completion help without changing which objects are proposed? |
| Current pipeline + matched geometry + unmatched detector proposals | Does additional recall outweigh false positives and duplicates? |

Sweep detector thresholds and compare recall **at comparable precision**, per class and distance; also report new true objects and duplicate false boxes. Compare center, dimension, yaw, and 3D IoU errors on common matched objects as well as each run's overall matches. Track continuity needs tracks per GT object, duplicate frames, identity switches, and the working tree's association metrics; final refined-label geometry needs the same scrutiny as proposal geometry. Measure actual GPU time, memory, engine build requirements, and offline throughput rather than importing an upstream FPS claim. The existing proposal evaluator already iterates over all declared provenances, including `lidar_detector`. [Proposal evaluator](../../src/proposal_generation/scripts/eval_proposals.py), [track evaluator](../../src/tracking/scripts/eval_tracks.py), [refinement evaluator](../../src/label_refinement/scripts/eval_refined.py).

Keep the source if it adds correct WATO objects at comparable precision or materially improves full-box geometry/continuity without losing current-source coverage, and those gains survive refinement. If it only reshapes already-detected objects, retain it as a geometry prior while deferring new detector births. If fusion cannot be evaluated cleanly because of camera calibration, test a true LiDAR-only backend before rejecting the idea of a learned geometric source. No numerical acceptance margin is justified by the source inspection alone.

[params]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/config/params.yaml#L4-L94
[frame-guards]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_node.cpp#L217-L256
[callbacks]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_node.cpp#L318-L485
[publication]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_node.cpp#L284-L292
[subscriptions]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_node.cpp#L1136-L1185
[calibration]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_node.cpp#L529-L631
[point-conversion]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_node.cpp#L905-L944
[core-input]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/include/bevfusion/bevfusion_core.hpp#L135-L185
[core-infer]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_core.cpp#L202-L278
[core-output]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_core.cpp#L265-L278
[ros-output]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_node.cpp#L762-L848
[classes]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/include/bevfusion/bevfusion_constants.hpp#L38-L62
[truncation]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_node.cpp#L719-L759
[model-paths]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_node.cpp#L139-L168
[model-build]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_core.cpp#L296-L357
[core-init]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_core.cpp#L80-L199
[trt-version]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/src/bevfusion_core.cpp#L40-L43
[vendor-build]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/cuda_bevfusion_vendor/CMakeLists.txt#L71-L138
[launch]: https://github.com/WATonomous/wato_monorepo/blob/9edb8b334fcb53d9434aa4f3aa728a2a8f09b095/src/perception/bevfusion/bevfusion/launch/bevfusion.launch.yaml#L14-L44
