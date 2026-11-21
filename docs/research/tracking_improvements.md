# Tracking improvements: fragmentation, duplicates and re-linking

Research checked 2026-11-09. Scope: what published 3D trackers, especially
offline auto-labellers, do about one object becoming several tracks. Five
topics are covered: association costs and gates for partial or
non-overlapping boxes; tracklet re-linking after a first pass; duplicate
boxes and tracks that exist at the same time; coasting and never-terminating
tracks; and, more briefly, appearance re-identification, motion models and
the metrics that expose fragmentation.

Sources are the papers' own text, linked by arXiv HTML section anchor, and
the authors' code at the commits pinned below. Sections A to G describe what
the sources do. Deductions about this repository appear only in "What this
means for wato_world tracking" and are labelled as deductions. Figures marked
"measured in this repo, 2026-11-09" come from this repository's data, not
from any source.

## Starting point

These are repository facts, not source claims.

- **The tracker.** Each chunk is tracked separately, with one step per frame
  that has fused rows.
  - Motion model: a constant-velocity Kalman filter. The state is
    `[x, y, z, θ, l, w, h, vx, vy, vz]`, and the heading is undirected
    (modulo π) ([`kalman.py`](../../src/common/src/wato_common/tracking/kalman.py)).
  - Cost: Chen et al.'s centre distance + (1 − IoU) + (1 − volume ratio),
    solved by the Hungarian method. The gates are 2 m, 0.95 and 0.7
    ([`association.py`](../../src/common/src/wato_common/tracking/association.py)).
    Chen gives these values and says they were "determined based on the
    validation data" of its own pipeline
    ([Chen §III-D](https://arxiv.org/html/2201.04501#S3.SS4),
    [§III-F](https://arxiv.org/html/2201.04501#S3.SS6)).
  - Matching order: strong boxes are matched first.
  - Instance cue: a box that shares the track's last lifted-instance id gets
    a cost bonus of 1.0 and is held only to the distance gate.
  - Life cycle: a track coasts for 2 s; it runs forward only; every unmatched
    box starts a new track
    ([`tracking.yaml`](../../src/tracking/config/tracking.yaml),
    [`tracker.py`](../../src/tracking/src/wato_tracking/tracker.py)).
  - `wato_common.tracking` is shared with Step F of lidar_preprocessing, so
    any change there must stay opt-in.
- **Measured fragmentation**
  ([`docs/tracking_measurements.md`](../tracking_measurements.md)).
  - Scene-0061: 1.79 tracks per GT object, with 45% of objects in one track.
  - Scene-1100: 1.68 tracks per object, with 52% in one track.
  - Removing the shape gates gives 1.50 tracks per object, but the share of
    switched tracks rises to 0.09–0.11.
- **Why a new track starts on an already-tracked object** (measured in this
  repo, 2026-11-09). These are 152 breaks from the raw per-chunk tracker on
  nuScenes scenes 0061 and 1100.

  | Cause | Share | What happens |
  |---|---|---|
  | Shape gate | 39% | The new box is about 0.65 m from the track's last centre, with IoU ≈ 0 and a volume ratio outside the gate (one-sided slivers). It carries a new lifted-instance id, so the instance exemption does not apply. On scene-0061, 41% of these are a camera change. |
  | Duplicate | 38% | A second fused box on the same object in the same frame, about 0.9 m beside the box the track took, with IoU 0. It usually comes from the same camera, has a different instance id, and is mostly weak (strong share 0.16–0.23). |
  | Distance gate | 12% | The CV prediction is 2–6 m off. In half of these, the box carries an instance id the track already had. |
  | Coast expired | 7% | The gap is longer than 2 s. |

  - WATO proxy (validated on nuScenes): 27% of kept per-chunk tracks start
    beside a live track that the shape gate rejected, 10% are duplicates, and
    12% start within 2 m of a track that just expired. The median WATO track
    is 3 frames long.
  - A ByteTrack-style rule that weak boxes never start tracks is ruled out:
    45 of the 76 detected objects in scene-0061 never get a strong box.
- **Camera evidence on disk.**
  - Each masklet has one DINOv2 ViT-L/14 CLS embedding, averaged over its
    frames. It is computed on the mask's bounding-box crop after Resize(224)
    and CenterCrop(224)
    ([`embeddings.py`](../../src/perception_2d/src/wato_perception_2d/models/embeddings.py)).
  - Back-filled images have no embedding.
  - Masklets end after one image without a mask (`drop_lost_after_frames: 1`).
  - Track rows carry `supporting_masklet_ids`. Nothing reads the embeddings.
  - Measured in this repo, 2026-11-09:
    - The stored embedding separates the same object's masklets from nearby
      same-class objects with an AUC of 0.55.
    - Mask-weighted patch tokens, with max-over-frame-pairs banks, reach an
      AUC of 0.70 on scene-0061 and 0.81 on scene-1100.
    - At actual break decisions with another track within 3 m, the correct
      track is the more similar one in 50–67% of shape-gate breaks and
      14–32% of duplicates.

## Sources and pins

"No licence file" means the repository grants no licence: implement from the
paper and config values, and do not copy code.

| Source | Paper | Code at commit | Licence |
|---|---|---|---|
| 3D Auto Labeling (3DAL) | [2103.05073](https://arxiv.org/html/2103.05073) | not released | — |
| CTRL | [2304.12315](https://arxiv.org/html/2304.12315) | [SST@7c95376](https://github.com/tusen-ai/SST/tree/7c95376a893461aeefc8603880fa2f2a390fe38a); forward tracker [ImmortalTracker-for-CTRL@03f25eb](https://github.com/Abyssaledge/ImmortalTracker-for-CTRL/tree/03f25ebbe04cfa7f2e9f136d163c1558545c0f34) | Apache-2.0 (SST); none in fork |
| DetZero | [2306.06023](https://arxiv.org/html/2306.06023) | [DetZero@051fad7](https://github.com/PJLab-ADG/DetZero/tree/051fad709be900f79a23f37f0c0507e0cf84c54e) | Apache-2.0 |
| MS3D++ | [2308.05988](https://arxiv.org/html/2308.05988) | [MS3D@ffed761](https://github.com/darrenjkt/MS3D/tree/ffed761a6846183966cc38d3144c90d91446fa4a) | Apache-2.0 |
| Offline-Poly | [2602.13772](https://arxiv.org/html/2602.13772) | unreleased; the abstract says "Code will be released" | — |
| Poly-MOT / Fast-Poly | [2307.16675](https://arxiv.org/html/2307.16675) / [2403.13443](https://arxiv.org/html/2403.13443) | [Poly-MOT@5fead68](https://github.com/lixiaoyu2000/Poly-MOT/tree/5fead68f88ec41af71c789da0f52a6f11a290891), [FastPoly@694dec3](https://github.com/lixiaoyu2000/FastPoly/tree/694dec303db0831cbed4f8cac3f39dc1afe397ba) | no licence file |
| ImmortalTracker | [2111.13672](https://arxiv.org/html/2111.13672) | [ImmortalTracker@11f8855](https://github.com/ImmortalTracker/ImmortalTracker/tree/11f8855dead7a7feee8b5303f945d0283bbea431) | no licence file |
| SimpleTrack | [2111.09621](https://arxiv.org/html/2111.09621) | [SimpleTrack@05c96bb](https://github.com/tusen-ai/SimpleTrack/tree/05c96bb7ed98fc179856f327544612a66c839b5e) | MIT |
| CenterPoint | [2006.11275](https://arxiv.org/html/2006.11275) | [CenterPoint@3cf7d87](https://github.com/tianweiy/CenterPoint/tree/3cf7d870537e287c99b43b68636ea392a5e6f519) | MIT |
| MCTrack | [2409.16149](https://arxiv.org/html/2409.16149) | [MCTrack@a648ee2](https://github.com/megvii-research/MCTrack/tree/a648ee2763ea351a77ddb5232ca00d848d0e46c9) | no licence file |
| BiTrack | [2406.18414](https://arxiv.org/html/2406.18414) | [BiTrack@055cd6c](https://github.com/Kemo-Huang/BiTrack/tree/055cd6c1252adae1be4bcb9016a15d08031723e0) | MIT |
| Chiu et al. (Mahalanobis) | [2001.05673](https://arxiv.org/html/2001.05673) | [mahalanobis_3d_mot@2933888](https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/tree/293388853c70ab04b80a1d75bb69c5cd0e657f9b) | non-commercial academic |
| AB3DMOT | Weng et al. | [AB3DMOT@61f3bd7](https://github.com/xinshuoweng/AB3DMOT/tree/61f3bd72574093e367916c757b4747ca445f978c) | non-commercial academic |
| CasTrack | IEEE T-ITS paper not read | [3D-Multi-Object-Tracker@d37df41](https://github.com/hailanyi/3D-Multi-Object-Tracker/tree/d37df41c8e512e14c198cb18a4609ed66253d7a1) | Apache-2.0 |
| EagerMOT | [2104.14682](https://arxiv.org/html/2104.14682) | [EagerMOT@2792a76](https://github.com/aleksandrkim61/EagerMOT/tree/2792a76a044250e55e7f25b3e75cd5961366bd33) | MIT |
| StrongSORT (AFLink, GSI) | [2202.13514](https://arxiv.org/html/2202.13514) | [StrongSORT@ee99507](https://github.com/dyhBUPT/StrongSORT/tree/ee995076da5083e28d0da1f885297df62705ebd7) | GPL-3.0 |
| GIAOTracker | [2202.11983](https://arxiv.org/html/2202.11983) | not read | — |
| SUSHI | [2212.03038](https://arxiv.org/html/2212.03038) | [SUSHI@ff1952b](https://github.com/dvl-tum/SUSHI/tree/ff1952b408835007f07d1fc78760872625fa6ae4) | MIT |
| Neural message passing (MPN) | [1912.07515](https://arxiv.org/html/1912.07515) | not read | — |
| Batch3DMOT | [2203.10926](https://arxiv.org/html/2203.10926) | [Batch3DMOT@a846929](https://github.com/robot-learning-freiburg/Batch3DMOT/tree/a846929244b1cb143c9230fdd26ed6a82a5800ae) | GPL-3.0 |
| Offline tracking w/ object permanence (OTOP) | [2310.01288](https://arxiv.org/html/2310.01288) | [OTOP@7894f03](https://github.com/tudelft-iv-students/offline-tracking-with-object-permanence/tree/7894f031dc47d65ab8f99f87499abc67dce578f4) | MIT |
| BoT-SORT | [2206.14651](https://arxiv.org/html/2206.14651) | [BoT-SORT@2519854](https://github.com/NirAharon/BoT-SORT/tree/251985436d6712aaf682aaaf5f71edb4987224bd) | MIT |
| Deep OC-SORT / OC-SORT / ByteTrack | [2302.11813](https://arxiv.org/html/2302.11813) / [2203.14360](https://arxiv.org/html/2203.14360) / [2110.06864](https://arxiv.org/html/2110.06864) | [Deep-OC-SORT@6bb51d0](https://github.com/GerardMaggiolino/Deep-OC-SORT/tree/6bb51d027b137233f5c520b6fcc4f2ae387a6ba9), [OC_SORT@8462e7e](https://github.com/noahcao/OC_SORT/tree/8462e7e729a93ccd3bd995c0a79a890336cb3a0b), [ByteTrack@d1bf019](https://github.com/ifzhang/ByteTrack/tree/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99) | MIT |
| MASA | [2406.04221](https://arxiv.org/html/2406.04221) | [masa@c5472b9](https://github.com/siyuanliii/masa/tree/c5472b9c7615f35abdf1188cb1a0c5408fe50d66) | Apache-2.0 (code) |
| SAM2MOT | [2504.04519](https://arxiv.org/html/2504.04519) (v5) | [SAM2MOT@7bdae12](https://github.com/TripleJoy/SAM2MOT/tree/7bdae12c1ba27f5cb2d2ea069dda2d53a2d0247f) | Apache-2.0 |
| HOTA / TrackEval | [2009.07736](https://arxiv.org/html/2009.07736) | [TrackEval@12c8791](https://github.com/JonathonLuiten/TrackEval/tree/12c8791b303e0a0b50f753af204249e622d0281a) | MIT |
| nuScenes tracking eval | — | [nuscenes-devkit@b40adc4](https://github.com/nutonomy/nuscenes-devkit/tree/b40adc467b919192899405d9b77871afee8efa07) | devkit licence |

## A. Association costs and gates for partial, non-overlapping boxes

### Published cost functions and thresholds

| Method | Cost | Gate (published default) | Solver | Class gate | Where |
|---|---|---|---|---|---|
| SimpleTrack | 1 − GIoU (3D) | GIoU > −0.5 "for every category" on WOD and nuScenes (`asso_thres: giou: 1.5`) | Hungarian | per-class runs | [§4.3.1](https://arxiv.org/html/2111.09621#S4.SS3.SSS1); [vc_kf_giou.yaml L7–12](https://github.com/tusen-ai/SimpleTrack/blob/05c96bb7ed98fc179856f327544612a66c839b5e/configs/waymo_configs/vc_kf_giou.yaml#L7-L12), [nu_configs/giou.yaml L6–13](https://github.com/tusen-ai/SimpleTrack/blob/05c96bb7ed98fc179856f327544612a66c839b5e/configs/nu_configs/giou.yaml#L6-L13); GIoU code [geometry.py L207–229](https://github.com/tusen-ai/SimpleTrack/blob/05c96bb7ed98fc179856f327544612a66c839b5e/mot_3d/utils/geometry.py#L207-L229) |
| ImmortalTracker | 1 − IoU (WOD), 1 − GIoU (nuScenes) | IoU ≥ 0.1 or GIoU ≥ −0.5 | Hungarian | — | [§4.3](https://arxiv.org/html/2111.13672#S4.SS3); [nu immortal.yaml](https://github.com/ImmortalTracker/ImmortalTracker/blob/11f8855dead7a7feee8b5303f945d0283bbea431/configs/nu_configs/immortal.yaml#L10-L15) |
| Poly-MOT | 1 − GIoU; `giou_bev` for bus, `giou_3d` for the others | first stage `1 − GIoU ≤` car 1.31, ped 1.69, bicycle 1.55, motorcycle 1.44, bus 1.34, trailer 1.25, truck 1.21; second stage 1.0 (`giou_bev`) | Hungarian | yes | [nusc_config.yaml L21–27](https://github.com/lixiaoyu2000/Poly-MOT/blob/5fead68f88ec41af71c789da0f52a6f11a290891/config/nusc_config.yaml#L21-L27) |
| Fast-Poly | 1 − A-GIoU (axis-aligned GIoU; `a_giou_bev` for bus, `a_giou_3d` for the others) | first stage: car 1.21, ped 1.78, bicycle 1.62, motorcycle 1.6, bus 1.16, trailer 1.16, truck 1.22; second stage 1.0 | Hungarian | yes | [Eq. 2, §III-B](https://arxiv.org/html/2403.13443#S3.SS2); [nusc_config.yaml L29–37](https://github.com/lixiaoyu2000/FastPoly/blob/694dec303db0831cbed4f8cac3f39dc1afe397ba/config/nusc_config.yaml#L29-L37); [`a_giou_3d` L241–302](https://github.com/lixiaoyu2000/FastPoly/blob/694dec303db0831cbed4f8cac3f39dc1afe397ba/geometry/nusc_distance.py#L241-L302) |
| MCTrack | 1 − Ro_GDIoU, where Ro_GDIoU = IoU − w1·(hull − union)/hull − w2·d²/diag², using cv2's minimum-area hull | `1 − Ro_GDIoU ≤` car 1.10, ped 2.06, bicycle 2.00, motorcycle 2.06, bus 1.60, trailer 1.26, truck 1.16; (w1, w2) = (0.5, 1.5) for car, (1, 1) otherwise; second stage in range view with SDIoU, greedy, threshold −0.3 | Hungarian, then greedy | yes (cost −inf) | [§3.2.3](https://arxiv.org/html/2409.16149#S3.SS2.SSS3); [nuscenes.yaml L27–63](https://github.com/megvii-research/MCTrack/blob/a648ee2763ea351a77ddb5232ca00d848d0e46c9/config/nuscenes.yaml#L27-L63); [cost_function.py L34–97](https://github.com/megvii-research/MCTrack/blob/a648ee2763ea351a77ddb5232ca00d848d0e46c9/tracker/cost_function.py#L34-L97) |
| CenterPoint | BEV centre distance, after moving the detection back by its predicted velocity | car 4, truck 4, bus 5.5, trailer 3, ped 1, motorcycle 13, bicycle 3 (m) | greedy (Hungarian optional) | yes | [§4](https://arxiv.org/html/2006.11275#S4); [pub_tracker.py L20–31, L83–102](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/tools/nusc_tracking/pub_tracker.py#L20-L102) |
| Chiu et al. | Mahalanobis distance, √(yᵀS⁻¹y), on the 7-dimensional measurement | 11 | greedy | per-class runs | [§3.4, Eq. 6](https://arxiv.org/html/2001.05673#S3.SS4); the value 11 is only in [run.sh L93](https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/293388853c70ab04b80a1d75bb69c5cd0e657f9b/run.sh#L93) |
| EagerMOT | ‖Δ[x,y,z,h,w,l]‖ · (2 − cos Δθ) | car −7.5, ped −1.8, bus −8.15, … (`thresholds_per_class`, nuScenes) | greedy | yes | [§III-B, Eq. 1–2](https://arxiv.org/html/2104.14682#S3.SS2); [params.py L74–88](https://github.com/aleksandrkim61/EagerMOT/blob/2792a76a044250e55e7f25b3e75cd5961366bd33/configs/params.py#L74-L88) |
| BiTrack | normalised centre distance, 1 − d(c, ĉ) / max vertex distance | β = 0.5 | Hungarian | — | [§IV-B1, Eq. 11](https://arxiv.org/html/2406.18414#S4.SS2.SSS1); [§V-B](https://arxiv.org/html/2406.18414#S5.SS2) |
| 3DAL | BEV IoU | IoU ≥ 0.1, after dropping detections scoring < 0.1 | Hungarian | — | [App. D](https://arxiv.org/html/2103.05073#A4) |
| DetZero | BEV IoU | see "Two-stage and low-score association" below | GNN | yes | [App. C](https://arxiv.org/html/2306.06023#A3) |
| MS3D++ (fork of SimpleTrack) | 1 − GIoU, or 1 − BEV IoU for static vehicles | vehicle GIoU ≥ −0.3; pedestrian ≥ −0.5; static vehicle BEV IoU ≥ 0.3 | Hungarian | per-class runs | [veh_kf_giou.yaml](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tracker/configs/nuscenes/ms3d_configs_rnd1/veh_kf_giou.yaml#L1-L19), [ped_kf_giou.yaml](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tracker/configs/nuscenes/ms3d_configs_rnd1/ped_kf_giou.yaml#L1-L21), [veh_static_kf_iou2d.yaml](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tracker/configs/nuscenes/ms3d_configs_rnd1/veh_static_kf_iou2d.yaml#L1-L19) |
| AB3DMOT (nuScenes, CenterPoint detections) | 1 − GIoU (3D), or centre distance for bicycle | car −0.4, ped −0.5, truck −0.4, trailer −0.3, bus −0.4, motorcycle −0.7; bicycle `dist_3d` 6 | greedy | per-class runs | [model.py L64–73](https://github.com/xinshuoweng/AB3DMOT/blob/61f3bd72574093e367916c757b4747ca445f978c/AB3DMOT_libs/model.py#L64-L73) |

What the sources say about these choices:

- **Why GIoU instead of IoU or distance (SimpleTrack).** IoU "cannot connect
  the detections and motion predictions if the IoU between them are all
  zeros". Distance metrics "lack discrimination on orientations"
  ([§4.3.1](https://arxiv.org/html/2111.09621#S4.SS3.SSS1)). IoU and GIoU work
  with Hungarian and greedy matching alike; Mahalanobis and L2 "demand greedy
  algorithm" ([§4.3.2](https://arxiv.org/html/2111.09621#S4.SS3.SSS2)).
- **Why A-GIoU (Fast-Poly).** The motivation is speed: overlap and hull of
  rotated boxes cost O(n²) and O(n log n), while axis-aligned boxes cost O(n)
  ([§III-B](https://arxiv.org/html/2403.13443#S3.SS2)). In its ablation,
  A-GIoU keeps AMOTA at 73.1, raises MOTA by 0.5 and gives 799 fewer FN
  ([§IV-D](https://arxiv.org/html/2403.13443#S4.SS4)). The metric ignores
  heading by construction.
- **Score threshold in MCTrack.** The cost matrix is `1 − Ro_GDIoU`, and pairs
  of different classes get similarity −inf
  ([matching.py L163–211](https://github.com/megvii-research/MCTrack/blob/a648ee2763ea351a77ddb5232ca00d848d0e46c9/tracker/matching.py#L163-L211)).
- **CenterPoint's thresholds** are commented as the "99.9 percentile of the l2
  velocity error distribution (per class / 0.5 second)" of its own velocity
  head, followed by "Tune this for your model"
  ([pub_tracker.py L20–31](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/tools/nusc_tracking/pub_tracker.py#L20-L31)).
- **Chiu's S** comes from Q and R estimated as statistics of detector error
  on the nuScenes training set ([§3.6](https://arxiv.org/html/2001.05673#S3.SS6)).
  Before computing the distance, Chiu rotates the prediction by 180° whenever
  the angle difference is between 90° and 270°
  ([§3.4](https://arxiv.org/html/2001.05673#S3.SS4)).
- **3DAL's heading handling.** 3DAL adds π to a detection's heading when the
  difference to the track is obtuse, and averages angles in cyclic space
  ([App. D](https://arxiv.org/html/2103.05073#A4)).
- **Range-view second stage (MCTrack).** The second stage re-matches
  leftovers in the projected range view rather than lowering a threshold
  ([§3.2.3](https://arxiv.org/html/2409.16149#S3.SS2.SSS3)). EagerMOT's second
  stage likewise matches remaining tracks to 2D detections by the 2D IoU of
  the projected predicted box ([§III-B](https://arxiv.org/html/2104.14682#S3.SS2)).

### Two-stage and low-score association

- **SimpleTrack.**
  - Detections above T_h are matched first. Unmatched tracklets may then
    take detections above T_l; the paper's example for pedestrians on WOD is
    T_l = 0.1, T_h = 0.5.
  - A second-stage match keeps the track alive. It is "not used for updating
    motion models", is not output, and the motion prediction stands in as the
    latest state ([§4.4](https://arxiv.org/html/2111.09621#S4.SS4)).
  - In code, the second stage returns `pred_bbox` with `update_mode = 3`
    ([redundancy.py L32–66](https://github.com/tusen-ai/SimpleTrack/blob/05c96bb7ed98fc179856f327544612a66c839b5e/mot_3d/redundancy/redundancy.py#L32-L66)).
  - On WOD the second-stage gate is GIoU > −0.5 at score > 0.1.
- **ByteTrack.**
  - τ = 0.6; IoU below 0.2 is rejected; lost tracks are kept 30 frames
    ([§4.1](https://arxiv.org/html/2110.06864#S4.SS1)).
  - Re-ID features of low-score boxes "are not reliable", so the second
    association uses IoU ([§4.2](https://arxiv.org/html/2110.06864#S4.SS2)).
- **DetZero.**
  - The appendix says: a high group (score > 0.1 and more than 3 points for
    vehicles, 1 for pedestrians and cyclists) matched at BEV IoU
    (0.3, 0.15, 0.15), then a low group at (0.2, 0.1, 0.1). A matched box
    replaces the track state "rather than updating them through Kalman
    filtering" ([App. C](https://arxiv.org/html/2306.06023#A3)).
  - The shipped config inverts this: `FIRST_STAGE` uses (0.2, 0.1, 0.1) and
    `SECOND_STAGE` uses (0.3, 0.15, 0.15), and `POINT_THRESHOLD` is
    (0, 0, 0) ([waymo_detzero_track.yaml L23–42](https://github.com/PJLab-ADG/DetZero/blob/051fad709be900f79a23f37f0c0507e0cf84c54e/tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track.yaml#L23-L42)).
- **MS3D++.**
  - The second stage ("redundancy") accepts score > 0.1, at GIoU ≥ −0.3 for
    vehicles and −0.5 for pedestrians, for up to `MAX_REDUNDANCY_AGE` 3
    frames ([ps_config.yaml L14–47](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tools/cfgs/target_nuscenes/label_generation/round1/cfgs/ps_config.yaml#L14-L47)).
  - The static tracker's strict BEV IoU is commented "if it matches with a
    box really far away, it'll shift the KF prediction and the car will be
    'moving'".

## B. Offline tracklet re-linking after a first pass

### Offline-Poly (learning-free, CPU; code unreleased)

Offline-Poly runs on the output of one or more upstream trackers
([§III](https://arxiv.org/html/2602.13772#S3), [§IV-A](https://arxiv.org/html/2602.13772#S4.SS1)).

- **Pre-processing.** A tracklet is removed when its age *and* mean
  confidence are *both* below θ_age and θ_score
  ([§IV-B](https://arxiv.org/html/2602.13772#S4.SS2)). This is the rule
  `filter` already implements.
- **STWO: re-linking tracklets whose lifecycles do not overlap**
  ([§IV-C](https://arxiv.org/html/2602.13772#S4.SS3)). The procedure
  iterates frame by frame:
  1. At frame t, each tracklet without an observation gets a predicted state
     (Eq. 2): forward from its last state after it ends, backward from its
     first state before it starts, and the mean of both predictions inside a
     gap. "Predictions exceeding 1 second are discarded to prevent drift."
     Predictions use explicit motion models: CV and CTRA, following
     Fast-Poly on nuScenes and CV on KITTI
     ([§V-B](https://arxiv.org/html/2602.13772#S5.SS2)).
  2. The cost is C = 1 − IoU, in BEV or 3D, of the two tracklets' observed
     or predicted states at t (Eq. 3). Self-pairs and "costs involving
     invalid states" are set to infinity.
  3. The pairs are solved with the blossom algorithm, a general-graph
     one-to-one matching, keeping only matches with cost < θ_blo.
  4. Each matched pair is fused under a new ID, its gaps are interpolated
     with Eq. 2, and the whole procedure is "applied iteratively".
- **STW: tracklets that overlap in time.** Tracklets are clustered through a
  frame-level gIoU adjacency, using the multi-tracker module of §IV-D and a
  DFS. They are split at the frames where they are connected, entangled nodes
  are merged, and the pieces are re-linked with STWO
  ([§IV-C](https://arxiv.org/html/2602.13772#S4.SS3)).
- **Parameters.**
  - θ_blo: "optimal performance occurs under a moderately relaxed threshold
    (around 0.9)"; θ_blo = 1 "leads to FP matching". STWO is "generally
    insensitive to threshold variations"
    ([§V-E](https://arxiv.org/html/2602.13772#S5.SS5)).
  - All Offline-Poly parameters were set by "a linear search … to maximize
    AMOTA on the validation set" ([§V-B](https://arxiv.org/html/2602.13772#S5.SS2)).
    No table lists the final θ_age, θ_score, θ_blo, θ_multi, K or M.
- **Results.**
  - On the nuScenes val set (CenterPoint detections, forward Fast-Poly
    tracker), the STWO metric comparison
    ([Table VIII](https://arxiv.org/html/2602.13772#S5.T8)) is:

    | STWO metric | MOTA | IDS |
    |---|---|---|
    | IoU | 65.0 | 251 |
    | gIoU (3D) | 64.7 | 380 |
    | A-gIoU (3D) | 64.8 | 380 |
    | OTOP, learned | 64.1 | 367 |

  - The [Table III](https://arxiv.org/html/2602.13772#S4.T3) baseline is IDS
    414. Table VIII does not state which other modules were active.
  - Adding the matching modules to the age and score filters plus refinement
    moves IDS from 387 to 372/373, and AMOTA from 74.6 to 74.7/74.8 (Table
    III, EXP5 to EXP6/EXP8).
  - Unspecified: how per-frame matches across a tracklet's frames combine
    into one decision, what makes a state "invalid", and when iteration
    stops.

### Other re-linkers

| Method | Unit and cost | Gates and defaults | Solver | Training-free | Note |
|---|---|---|---|---|---|
| BiTrack | Runs forward and backward over the same detections, then clusters trajectories that share a detection by BFS, splits them at common links and selects fragments with no frame overlap by the priority h(F), favouring fragments common to both directions or from longer trajectories ([§IV-C1–2](https://arxiv.org/html/2406.18414#S4.SS3.SSS1), [Eq. 8–9](https://arxiv.org/html/2406.18414#S3)) | θ_hit = 6; θ_miss 5 (new) / 28 (confirmed) frames; β = 0.5 ([§V-B](https://arxiv.org/html/2406.18414#S5.SS2)) | greedy | yes | Bidirectional fusion gives HOTA 83.55 (forward) → 83.59 on KITTI ([§V-D, Table VI](https://arxiv.org/html/2406.18414#S5.SS4)) |
| DetZero | Re-runs tracking in reverse order. In code, it seeds each forward track at its first box with Δt = −0.1 s, uses second-stage matching only, and prepends the result ([track_manager.py L109–142, L218–260](https://github.com/PJLab-ADG/DetZero/blob/051fad709be900f79a23f37f0c0507e0cf84c54e/tracking/detzero_track/models/tracking_modules/track_manager.py#L109-L260)) | see §A | GNN | yes | The paper also mentions "location-aware similarity" and WBF fusion ([§3.2](https://arxiv.org/html/2306.06023#S3.SS2)); neither appears in the tracking code at this commit |
| CTRL | Extension only: ImmortalTracker forward, then backtraces each track and extends it into the past ([§3.2](https://arxiv.org/html/2304.12315#S3.SS2)) | paper: tracks over 100 frames are extended to the sequence ends, others by 20 frames ([App. D](https://arxiv.org/html/2304.12315#A4)); code: backward `extend_length: 10`, `min_length_to_extend: 3`, `score_multiplier: 0.01` ([extend.yaml](https://github.com/tusen-ai/SST/blob/7c95376a893461aeefc8603880fa2f2a390fe38a/tools/ctrl/data_configs/extend.yaml#L1-L6)) | — | yes | The added boxes "are not accurate, leading to many false positives"; CTRL's learned refiner handles them ([§3.2](https://arxiv.org/html/2304.12315#S3.SS2)) |
| AFLink (StrongSORT) | The last 30 frames of (frame, x, y) of two tracklets go into a conv net that predicts a link probability ([§IV-A](https://arxiv.org/html/2202.13514#S4.SS1)) | thrT (0, 30) frames, thrS 75 px, thrP 0.05 ([strong_sort.py L35–44](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/strong_sort.py#L35-L44); [AppFreeLink.py L72–108](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/AFLink/AppFreeLink.py#L72-L108)) | Hungarian | no (trained on MOT17, 2D pixels) | "helpless against false association problems" ([§V-E](https://arxiv.org/html/2202.13514#S5.SS5)) |
| GSI (StrongSORT) | Linear interpolation, then Gaussian-process regression with λ = τ·log(τ³/l) ([§IV-B](https://arxiv.org/html/2202.13514#S4.SS2)) | gaps up to 20 frames, τ = 10 ([§V-A](https://arxiv.org/html/2202.13514#S5.SS1)); code clips λ to [1/τ, τ²] ([GSI.py L37–43](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/GSI.py#L37-L43)) | — | yes | Fills gaps; does not link tracklets |
| GIAOTracker global link | Appearance cost C_a = minimum cosine distance between two tracklets' feature banks, plus time and space costs (Eq. 5–7, [§3.2](https://arxiv.org/html/2202.11983#S3.SS2)) | Th_a 0.4, Th_t 200 frames, Th_s 150 (pixels); λ_a 40, λ_t 1, λ_s 1 ([§4.2](https://arxiv.org/html/2202.11983#S4.SS2)) | Hungarian | no (learned GIModel) | All three thresholds must pass before any cost is computed |
| SUSHI | Hierarchical GNN over tracklets: 9 levels, 2^l frames per level, 512 frames, 10 neighbours each ([§5.2](https://arxiv.org/html/2212.03038#S5.SS2)) | learned | edge classifier | no | Clip stitching by shared detections ([App. A.2](https://arxiv.org/html/2212.03038#A1.SS2)) is the link `stitch.py` already borrows |
| MPN | Learned edge classifier over a simplified min-cost-flow formulation ([§3.2](https://arxiv.org/html/1912.07515#S3.SS2)) | learned | — | no | — |
| Batch3DMOT | Multimodal GNN (camera, LiDAR, radar) over a 5-frame sliding window ([§III-B](https://arxiv.org/html/2203.10926#S3.SS2)) | learned on nuScenes | — | no | GPL-3.0 |
| OTOP | Learned motion and lane-map affinities, C = w·C_map + (1 − w)·C_motion with w = 0.5 ([§III-A4](https://arxiv.org/html/2310.01288#S3.SS1.SSS4)) | needs a vectorised lane map | Hungarian | no | Offline-Poly's IoU STWO beats it (Table VIII above) |

### How the references gate appearance at the link step

- GIAOTracker requires all three thresholds (appearance, time and space)
  before it computes a link cost ([§3.2](https://arxiv.org/html/2202.11983#S3.SS2)).
- BoT-SORT counts appearance only when the IoU distance is below θ_iou, and
  then takes the minimum of the two distances (§E).
- OTOP and AFLink use learned scores. AFLink deliberately uses no appearance
  because "overreliance on appearance features can be vulnerable to
  occlusion" ([§IV-A](https://arxiv.org/html/2202.13514#S4.SS1)).
- Offline-Poly and BiTrack use geometry only.

## C. Concurrent duplicates: boxes and tracks on one object at once

### Detection-level suppression inside trackers

| Method | Rule | Defaults | Where |
|---|---|---|---|
| CenterPoint (circle NMS, in the detector head) | Within a task head, keep the highest score and suppress any box whose squared BEV centre distance is ≤ `min_radius` | `[4, 12, 10, 1, 0.85, 0.175]` for heads [car], [truck, construction], [bus, trailer], [barrier], [motorcycle, bicycle], [pedestrian, cone], which is a radius of 2.0, 3.46, 3.16, 1.0, 0.92 and 0.42 m | [circle_nms_jit.py L4–28](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/core/utils/circle_nms_jit.py#L4-L28) (squared distance); [config L70–73](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms.py#L70-L73); not described in the paper text |
| SimpleTrack | IoU NMS on detections | nuScenes 0.1, WOD 0.25 | [nu giou.yaml L24–27](https://github.com/tusen-ai/SimpleTrack/blob/05c96bb7ed98fc179856f327544612a66c839b5e/configs/nu_configs/giou.yaml#L24-L27) |
| ImmortalTracker | IoU NMS | nuScenes 0.1, WOD 0.25 | [immortal.yaml L17–19](https://github.com/ImmortalTracker/ImmortalTracker/blob/11f8855dead7a7feee8b5303f945d0283bbea431/configs/nu_configs/immortal.yaml#L17-L19) |
| Poly-MOT | Per-class score filter, then greedy NMS on BEV IoU; boxes of different classes get −inf, so they never suppress each other | `NMS_thre 0.08` | [nusc_config.yaml L15–19](https://github.com/lixiaoyu2000/Poly-MOT/blob/5fead68f88ec41af71c789da0f52a6f11a290891/config/nusc_config.yaml#L15-L19); [nusc_nms.py L17–40](https://github.com/lixiaoyu2000/Poly-MOT/blob/5fead68f88ec41af71c789da0f52a6f11a290891/pre_processing/nusc_nms.py#L17-L40) |
| Fast-Poly ("scale NMS", code only) | Per class, scale w/l/h by a factor, then NMS | threshold 0.08 for every class; scale bicycle 1.4, pedestrian 1.1, others 1; metric `a_giou_bev` for bicycle, bus, pedestrian, truck and `iou_bev` otherwise | [nusc_config.yaml L16–27](https://github.com/lixiaoyu2000/FastPoly/blob/694dec303db0831cbed4f8cac3f39dc1afe397ba/config/nusc_config.yaml#L16-L27); [nusc_nms.py L66–106](https://github.com/lixiaoyu2000/FastPoly/blob/694dec303db0831cbed4f8cac3f39dc1afe397ba/pre_processing/nusc_nms.py#L66-L106) |
| DetZero | Overlap ratio = intersection / area of the other box, to remove boxes "completely wrapped by other boxes" | BEV ratio 0.3 for vehicles, 0.2 for pedestrians and cyclists | [App. C](https://arxiv.org/html/2306.06023#A3); effect in [Table 17](https://arxiv.org/html/2306.06023#A6) |
| CTRL | The base detector's own NMS | IoU 0.2; the tracker's extra NMS is disabled | [App. D](https://arxiv.org/html/2304.12315#A4) |
| 3DAL | Score filter only | detections with score < 0.1 are dropped | [App. D](https://arxiv.org/html/2103.05073#A4) |

### Track-level merging

- **DetZero.**
  - Every frame, same-class live tracks whose current BEV boxes overlap at
    least a class threshold of a track's own area are reduced to the lowest
    track id, "keeping the earlier birth object ID"; the others are dropped.
    The thresholds are vehicle 0.5, pedestrian 0.4 and cyclist 0.4
    ([App. C](https://arxiv.org/html/2306.06023#A3);
    [track_manager.py L262–310](https://github.com/PJLab-ADG/DetZero/blob/051fad709be900f79a23f37f0c0507e0cf84c54e/tracking/detzero_track/models/tracking_modules/track_manager.py#L262-L310)).
  - Post-processing deletes tracks with fewer than 5 hits and trims unmatched
    predicted boxes at both ends
    ([post_process.py L23–53](https://github.com/PJLab-ADG/DetZero/blob/051fad709be900f79a23f37f0c0507e0cf84c54e/tracking/detzero_track/models/tracking_modules/post_process.py#L23-L53)).
- **Offline-Poly.**
  - STW, described in §B.
  - Multi-tracker matching ([§IV-D](https://arxiv.org/html/2602.13772#S4.SS4)).
    The frame-level adjacency is `1(C < θ_multi)`, with C = 1 − IoU
    (Eq. 6). It is max-pooled over time, with the authors noting
    "susceptibility to spurious edge responses" (Eq. 8), then clustered by
    DFS (Eq. 9).
  - Each cluster is fused into one tracklet by confidence-weighted averaging
    of the valid observations in each frame, under "category-consistent
    association".
  - θ_multi "favors a relatively relaxed threshold (> 0.7)"
    ([§V-E](https://arxiv.org/html/2602.13772#S5.SS5)).
- **MS3D++.** Static-vehicle, tracked-vehicle, dynamic-pedestrian and
  ensemble boxes are added per frame and filtered by NMS
  ([§IV-E](https://arxiv.org/html/2308.05988#S4.SS5)). The code defaults are
  `veh_nms_th=0.05` and `ped_nms_th=0.5`
  ([temporal_refinement.py L54–58](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tools/temporal_refinement.py#L54-L58)).
- **GIAOTracker** "denoising" applies Soft-NMS between whole trajectories
  using temporal IoU, on 2D boxes
  ([§3.3](https://arxiv.org/html/2202.11983#S3.SS3)).

Every IoU- or overlap-based rule above needs the boxes to overlap. Fast-Poly's
scaling enlarges boxes before NMS. CenterPoint's circle NMS is the only rule
in these sources that suppresses boxes with no overlap at all, and it acts
only within one task head. No source merges two boxes of different classes on
one object, such as a pedestrian and a motorcycle for one rider.

## D. Coasting and never-terminating tracks

- **ImmortalTracker.**
  - It "always maintains tracklets for objects even if they are invisible".
    The predictions of unmatched tracklets are not output, so they "will not
    cause false positives" ([§3](https://arxiv.org/html/2111.13672#S3)).
  - Motion model: a vanilla 3D Kalman filter. Association: IoU ≥ 0.1 or
    GIoU ≥ −0.5 ([§4.3](https://arxiv.org/html/2111.13672#S4.SS3)).
  - In code, `immortal` skips every `dead` transition
    ([hit_manager.py L25–28, L73–84](https://github.com/ImmortalTracker/ImmortalTracker/blob/11f8855dead7a7feee8b5303f945d0283bbea431/mot_3d/life/hit_manager.py#L25-L84)).
    `min_hits_to_birth: 1`; the score threshold is 0.01 on nuScenes and 0.5
    on WOD.
  - Evidence: on WOD val, CenterPoint has 2,891 vehicle IDS, of which 2,890
    are early terminations; ImmortalTracker has 114
    ([Table 1](https://arxiv.org/html/2111.13672#S1.T1)).
  - Limit: the authors "can not verify the effectiveness … in long-term
    (≫ 20 seconds)" scenarios ([§5](https://arxiv.org/html/2111.13672#S5)).
- **SimpleTrack.** Early termination causes 95% of vehicle and 91% of
  pedestrian ID switches on WOD
  ([§4.4](https://arxiv.org/html/2111.09621#S4.SS4)). After two-stage
  association, 88% and 72% of the remaining switches still come from it
  ([§6.2](https://arxiv.org/html/2111.09621#S6.SS2)).
- **Life-cycle defaults:**

| Method | Coast before termination | Birth | Notes |
|---|---|---|---|
| ImmortalTracker | never | 1 hit | predictions not output |
| CTRL's fork | never | 0 hits | outputs predicted boxes up to 10 frames after the last update (`merge: max_time_since_update: 10`, [config](https://github.com/Abyssaledge/ImmortalTracker-for-CTRL/blob/03f25ebbe04cfa7f2e9f136d163c1558545c0f34/configs/waymo_configs/immortal_for_ctrl_keep10.yaml#L1-L26)) |
| DetZero | never (`DEATH_AGE: -1`) | 1 | trims unmatched boxes at the ends afterwards |
| SimpleTrack | 2 missed frames | 3 (WOD), 0 (nuScenes) | [App. A](https://arxiv.org/html/2111.09621#S1.SS2.SSS0.Px1) |
| Poly-MOT / Fast-Poly | 10 frames at 2 Hz for bicycle, bus, pedestrian and trailer; 15 for car; 20 for motorcycle and truck | 1 | scores of unmatched tracks decay; only `punish_num: 1` predicted frame is output ([§III-D](https://arxiv.org/html/2307.16675#S3.SS4); [config L38–45](https://github.com/lixiaoyu2000/Poly-MOT/blob/5fead68f88ec41af71c789da0f52a6f11a290891/config/nusc_config.yaml#L38-L45)) |
| BiTrack | 5 frames (new) / 28 frames (confirmed) at 10 Hz | 6 | [§V-B](https://arxiv.org/html/2406.18414#S5.SS2) |
| CasTrack | 12 frames at 10 Hz (4 for new tracks) | — | [casa_mot.yaml L20–22](https://github.com/hailanyi/3D-Multi-Object-Tracker/blob/d37df41c8e512e14c198cb18a4609ed66253d7a1/config/global/casa_mot.yaml#L20-L22) |
| CenterPoint | 3 frames, advanced by the track's last velocity | 1 | [§4](https://arxiv.org/html/2006.11275#S4) |
| Chiu / AB3DMOT | 2 missed frames | 3 consecutive | [§3.5](https://arxiv.org/html/2001.05673#S3.SS5) |
| ByteTrack / OC-SORT | 30 frames | — | [§4.1](https://arxiv.org/html/2110.06864#S4.SS1); [ocsort.py L180–181](https://github.com/noahcao/OC_SORT/blob/8462e7e729a93ccd3bd995c0a79a890336cb3a0b/trackers/ocsort_tracker/ocsort.py#L180-L181) |
| Chen et al. (current base) | n_old = 5 | — | [§III-F](https://arxiv.org/html/2201.04501#S3.SS6) |

- **Re-acquiring after a gap.**
  - OC-SORT's ORU: when a track is re-activated, it replays KF predict and
    update along a virtual trajectory that interpolates linearly between the
    last observation before the gap and the new one
    ([§4.1](https://arxiv.org/html/2203.14360#S4.SS1); CV interpolation per
    [§5.1](https://arxiv.org/html/2203.14360#S5.SS1);
    [kalmanfilter.py L383–432](https://github.com/noahcao/OC_SORT/blob/8462e7e729a93ccd3bd995c0a79a890336cb3a0b/trackers/ocsort_tracker/kalmanfilter.py#L383-L432)).
  - OC-SORT's OCM adds λ·(direction difference) to the cost, with Δt = 3 and
    λ = 0.2. OCR is a second association against each unmatched track's last
    observation ([§4.2](https://arxiv.org/html/2203.14360#S4.SS2)).
  - BiTrack re-initialises velocity after a track's first match as
    displacement divided by elapsed frames, which gave "small improvements"
    ([§IV-B3](https://arxiv.org/html/2406.18414#S4.SS2.SSS3),
    [§V-D](https://arxiv.org/html/2406.18414#S5.SS4)).

## E. Appearance re-identification (brief)

| Method | Embedding and aggregation | Combined with motion | Gates and defaults |
|---|---|---|---|
| BoT-SORT | EMA with α = 0.9, updated from high-confidence detections only ([§3.3](https://arxiv.org/html/2206.14651#S3.SS3); [bot_sort.py L33–43](https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/bot_sort.py#L33-L43)) | C = min(d_iou, d̂_cos) (Eq. 13) | d̂_cos = 0.5·d_cos only if d_cos < θ_emb = 0.25 and d_iou < θ_iou = 0.5, else 1 (Eq. 12). θ_emb was read off a positive/negative histogram for *its* FastReID model ([Fig. 5](https://arxiv.org/html/2206.14651#S3.F5)). The code halves the distance before comparing to 0.25 ([L310–314](https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/bot_sort.py#L310-L314)), which differs from the paper. |
| StrongSORT | EMA with α = 0.9 ([§III-B](https://arxiv.org/html/2202.13514#S3.SS2)) | C = λA_a + (1 − λ)A_m, λ = 0.98 | matching distance 0.45 ([§V-A](https://arxiv.org/html/2202.13514#S5.SS1)) |
| Deep OC-SORT | EMA with α_t = α_f + (1 − α_f)(1 − (s − σ)/(1 − σ)), α_f = 0.95 ([§3.3](https://arxiv.org/html/2302.11813#S3.SS3)) | C = IoU + [a_w + w_b]·A_c ([§3.4](https://arxiv.org/html/2302.11813#S3.SS4)) | a_w = 0.75, ε = 0.5 (MOT17/20) ([§4](https://arxiv.org/html/2302.11813#S4)) |
| GIAOTracker | EMA bank, L_b = 100, α = 0.9 | link cost (§B) | Th_a = 0.4 |
| MASA | Adapter on frozen SAM, Detic or GDINO features, trained on SA-1B-500K raw images with SAM masks and no tracking labels ([§4.1](https://arxiv.org/html/2406.04221#S4.SS1)); memory update (1 − m)·old + m·new with m = 0.8 | score = (bi-softmax + cosine) / 2, times a soft distance mask | `match_score_thr 0.5`, `memo_tracklet_frames 10`, `init_score_thr 0.1` ([config L177–187](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/configs/masa-gdino/masa_gdino_swinb_inference.py#L177-L187); [tracker L113–117, L286–323](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/masa/models/tracker/masa_tao_tracker.py#L113-L323)) |

- **MASA's own caution.** Under MASA's training, DINO ViT-B features trained
  on BDD+COCO reach AssocA 23.1 on BDD MOT, against 51.9 for MASA
  ([Table 7](https://arxiv.org/html/2406.04221#S4.T7)). "Directly using the
  off-the-shelf SAM features … yields poor results"
  ([§4.3](https://arxiv.org/html/2406.04221#S4.SS3)). This was DINO v1, not
  DINOv2.
- **DINOv2's own evidence.** DINOv2 evaluates instance-level retrieval by
  cosine similarity on landmarks, Met and AmsterTime. It does not evaluate
  object re-identification ([§7.3](https://arxiv.org/html/2304.07193#S7.SS3)).
- **DINO-MOT** (IEEE RA-L 2025,
  [doi:10.1109/LRA.2024.3500882](https://doi.org/10.1109/LRA.2024.3500882))
  uses DINOv2 for pedestrian re-identification in 3D MOT. Only the
  bibliographic record was verified; method and results were not read.
- **No camera appearance.**
  - SAM2MOT has no ReID; it keeps a lost object for 25 frames. Long
    occlusions are its stated weakness, and it suggests appearance for
    implicit re-identification ([§4.2, Limitations](https://arxiv.org/html/2504.04519)).
  - EagerMOT uses no appearance and leaves "a (learned) appearance model" as
    future work ([§III-B](https://arxiv.org/html/2104.14682#S3.SS2)).
  - DeepFusionMOT's feature code is commented out at the pinned commit.
  - CAMO-MOT's appearance heads are trained with ground-truth association
    ([§III-D](https://arxiv.org/html/2209.02540#S3.SS4)); no code link was
    found.
- **Not found:** a primary source that evaluates training-free, mask-pooled
  DINOv2 features for tracklet re-identification, or training-free
  point-cloud shape ReID.

## F. Motion models (brief)

- **Poly-MOT.** CTRA for car-like objects and pedestrians couples heading,
  velocity and acceleration "on the same straight line". The bicycle model
  is "susceptible to erroneous predictions caused by incorrect object
  structure" ([§III-B](https://arxiv.org/html/2307.16675#S3.SS2)).
- **MCTrack** uses CV for all seven nuScenes classes
  ([nuscenes.yaml L75–76](https://github.com/megvii-research/MCTrack/blob/a648ee2763ea351a77ddb5232ca00d848d0e46c9/config/nuscenes.yaml#L75-L76)).
- **Fast-Poly** takes z, w, l and h out of the KF into a median filter over
  the last 3 observations. This cost 0.1 MOTA
  ([§III-C](https://arxiv.org/html/2403.13443#S3.SS3),
  [§IV-A](https://arxiv.org/html/2403.13443#S4.SS1)).
- **MCTrack's "GLOBAL" mode** and **CasTrack's "global" mode** only smooth
  and interpolate or re-score tracks. They do not re-link
  ([main.py L57–85](https://github.com/megvii-research/MCTrack/blob/a648ee2763ea351a77ddb5232ca00d848d0e46c9/main.py#L57-L85);
  [trajectory.py L253–299](https://github.com/hailanyi/3D-Multi-Object-Tracker/blob/d37df41c8e512e14c198cb18a4609ed66253d7a1/tracker/trajectory.py#L253-L299)).

## G. Evaluation: what exposes one object split into several tracks

- **HOTA** ([§6](https://arxiv.org/html/2009.07736#S6)).
  - "Association recall errors occur when trackers assign two different
    prIDs to the same gtTraj", which equals "fragmentations" in Leichter and
    Krupka's taxonomy.
  - TrackEval computes, per match count, AssRe = Σ m·(m / |gt_id|) / TP,
    AssPr with |tracker_id| in the denominator, and
    AssA = m / (|gt| + |tr| − m), all at each α
    ([hota.py L103–112](https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/metrics/hota.py#L103-L112)).
  - **AssRe is the direct measure** of the failure in this note; AssPr
    measures the opposite failure, one track switching between objects.
- **IDF1.** One global ID mapping, IDF1 = IDTP / (IDTP + ½IDFP + ½IDFN)
  ([identity.py L80–134](https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/metrics/identity.py#L80-L134)).
  An object split into k tracks earns credit only for its best-mapped track.
- **CLEAR**
  ([clear.py L56–123](https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/metrics/clear.py#L56-L123)).
  - IDSW counts a matched tracker id that differs from the gt object's last
    matched id. Matching first prefers to continue the previous frame's
    pairs: score + 1000·same-id.
  - **Frag counts interruptions in coverage.** Untracked → tracked
    transitions minus one: a continuous split into two ids is 0 Frag and 1
    IDSW.
- **nuScenes.**
  - Match by 2 m BEV centre distance. Before evaluation, GT and predicted
    tracks are linearly interpolated, and GT boxes without points are
    removed ([README L221–232](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/eval/tracking/README.md#L221-L232)).
  - IDS, Frag, TID (initialisation duration) and LGD (longest gap duration)
    are defined at the confidence threshold that gives the best MOTA
    ([README L262–281](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/eval/tracking/README.md#L262-L281)).

---

## What this means for wato_world tracking

Everything below is deduction, not a claim about the sources. The ranking
weighs the measured break causes (39% shape gate, 38% duplicate, 12% distance
gate, 7% coast) against effort. Each item names the published defaults to
test end to end, unchanged.

Items 0 to 4 are implemented (item 1 with CenterPoint's vehicle heads only,
item 2 with SimpleTrack's all-class GIoU); their measurements, and what they
leave, are in [`tracking_measurements.md`](../tracking_measurements.md).

### 0. Score splits the way the references do (prerequisite)

- **What.** Add AssRe, AssPr and AssA at the 2 m match (TrackEval's formulas
  with a single α), IDF1, IDSW and Frag to
  [`eval_tracks.py`](../../src/tracking/scripts/eval_tracks.py).
- **How.**
  - IDSW and Frag must use CLEAR's matching, which prefers to continue the
    previous frame's pairs. The current `match_frame` is score-greedy and
    would mis-count IDSW.
  - Keep "tracks per object" and purity, and report the break-cause
    breakdown already measured.
- **No-GT WATO proxy.** Count lifted-instance ids whose rows land in more
  than one track. SAM2 identity serves as weak ground truth. It over-counts
  where a masklet drifts.
- **Why first.** AssRe and AssPr make the trade-off in items 1 to 3 visible:
  fewer splits against more switches.

### 1. Duplicate suppression before association (targets the 38%)

- **What.** Per frame, suppress fused rows by centre distance within a class
  group, as CenterPoint's circle NMS does.
- **Published defaults.** Squared radii car 4, truck 12, bus 10, two-wheelers
  0.85 and pedestrian 0.175 (2.0, 3.46, 3.16, 0.92 and 0.42 m). The
  task-head grouping includes [motorcycle, bicycle] and [pedestrian, cone].
- **Why distance.** Only a distance rule reaches the measured IoU-0 boxes
  about 0.9 m apart. SimpleTrack's and ImmortalTracker's IoU NMS, Poly-MOT's
  0.08 BEV IoU, DetZero's overlap ratio and Offline-Poly's IoU/gIoU adjacency
  all need overlap. Fast-Poly's scale NMS (×1.1 pedestrian, ×1.4 bicycle)
  is the published middle ground.
- **Where.** A pre-processing step in
  [`detections.frames`](../../src/tracking/src/wato_tracking/detections.py),
  as SimpleTrack and Poly-MOT pre-process inside the tracker. Removing the
  rows upstream in proposal_generation's fusion is the alternative
  ownership.
- **Inputs.** Fused rows, `cls`, `score` and `strong`; all exist.
- **Risks.**
  - CenterPoint orders by calibrated detector score; fused scores are not
    calibrated. Ordering strong rows first, then by score, is a deduction.
  - `cls` is None on lidar_mos-only rows.
  - The rider case (pedestrian + motorcycle) crosses head groups and is not
    covered.
  - A 2 m car radius can delete a real adjacent object's sliver.
  - The two pedestrians walking 0.5–1.3 m apart in
    `tracking_measurements.md` survive the 0.42 m pedestrian radius.
  - Suppressed rows must stay visible in diagnostics, so recall loss is
    measurable: compare frame recall before and after.
- **Measure.** The duplicate share of breaks, AssRe/AssPr, frame recall and
  precision, and the number of tracks on scene-0061 and scene-1100 and on the
  WATO proxy.
- **Rule.** The radii are CenterPoint code defaults tied to its detector heads
  and class grouping, not described in its paper. They are published values
  used unchanged; mapping classes without a head (e.g. None) is a choice to
  document.

### 2. Association cost that tolerates partial boxes (targets the 39%, part of the 12%)

- **What.** Give tracking an alternative cost in `associate()`, keeping Chen
  as the default for Step F. Test these in this order:
  1. **GIoU-3D > −0.5 for all classes, Hungarian** (SimpleTrack;
     ImmortalTracker on nuScenes). This is the simplest published all-class
     default.
  2. **Poly-MOT per-class GIoU**, class-gated. `1 − GIoU ≤` car 1.31,
     ped 1.69, bicycle 1.55, motorcycle 1.44, bus 1.34 (BEV), trailer 1.25,
     truck 1.21.
  3. **Fast-Poly per-class A-GIoU.** Car 1.21, ped 1.78, bicycle 1.62,
     motorcycle 1.6, bus 1.16 (BEV), trailer 1.16, truck 1.22. It ignores
     the undirected and noisy heading entirely.
  4. **MCTrack Ro_GDIoU.** Car 1.10 with (w1, w2) = (0.5, 1.5); pedestrian
     2.06 with (1, 1). Its centre-distance term keeps a gradient once the
     boxes stop overlapping.
- **Where.**
  [`association.py`](../../src/common/src/wato_common/tracking/association.py),
  with a `cost` option, plus `giou_3d` in
  [`boxes.py`](../../src/common/src/wato_common/tracking/boxes.py) using the
  existing polygon clipping and `scipy` ConvexHull. SimpleTrack's
  `giou3d` (MIT) is a reference. Selected in `tracking.yaml`.
- **Inputs.** Boxes and `cls`; all exist. The class gate must let a None
  class match any class (deduction).
- **Worked numbers on our boxes (deduction).**
  - Case 1: a car's 4.5 × 1.8 m box and a 1.0 × 0.2 m sliver inside it.
    IoU = GIoU ≈ 0.025, so it fails Chen (IoU ≥ 0.05, volume ratio ≥ 0.3)
    but passes GIoU > −0.5.
  - Case 2: two parallel 1.0 × 0.2 m slivers 0.65 m apart. GIoU ≈ −0.53, so
    it fails −0.5 and passes Poly-MOT's pedestrian −0.69 but not its car
    −0.31.
  - So GIoU rescues sliver-inside-whole breaks, not every face change.
    Re-linking in item 3 covers the rest.
- **Risks.**
  - Wider gates admit neighbour jumps; watch AssPr and switched share.
  - The instance bonus of 1.0 is on Chen's cost scale and has no published
    equivalent on a 1 − GIoU scale. Keep `instance_opens_shape_gates`, which
    has no magnitude, and re-measure the bonus or drop it.
  - Mahalanobis (Chiu, threshold 11, greedy) is not a direct candidate. Its S
    comes from detector-error statistics on nuScenes train, which do not
    exist for WATO. On our hand-set `KalmanNoise`, the extent residual of a
    sliver against a whole box would dominate the distance.
  - CenterPoint's distance gates (car 4 m, ped 1 m) are the 99.9th
    percentile of its velocity-head error per 0.5 s. We have no detector
    velocity, so they are not transferable.
- **Measure.** Shape-gate break share, AssRe/AssPr, tracks per object,
  switched share.
- **Rule.** GIoU −0.5 and the per-class thresholds are published defaults.
  The Poly-MOT, Fast-Poly and MCTrack repositories have no licence: use the
  values, write the code.

### 3. Training-free tracklet re-linking after the pass (shape and distance remainder)

- **What.** Offline-Poly STWO:
  - Predict each tracklet at most 1 s forward from its end and backward from
    its start, with our CV model.
  - Compute per-frame 1 − IoU between a tracklet's observed or predicted
    state and the other's.
  - Match one-to-one with θ_blo ≈ 0.9 and iterate.
- **Additions.** Two cues the references do not use, as gates rather than
  costs:
  - Class consistency, as in Offline-Poly's MTM and Poly-MOT.
  - Shared lifted-instance id. Half the distance-gate breaks carry an id the
    track already had, so a shared id can admit a pair the 1 s IoU test
    rejects. This mirrors the existing per-frame rule.
- **Where.** A new `relink.py` called in
  [`pipeline.track_chunk`](../../src/tracking/src/wato_tracking/pipeline.py)
  after `tracker.run`, before `stitch.chains`.
- **Inputs.** Per-observation filtered boxes exist. Velocity at each
  tracklet's first and last observation is needed for prediction:
  `TrackObservation` would need to store `kf.velocity`, an additive change
  in common.
- **Solver.** Blossom needs `networkx`, which is not in
  `docker/requirements/tracking.txt`; adding it follows the lock procedure.
  A Hungarian solve on track ends against track starts is not the paper's
  method: it allows chains in one pass.
- **Unspecified in the paper, so record as choices:**
  - How per-frame matches across a tracklet are combined.
  - That temporally overlapping pairs are "invalid" (STW's domain).
  - The iteration stop condition.
- **Cost metric on slivers.** IoU between slivers is often 0. Offline-Poly
  Table VIII treats IoU, gIoU and A-gIoU as interchangeable STWO metrics, so
  reuse whichever cost wins item 2.
- **Appearance as a supporting cue only.** The measured AUC with nearby
  same-class objects is 0.55 for the stored embedding and 0.70–0.81 for
  mask-weighted patch tokens. The correct track is the more similar one in
  only 50–67% of shape-gate breaks.
  - If used, apply it BoT-SORT-style: appearance counts only after the
    geometric gate passes, and the link cost is GIAOTracker's minimum over
    banks.
  - It needs per-frame mask-weighted embeddings recomputed in perception_2d.
    The stored per-masklet bbox-crop mean is near chance on the cases that
    matter.
- **Measure.** AssRe/AssPr, tracks per object, IDS-like count, WATO key-split
  proxy and median track length.
- **Rule.**
  - The 1 s horizon is published.
  - θ_blo "around 0.9" is the reported optimum of a per-dataset linear
    search, not a fixed default. Use 0.9 as published; tuning it on two mini
  scenes would be the forbidden kind of tuning.
  - There is **no published appearance threshold for DINOv2 vehicle crops**.
    BoT-SORT's 0.25 is specific to its model. Calibrating θ_emb from
    self-labelled pairs is not a published procedure: same instance across
    cameras as positives, distinct masklets in one camera frame as
    negatives. It is a deduction that needs sign-off under the
    no-ad-hoc-thresholds rule.

### 4. Never terminate inside a chunk (targets the 7%; WATO 12% expired-then-reborn)

- **What.** ImmortalTracker's rule: predictions continue until the chunk
  ends, and predicted frames are not output (already true).
- **Where.** `max_coast_s` in
  [`tracking.yaml`](../../src/tracking/config/tracking.yaml) and
  [`config.py`](../../src/tracking/src/wato_tracking/config.py), for example
  an explicit "never". Step F keeps n_old = 5.
- **Risks.**
  - A parked car's jitter velocity is carried forward by CV. For example,
    0.5 m/s drifts 10 m in 20 s, after which the track can grab a
    neighbour's box. Watch AssPr.
  - The number of live tracks grows, and `associate()` computes `iou_3d` in
    Python for every pair. Add a centre-distance pre-check before IoU; the
    result is identical and it is only an optimisation.
  - ImmortalTracker was verified only on sequences of about 20 s or less;
    our chunks are about 30 s.
- **Rule.** No number to choose.

### 5. Better state after a gap (rest of the 12% distance-gate breaks)

- **What.**
  - OC-SORT's ORU: on re-activation, replay KF updates along the linear
    virtual trajectory.
  - BiTrack's velocity re-initialisation after the first match.
  - Both are opt-in in
    [`kalman.py`](../../src/common/src/wato_common/tracking/kalman.py) and
    `MultiObjectTracker`.
- **Risk.** A one-sided box's centre jumps as its visible face changes
  (`label_refinement` README). Both methods turn that jump into velocity.
- **Measure.** Distance-gate break share.
- **Rule.** No thresholds: OCM's Δt = 3 and λ = 0.2 only apply if OCM is
  added too.

### 6. Bidirectional pass with BiTrack fusion (later)

- **What.** Run `tracker.run` on time-reversed frames, which needs negated
  timestamps because `predict` clamps dt ≥ 0. Then cluster forward and
  backward tracks by shared `proposal_id` and fuse fragments by BiTrack's
  priority h.
- **Why later.** The published gain is small (KITTI HOTA +0.04). Items 2 and
  3 address the measured causes more directly.

### Not recommended now

- CTRA and bicycle motion models: they need a directed heading coupled to
  velocity; ours is modulo π, and MCTrack uses CV on nuScenes anyway.
- Learned linkers: SUSHI, MPN, Batch3DMOT, OTOP, AFLink (2D pixels, MOT17,
  GPL-3.0) and CAMO-MOT. There are no WATO labels.
- MASA embeddings: heavy mmdet/GPU stack, and its weight licence is
  unchecked.
- CTRL-style extension: it adds unmatched boxes without fixing identity.
- "Weak never starts a track": ruled out by measurement (45/76 objects never
  strong).
- SimpleTrack's "second-stage matches do not update the KF": our weak boxes
  are most of the data. It is a published difference worth a later
  ablation, not a fragmentation fix.

### Rule check

| Item | Published values used | Gap needing a decision |
|---|---|---|
| 1 Circle NMS | CenterPoint radii (code) | ordering on uncalibrated scores; class groups for None and cross-class |
| 2 GIoU, A-GIoU, Ro_GDIoU | SimpleTrack −0.5; Poly-MOT, Fast-Poly and MCTrack per-class values | new magnitude for the instance bonus |
| 3 STWO | 1 s horizon, IoU cost, θ_blo ≈ 0.9 | per-frame aggregation, invalid pairs, stop rule; appearance threshold unpublished |
| 4 Immortal | never terminate | none |
| 5 ORU / velocity re-initialisation | method only | none |

## Not verified or open

- **Offline-Poly's final hyperparameters** (θ_age, θ_score, θ_blo, θ_multi,
  K, M) are not printed; only sensitivity ranges are. The code is
  unreleased, so STWO's per-frame aggregation, invalid-state definition and
  stop rule could not be checked.
- **Paper and code disagree in several places:**
  - DetZero: two-stage thresholds swapped; no forward/reverse WBF in the
    tracking code.
  - CTRL: extension length 20/100 frames in the paper, 10 frames backward in
    code.
  - BoT-SORT: θ_emb applied after halving in code.
- **DINO-MOT's method and numbers**: only the bibliographic record was
  verified.
- **CasTrack's paper** (IEEE T-ITS) was not read; only its code and README.
- **MASA weight licence** on Hugging Face was not checked.
- **Out of scope.** perception_2d cites SAM2MOT's untracked-area threshold
  as 0.5. That matches arXiv v1–v3 (§4.2); v4 and v5 state r = 0.7
  ([v5 hyperparameters](https://arxiv.org/html/2504.04519)).
