# Offline tracking, track filtering and track-level refinement for 3D auto-labeling

Scope: how offboard pipelines track per-frame proposals, kill false tracks, split static from moving objects, refine boxes per track (training-free where possible), stitch long or chunked sequences, and estimate label quality without ground truth. Sources were read in full text (arXiv PDFs) unless marked "search summary only". MS3D thresholds come from the repo's config files and code at commit `ffed761` (Aug 2024). Years are first-arXiv or venue years. Date of notes: 2026-11-02.

Short names used below: MS3D = darrenjkt/MS3D repo; TBT = Tracking-by-Tracking (Offline-Poly); KBF = Kernel-density Box Fusion; s2e = start-to-end (begin-to-end) distance of a track's box centers; WOD = Waymo Open Dataset.

---

## 1. 3D Auto Labeling (Qi et al., CVPR 2021): motion-state classification, static and dynamic refinement, and the gain from each

### Takeaway
3DAL classifies each whole track as static or dynamic with a linear classifier on two heuristic track features (box-center variance and begin-to-end distance, in world coordinates, for tracks with at least 7 measurements) and gets >99% accuracy for vehicles. All pedestrians are treated as dynamic. Static objects get one box from the points merged over all frames, in the frame of the highest-score detector box. Dynamic objects get per-frame boxes from a 5-frame point window plus a 101-box trajectory window. Using the full track is the main source of the gain (static Acc@0.8 43.56 → 56.92 over the multi-frame detector). The refinement networks themselves are supervised.

### Cited Findings
- **Pipeline.** Multi-frame MVF++ detector (5 frames, test-time augmentation over 10 rotations fused with Weighted Box Fusion), then a tracker (a variant of AB3DMOT: detector boxes for association, a Kalman filter for state), then object-track data extraction (all boxes and points in world coordinates), then the track-level motion-state classifier, then the static or dynamic auto-labeling model. — [3DAL, arXiv 2103.05073](https://arxiv.org/abs/2103.05073)
- **Tracker details.** Detections with score < 0.1 are dropped before tracking. Association uses BEV boxes and the Hungarian algorithm with an IoU threshold of 0.1, in world coordinates. Heading is handled specially: if the detection's heading differs from the track's by an obtuse angle, π is added, and angles are averaged in cyclic space. — [3DAL App. D](https://arxiv.org/abs/2103.05073)
- **Motion-state classifier (App. E).** Two features: the variance of the detection box centers, and the begin-to-end distance (first box center to last), both in world coordinates. A linear classifier is fit on them, and it only considers tracks with ≥ 7 valid measurements. Short tracks get neither classification nor auto-labeling; their boxes are "merged directly to the final auto labels". An object is static only if it is stationary for the whole sequence. Ground-truth motion states are defined with thresholds of 1.0 m begin-to-end distance and 1 m/s max speed, to tolerate small pose drift. For vehicles the classifier exceeds 99% accuracy; the remaining errors are "short tracks with noisy detection boxes, or … heavily occluded or far away" objects. All pedestrian tracks are treated as dynamic because "even the static ones tend to move their arms and heads". — [3DAL §4.4, App. E](https://arxiv.org/abs/2103.05073)
- **Static refinement.** The merged world-frame points of all frames are transformed into the coordinate frame of one "initial box" (+X = heading, origin = center). A PointNet segments the foreground, then a box-regression PointNet is applied twice (iterative, shared weights), plus test-time augmentation. One box is produced per static object. Inputs are XYZ only; intensity was not used, and PointNet++ gave no improvement. — [3DAL §4.4, App. F.1](https://arxiv.org/abs/2103.05073)
- **Choice of initial box (Table 14, static Acc@0.7/0.8).** With a multi-frame MVF++ initialization: random box 79.66/52.46, average box 81.22/53.96, highest-score box 82.28/56.92. That is a 4.46-point Acc@0.8 swing, and the paper calls the highest-score choice "similar to choosing the best viewpoint". The raw detector boxes score 73.96/43.56 (multi-frame) and 67.17/36.61 (single-frame). — [3DAL App. I](https://arxiv.org/abs/2103.05073)
- **Static ablation (Table 6, Acc@0.7/0.8).** No transform 78.82/50.90; with box-frame transform 81.35/54.76; plus segmentation 81.37/55.67; plus iterative regression 82.02/56.77; plus test-time augmentation 82.28/56.92. — [3DAL Table 6](https://arxiv.org/abs/2103.05073)
- **Dynamic refinement.** At center frame T, the points of frames T±2 (5 frames, 1,024 points each, a time channel of ±0.1·r) are merged, unaligned, into the frame of the detector box at T. A segmentation network and a point encoder produce one embedding. The box sequence T±50 (101 boxes, each a 7-DoF + time "point") goes through a PointNet to give a trajectory embedding. The two embeddings are concatenated and regressed to the box at T. — [3DAL §4.4, App. F.2](https://arxiv.org/abs/2103.05073)
- **Dynamic design ablation (Table 7, Acc@0.7/0.8).** Align & refine (registering context-frame points to the center frame by aligning detector-box centers and headings) 83.33/60.69; unaligned points only 83.79/61.95; box sequence only 83.13/58.96; joint 85.67/65.77. On why alignment is worse: "alignment can be a harder problem than box estimation especially for occluded or faraway objects with fewer points". — [3DAL Table 7, §4.4](https://arxiv.org/abs/2103.05073)
- **Temporal context (Table 8, vehicles, static / dynamic Acc@0.7/0.8).** Single-frame MVF++ 67.17/36.61 and 80.07/57.71. Multi-frame MVF++ 73.96/43.56 and 82.21/59.52. 3DAL with [−0,+0] 78.13/50.30 and 80.65/57.97; [−2,+2] 79.60/52.52 and 84.34/63.60; [−5,+5] 80.48/55.02 and 85.10/64.51; all frames 82.28/56.92 and 85.67/65.77. Dynamic vehicles score higher "because they are closer to the sensor than static ones". — [3DAL Table 8](https://arxiv.org/abs/2103.05073)
- **End-to-end gains and the human study.** Vehicle 3D AP@0.8 improves by 14.52 points over single-frame MVF++ and 8.39 over multi-frame MVF++. Pedestrian AP@0.6 improves by 7.67 and 3.13 points. On 5 re-labeled WOD sequences, human vs 3DAL is 86.45 vs 85.37 3D AP@0.7 and 60.49 vs 56.93 AP@0.8. The gap is attributed mostly to inaccurate heights; BEV AP@0.8 is 86.27 for humans vs 87.55 for 3DAL. Human APs were computed using "number of points in boxes as human label scores". — [3DAL Tables 1–2](https://arxiv.org/abs/2103.05073)
- **Auto labels as training data.** A student trained on 10% human labels plus 90% 3DAL labels comes within 1 AP point of 100% human labels; only auto labels with score > 0.1 were kept. In cross-domain use, 3DAL labels improve the student by about 5 AP points, against about 1 point for self-labeling. — [3DAL §5.3](https://arxiv.org/abs/2103.05073)

### Inferences
- The motion-state rule is cheap and training-free in spirit: two features plus a threshold. MS3D (below) ships hand-set thresholds for exactly these features. What is learned in 3DAL is the box regression, not the split.
- "Highest-score box as the reference frame" is the training-free lever that carries over directly. For WATO, the analogue is the frame with the most points, the highest lift_confidence or n_sources, or the nearest range. This matters most for far parked cars whose boxes cover one side.
- The Table 7 finding (aligning by noisy detector boxes hurts) warns against naive box-based registration of dynamic-object points. ICP or flow-based registration is a different method and was not tested there.

### Gaps
- 3DAL does not report how much of the end-to-end AP gain comes from the static branch vs the dynamic branch separately, beyond the per-branch accuracy tables.
- The learned linear classifier's weights and thresholds are not published; only the ground-truth definition (1.0 m, 1 m/s) is.

---

## 2. MS3D++ (Tsai et al., arXiv 2308.05988, 2023) and the MS3D code: static/dynamic rules, static box fusion, retroactive labeling, pedestrians from dynamic tracks, and config thresholds

### Takeaway
MS3D++ is the most directly reusable training-free track-refinement recipe. It runs a low-threshold Kalman/SimpleTrack-style tracker twice for vehicles: once permissive (GIoU) for all vehicles, and once strict (BEV IoU 0.7) aimed at static vehicles. It labels a track static iff s2e < 1 m and mean xy-variance < 0.1 m². It keeps a track only if it has enough confident detections ("retroactive labeling"). It replaces static boxes with a KDE-fused box over a rolling 10–16-frame window and propagates them up to 40–100 frames beyond the track's ends with decaying score. For pedestrians it uses only dynamic tracks (s2e > 2 m) in early rounds, to reject poles. Refinement raises recall at 30–80 m by 6–10 points at a cost of 1–5 points of precision.

### Cited Findings
- **Tracking.** A Kalman-filter tracker (SimpleTrack) runs at low score thresholds on the KBF-fused boxes. There are separate tracks for pedestrians (T_p) and two vehicle track sets: T_v,all ("tracks both dynamic and static vehicles") and T_v,static ("high BEV IoU threshold for track association"). "A high IoU for T_v,static is critical in minimising false positive tracks of parked vehicles that are classified as 'dynamic' due to inaccurate box localization." Interpolation and extrapolation are used "only for confident dynamic tracks by setting a higher score threshold for T_v,all than T_v,static", because an extrapolated dynamic track "can easily deviate significantly" at corners, under occlusion or at far range. — [MS3D++ §4.3.3](https://arxiv.org/abs/2308.05988)
- **Tracker configs (Waymo round 1; nuScenes round 1 in brackets).** Positive score thresholds POS_TH veh/ped/cyc = 0.7/0.6/0.5 in both datasets. VEH_ALL: score 0.6 [0.5], max age 2, 2 hits to birth, GIoU association threshold 1.0 [1.3]. VEH_STATIC: score 0.5 [0.3], max age 4 [3], 2 hits to birth, `iou_2d` association threshold 0.7. PEDESTRIAN: score 0.2, max age 2, 2 hits to birth, GIoU 0.8 [1.5]. — [MS3D Waymo ps_config.yaml](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tools/cfgs/target_waymo/label_generation/round1/cfgs/ps_config.yaml); [MS3D nuScenes ps_config.yaml](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tools/cfgs/target_nuscenes/label_generation/round1/cfgs/ps_config.yaml)
- **Static/dynamic rule (code).** `get_motion_state(box_seq, s2e_th=1, var_th=0.1)` uses only the xy centroids: static iff `||c_first − c_last|| < 1 m` and `mean(var(x), var(y)) < 0.1`. For vehicles it is computed only on boxes with score above the positive threshold. Pedestrians use `s2e_th=2` (variance threshold still 0.1). — [MS3D tracker_utils.py L77–90](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/tracker_utils.py#L77-L90); [ms3d_utils.py refine_ped_labels](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/ms3d_utils.py#L163)
- **Motion-state reconciliation (code).** `motion_state_refinement` compares the static tracks (from the 16-frame-accumulated detector) against the boxes of the all-vehicle tracks plus the full "motion trail" of every dynamic all-vehicle track. Pairs with IoU > 0.1 are matched, and if either is dynamic both become dynamic. Disjoint tracks are then merged (`merge_disjointed_tracks`). — [MS3D ms3d_utils.py L262+](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/ms3d_utils.py#L262)
- **Retroactive object labeling.** Tracks are not used as labels "until we confidently recognize the object class, through a number of high confidence detections (i.e., > s_pos)". Every track must contain at least N_pos confident detections. Once it qualifies, the class applies to all its past and future observations. — [MS3D++ §4.4.1](https://arxiv.org/abs/2308.05988)
- **Track-filtering thresholds (Waymo r1; nuScenes r1 in brackets).** MIN_NUM_STATIC_VEH_TRACKS 10 [7] and MIN_NUM_PED_TRACKS 10 [7] (minimum track length). Minimum detections above POS_TH per track: veh_all 4 [3], veh_static 2 [3], ped 2 [1]. USE_STATIC_PED_TRACKS false in both. — [MS3D Waymo ps_config.yaml](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tools/cfgs/target_waymo/label_generation/round1/cfgs/ps_config.yaml); [MS3D nuScenes ps_config.yaml](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tools/cfgs/target_nuscenes/label_generation/round1/cfgs/ps_config.yaml)
- **Static vehicle refinement.** KBF fuses the boxes of H historical observations of a static track into that frame's box. KBF fuses centroid, dimensions, heading and score separately by kernel density estimation with per-box weights. Code (`get_track_rolling_kde_interpolation`): for each frame, take the previous `window` frames' boxes (mirrored into the future when there is not enough history) and fuse them weighted by score; tracks shorter than the window get a single KBF box for all frames; when the fused score exceeds the static-track threshold it is raised to at least MIN_STATIC_SCORE; missing frames are interpolated. Window 16 (Waymo) [10, nuScenes]; MIN_STATIC_SCORE 0.75 [0.8]. The paper's rationale: "For T_v,static, we simply require a few strong detections of the object to be able to use it for refining all tracked observations." — [MS3D++ §4.2, §4.4.2](https://arxiv.org/abs/2308.05988); [MS3D ms3d_utils.py L371](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/ms3d_utils.py#L371)
- **Static box propagation (code).** `propagate_static_boxes` applies only to static tracks with ≥ MIN_STATIC_TRACKS boxes above POS_TH: 24 (Waymo) [10]. It copies the first or last fused box up to N_EXTRA_FRAMES = 100 [40] frames before or after the track. The score is multiplied by DEGRADE_FACTOR 0.98 per frame and clipped below at MIN_SCORE_CLIP 0.5 [0.3]. — [MS3D ms3d_utils.py L426](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/ms3d_utils.py#L426); [Waymo config](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tools/cfgs/target_waymo/label_generation/round1/cfgs/ps_config.yaml)
- **Pedestrians vs poles.** "Pre-trained detectors commonly misclassify pole-like objects as 'Pedestrians'", with scores in a range (≈0.3–0.6) similar to real pedestrians. "While pole-like objects and pedestrians may appear similar when static, the distinction becomes clearer when the object starts moving", so only dynamic pedestrian tracks are used as labels. BEV symmetry makes pedestrian IoU tolerant of heading errors. Static pedestrians are added only in later self-training rounds, once scores are calibrated. In code, the boxes of surviving pedestrian tracks have their scores raised to ped_pos_th and static ones are deleted. — [MS3D++ §4.4.3](https://arxiv.org/abs/2308.05988); [ms3d_utils.py refine_ped_labels](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/ms3d_utils.py#L163)
- **Final label set.** Per frame, the union of the KBF boxes, static vehicle boxes, refined vehicle tracks and dynamic pedestrian tracks goes through NMS. Labels with < 1 point in the single-frame cloud are removed, and only labels with score ≥ s_pos are kept. — [MS3D++ §4.5](https://arxiv.org/abs/2308.05988)
- **Effect of temporal refinement (Table 14, Waymo, precision/recall at 0–30 / 30–50 / 50–80 m).**
  - Vehicles, ensemble: 0.950/0.899, 0.904/0.704, 0.866/0.418. Refined: 0.944/0.912, 0.893/0.763, 0.851/0.514.
  - Pedestrians, ensemble: 0.872/0.646, 0.853/0.518, 0.849/0.351. Refined: 0.846/0.689, 0.818/0.574, 0.802/0.431.
  - So the recall gain is +1.3 / +5.9 / +9.6 points for vehicles and +4.3 / +5.6 / +8.0 for pedestrians, for precision losses of 0.6–4.7 points. — [MS3D++ Table 14](https://arxiv.org/abs/2308.05988)
- **Calibration caveat and runtime.** "High confidence scores may not correlate with an accurate box localization or object class". Runtime: on 20k Waymo frames the tracker took up to 11 of about 20 hours; the label steps after detection need no GPU. — [MS3D++ §4.4, §5.3.5](https://arxiv.org/abs/2308.05988)

### Inferences
- The two-tracker trick fits WATO's far-parked-car problem. A strict, high-IoU static tracker keeps a parked car whose one-sided box jitters from being split into short "moving" pieces, and the reconciliation step stops a static track from swallowing a passing car's trail.
- The thresholds (s2e 1 m, variance 0.1 m², ≥ 7–10 frames at 10 Hz) assume about 1–2 s of track at 10 Hz. At WATO's ~20 Hz canonical LiDAR, a frame-count minimum should be restated in seconds.
- WATO has no detector score. The retroactive "N_pos confident detections" test would have to use the existing evidence instead, for example frames with n_sources ≥ 2, lift_confidence above a threshold, or motion_score above a threshold.

### Gaps
- MS3D++ gives no per-component ablation isolating static KBF, propagation or retroactive filtering; only the ensemble-vs-refined breakdown (Table 14) and the per-round tables.
- No sensitivity study for s2e_th / var_th was found.

---

## 3. DetZero, CTRL, Offline-Poly, LabelFormer, UniLiPs, Fusion4DAL and other 2024–2026 offboard trackers/refiners: what they do and which parts work without training

### Takeaway
The offline trackers have converged on a few training-free ideas. They keep tracks alive through misses ("immortal"), associate in two stages (high then low score), run forward and backward passes and fuse them, re-identify fragments by motion extrapolation, and smooth trajectories with an explicit motion model (Offline-Poly sliding-window LM, LISO jerk minimization, UniLiPs splines). The big refinement gains (DetZero, CTRL, LabelFormer, 3DAL) come from supervised track-level networks. Of these, Offline-Poly is fully learning-free. UniLiPs's spline step is training-free, and dropping it costs 7.1 mAP (31.0 → 23.9).

### Cited Findings
- **DetZero (ICCV 2023) — tracking.** Two-stage association: tracks first match a high-score group (score > 0.1 and > 3 points for vehicles, > 1 for pedestrians and cyclists) at BEV IoU (0.3, 0.15, 0.15) for veh/ped/cyc. Leftover boxes form a low-score group, matched at (0.2, 0.1, 0.1). Birth rate 1, death rate infinite ("immortal until the sequence terminates"). Matched boxes replace the track state rather than updating a Kalman filter. Tracks overlapping above (0.5, 0.4, 0.4) are merged, keeping the earlier ID. Un-updated redundant boxes are removed. — [DetZero, arXiv 2306.06023](https://arxiv.org/abs/2306.06023)
- **DetZero — other tracking steps.** The tracker is re-run in reverse time order and the paired tracks are fused with WBF ("forward and reverse order tracking fusion"). Short tracks skip refinement and pass through. A BEV "overlap ratio" filter (intersection over the *other* box's area; 0.3 vehicle, 0.2 ped/cyc) removes boxes wrapped inside other boxes, such as a pedestrian inside a vehicle, which IoU-based NMS misses. — [DetZero §3.2, App. C](https://arxiv.org/abs/2306.06023)
- **DetZero — refinement (all learned).** It rejects the static/dynamic split ("results in the propagation of misclassification"). Instead it decomposes refinement into a Geometry Refining Model (points of all frames merged in the local box frame; t sampled views predict sizes, which are averaged into one size), a Position Refining Model and a Confidence Refining Model. The confidence model has a classification branch and an IoU-regression branch, and the final score is their geometric mean. — [DetZero §3.3](https://arxiv.org/abs/2306.06023)
- **DetZero — ablations.**
  - At IoU 0.7, the geometry model adds +1.92/+1.93 (vehicle L1/L2) and +1.64/+1.7 (pedestrian); the position model adds +2.91/+3.31 and +1.25/+1.44. At IoU 0.8 the full refiner improves vehicle APH by 26.49%/29.08% relative.
  - Its offline tracker raises Recall@track (≥ 80% of a ground-truth track's boxes matched by one predicted track) by +16.32 (vehicle) and +18.56 (pedestrian) points over AB3DMOT-style tracking, with MOTA only slightly higher.
  - On a 3DAL-style baseline (Table 6), swapping in the attribute refiner gives +1.83/+1.81 vehicle APH, while swapping in the offline tracker gives +1.18/+0.73. — [DetZero Tables 5–7](https://arxiv.org/abs/2306.06023)
- **CTRL, "Once Detected, Never Lost" (ICCV 2023) — bidirectional tracking.** It uses the Immortal Tracker with default settings for forward filling and extension, then backtraces to each track's start and extends it into the past with a backward motion model. The added boxes are "not accurate, leading to many false positives", which a learned track-centric module re-scores and refines. That module is a sparse UNet over full-length tracks (Multiple-In-Multiple-Out). Empty predictions are removed afterwards, to match the human annotation convention. — [CTRL, arXiv 2304.12315](https://arxiv.org/abs/2304.12315)
- **CTRL — bidirectional extension (Table 4, WOD L2).** Totally missed vehicle ground truths fall from 21.4k (2.05%) with the offline FSD detector to 5.1k (0.48%) after bidirectional extension and refinement. High-confidence FPs stay at 8.1k, and high-precision TPs (BEV IoU ≥ 0.9) rise from 46.4% to 56.6%. Vehicle AP/APH is 74.9/74.4 for the detector, then 80.7/80.1 after refinement, 81.4/80.7 with forward extension and 81.7/81.0 with bidirectional extension. — [CTRL Table 4](https://arxiv.org/abs/2304.12315)
- **CTRL — track-centric ablations (Table 5, L2 APH).**
  - Capping input track length at [0,50) / [0,100) / [0,200) frames gives vehicle 77.1 / 78.6 / 80.1 and pedestrian 77.6 / 78.0 / 78.3.
  - Separate static/dynamic models vs one mixed model: vehicle 79.8 vs 80.1, pedestrian 78.3 vs 78.3, so separating does not help.
  - The objects still missed are those with very few points ("50% totally missed GTs contain less than 5 points and 90% … less than 24") and short-lived tracks (mostly < 2.5 s). — [CTRL Table 5, §5](https://arxiv.org/abs/2304.12315)
- **Offline-Poly (arXiv 2602.13772, Feb 2026) — learning-free "Tracking-by-Tracking".** Input is one or more off-the-shelf tracking results. Steps:
  1. Tracklet-level pre-filter: drop tracklets whose age *and* mean confidence are both below thresholds.
  2. STWO: re-identify temporally disjoint fragments using forward-backward motion-model extrapolation, a cost matrix, a global assignment and merging.
  3. STW: separate entangled overlapping tracklets.
  4. MTM: match tracklets across multiple trackers (3D IoU works best) and fuse them heuristically.
  5. Global refinement: one size per rigid object, the softmax-score-weighted mean of the Top-K most confident observations, followed by corner-centric center alignment at the L-shape inflection point.
  6. Local refinement: sliding-window Levenberg–Marquardt fit of center, velocity and heading to a motion model.
  "Our method is learning-free and implemented in Python using NumPy." — [Offline-Poly, arXiv 2602.13772](https://arxiv.org/abs/2602.13772)
- **Offline-Poly — results.**
  - nuScenes val, CenterPoint + Fast-Poly: +1.1 / +1.2 / +2.5 AMOTA for forward / backward / bidirectional input (76.2 AMOTA bidirectional; 81.4 with more trackers fused). 77.6 AMOTA on nuScenes test; 83.00 HOTA on KITTI.
  - The dual age+score filter gives +0.3 AMOTA, while either criterion alone is "overly restrictive".
  - Refinement adds +1.4 AMOTA with multi-tracker input and +0.6 with a single tracker. Corner alignment gives +3.4% MOTA on rigid objects.
  - Local refiners compared (Table VI, AMOTA/MOTA): none 75.3/64.0; Kalman filter 75.6/64.4; RTS smoother 76.0/64.5; Gaussian process regression 75.9/64.6; sliding-window optimization 76.2/64.8.
  - The fragment re-ID threshold is best near 0.9; at 1.0 it makes false matches. — [Offline-Poly §V, Tables I, III, VI, VII](https://arxiv.org/abs/2602.13772)
- **LabelFormer (CoRL 2023).** One transformer refines a whole trajectory in a single pass, with no motion-state classifier: per-frame box and point encoders, self-attention across all frames, then a size residual plus per-frame pose residuals. "On average 92% more mean IoU gains" than prior refiners (Auto4D, 3DAL). It is 2.7× faster than windowed dynamic 3DAL on Highway. Accuracy "steadily increases with more temporal context". It is supervised: trained on 150 human-labelled Highway snippets, then used to label 500 more, which gives the downstream detector +3% AP at IoU 0.8. — [LabelFormer, arXiv 2311.01444](https://arxiv.org/abs/2311.01444)
- **UniLiPs (arXiv 2601.05105, Jan 2026) — training-free dynamic-object boxes.**
  - Pipeline: points labeled moving in three consecutive scans are aligned and clustered with HDBSCAN. Each cluster gets a minimum enclosing cuboid with a minimum size, PCA for initial yaw, and a constant-velocity Kalman tracker with yaw dynamics.
  - Each trajectory is then refined by spline optimization: yaw is fit as sine and cosine basis functions, `e_yaw = ½Σ(f_c(t_j) − cos ψ_j)² + (f_s(t_j) − sin ψ_j)²`, and x(t), y(t) are fit by least squares.
  - Table 8a: full method 31.0 mAP / 45.2 ND-score; without spline 23.9 / 33.2; without f_iwu 11.7 / 30.8.
  - Metric: mAP with a 6 m center-distance threshold out to 250 m, on their highway dataset. Baselines: ICP-Flow-derived boxes 7.2 / 11.4, LISO 21.1 / 40.9. — [UniLiPs §3, Tables 5 and 8](https://arxiv.org/abs/2601.05105)
- **LISO (ECCV 2024) — training-free tracking and track optimization before self-training.**
  - Candidates: points with residual scene flow > 1 m/s, clustered with DBSCAN in 6D (ε 1.0, minPts 5). Boxes are discarded if l/w > 4, lw < 0.35 m² or lwh < 0.5 m³. Heading is set to the flow direction.
  - Tracker: boxes are propagated by flow and matched greedily on center distance (new track if > 1.5 m from any prediction), coasting at most one step. It runs forward and reverse and joins the two.
  - Filtering: tracks shorter than 4 steps or with median confidence < 0.3 are dropped.
  - Smoothing (tracks > 3 m long): minimize the 4th derivative of position plus β = 3 times the squared distance to the observations. This beat bicycle-model fits, which "overshoot" corners.
  - Heading is then aligned to the smoothed path, and box dimensions are set to the 90th percentile per track.
  - Ablation (WOD, my parse of the flattened Table 4, BEV/3D AP@0.4): the flow tracker is the largest factor (an Oyster-style tracker matches at 5 m, LISO at 1.5 m). Track optimization takes BEV AP 0.279 → 0.292 without self-training and 0.360 → 0.380 with it. — [LISO, arXiv 2403.07071](https://arxiv.org/abs/2403.07071)
- **Immortal Tracker (arXiv 2111.13672, 2021) — training-free.** Tracklets never die. Unmatched tracklets keep predicting with a 3D Kalman filter, but only alive tracklets matched in the current frame are output, "therefore the predictions of unmatched tracklets will not cause false positives". Association is Hungarian on 3D IoU (> 0.1) or GIoU (> −0.5), with M_hits = 1. It avoids 96% of the vehicle ID switches caused by premature termination, and its WOD mismatch ratio is at the 0.0001 level. — [Immortal Tracker](https://arxiv.org/abs/2111.13672)
- **SimpleTrack (arXiv 2111.09621, 2021) — training-free.** NMS beats score filtering as pre-processing. 3D GIoU association with threshold > −0.5 for all classes. "Two-stage association": a low and a high score threshold (e.g. 0.1 and 0.5 for WOD pedestrians), so that low-score boxes extend existing tracklets but do not start new ones. — [SimpleTrack](https://arxiv.org/abs/2111.09621)
- **BiTrack (arXiv 2406.18414, v2 Mar 2025).** Camera-LiDAR offline MOT. 2D-3D fusion counts projected LiDAR points inside 2D segmentation masks (a density-based similarity). It adds vertex-based association, false-alarm rejection and track recovery, then a bidirectional greedy re-organization of track fragments with completion and smoothing. SOTA on KITTI offline MOT per the authors. Offline-Poly reports being +0.3 HOTA above it and notes that BiTrack "depends on upstream tracking information and 2D segmentation". — [BiTrack](https://arxiv.org/abs/2406.18414); [Offline-Poly §V-C](https://arxiv.org/abs/2602.13772)
- **OTOP and RethinkMOT (as summarized by Offline-Poly).** OTOP "leverages static map priors" to complete tracklets with a learned affinity; RethinkMOT "introduces an outlier detector" and "requires additional training"; PC3T "employs score filtering and empty-frame interpolation". Offline-Poly's motion-based re-ID "surpasses OTOP in performance, requires no training". — [Offline-Poly §II, §V-D](https://arxiv.org/abs/2602.13772)
- **Fusion4DAL (IJCV 2025, vol. 133(7), pp. 3951–3969; Yang, Wang, Zhang, Tan, Lu, Wang, Ding, Zhao).** Indexed keywords: offline 3D detection for 4D auto-labeling, a multi-modal mixed-feature fusion module, global point attention, a virtual point loss. — [dblp](https://dblp.dagstuhl.de/pid/180/5531.html); [AMiner record](https://resource.aminer.org/pub/6845189f163c01c8506d6f28/fusion-dal-offline-multi-modal-d-object-detection-for-d-auto-labeling) (metadata only; full text not accessed)

### Inferences
- Training-free parts that can be lifted directly:
  - Immortal track maintenance, outputting only matched states.
  - SimpleTrack/DetZero two-stage association.
  - Forward plus reverse passes with fusion (DetZero, LISO, OYSTER, CTRL).
  - Offline-Poly's dual-criterion tracklet filter, motion-extrapolated fragment re-ID, Top-K size and corner alignment.
  - A smoother: sliding-window LM, LISO's jerk objective, or UniLiPs splines.
  - DetZero's overlap-ratio filter.
- Parts that need labels: DetZero's three refiners, CTRL's track-centric learner, LabelFormer, 3DAL's PointNets. Their reported gains (+2 to +6 AP at strict IoU) are the ceiling a later learned stage could add.
- CTRL's bidirectional extension only paid off because a learned re-scorer suppressed the extrapolated false positives. Without one, extrapolated boxes should stay unlabelled or low-confidence, as MS3D does with its decaying score.
- Two results argue against a hard static/dynamic split inside a learned refiner: CTRL Table 5c and DetZero's rationale. For a training-free pipeline the split is still useful, because the static and dynamic refinement procedures differ (one box vs smoothed per-frame boxes).

### Gaps
- Fusion4DAL's method details, ablations and any training-free components could not be verified (paywalled; Springer link 404).
- BiTrack's thresholds and ablations were not extracted.
- No 2025–2026 offboard tracker evaluated on 3-LiDAR + 8-camera rigs or on non-synchronized cameras was found.

---

## 4. Track filtering: length, score, motion- and size-consistency rules; pedestrians vs poles; occlusion-aware continuation

### Takeaway
Every pipeline filters at the track level rather than the frame level. The common recipe is a minimum length (LISO 4 steps, OYSTER 6, 3DAL 7, MS3D 7–10 frames), a minimum count of confident frames (MS3D: 1–4 above the positive threshold), and a mean or median confidence threshold (LISO median 0.3). Offline-Poly finds that requiring *both* short and low-confidence to drop a track beats either rule alone. Motion-based class rules (MS3D: pedestrians only if s2e > 2 m) handle poles. Occlusion is handled by immortal or coasting tracks whose predictions are not output, plus offline fragment re-ID.

### Cited Findings
- **Length and score rules.**
  - 3DAL: detections with score < 0.1 dropped; tracks with < 7 measurements not refined. — [3DAL](https://arxiv.org/abs/2103.05073)
  - LISO: drop tracks shorter than 4 time steps or with median confidence < 0.3. — [LISO §3](https://arxiv.org/abs/2403.07071)
  - OYSTER (CVPR 2023): each pseudo-label gets a temporal-consistency score, the length of the tracklet it belongs to (the max of the forward and backward runs). Labels with score < q = 6 are ignored in the next self-training round. — [OYSTER, arXiv 2311.02007](https://arxiv.org/abs/2311.02007)
  - MS3D: minimum track length 10 (Waymo) / 7 (nuScenes) for static vehicles and pedestrians, and 2–4 detections above POS_TH (0.7 veh, 0.6 ped). — [MS3D config](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/tools/cfgs/target_waymo/label_generation/round1/cfgs/ps_config.yaml)
  - Offline-Poly: drop a tracklet iff its age < θ_age **and** its mean confidence < θ_score. The dual criterion gives +0.3 AMOTA; either one alone loses recall. — [Offline-Poly §IV, §V-D](https://arxiv.org/abs/2602.13772)
- **Cost of length filters.** CTRL finds that the objects it misses belong mostly to short-lived ground-truth tracks (< 2.5 s) and have very few points. A length filter therefore trades recall on exactly the sparse or far objects. — [CTRL §5](https://arxiv.org/abs/2304.12315)
- **Geometric consistency filters.**
  - DetZero's overlap-ratio filter removes boxes nested inside other boxes (a pedestrian inside a vehicle). — [DetZero App. C](https://arxiv.org/abs/2306.06023)
  - LISO discards boxes with l/w > 4, lw < 0.35 m² or lwh < 0.5 m³. — [LISO](https://arxiv.org/abs/2403.07071)
  - MS3D removes labels with < 1 point in the single-frame cloud. — [MS3D++ §4.5](https://arxiv.org/abs/2308.05988)
  - CTRL removes empty predictions after refinement. — [CTRL §4.5](https://arxiv.org/abs/2304.12315)
- **Motion-consistency rules.**
  - MS3D's static test (s2e < 1 m, xy variance < 0.1). A static vehicle track overlapping a dynamic track's motion trail is relabeled dynamic. — [MS3D code](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/tracker_utils.py#L77-L90)
  - UNION (NeurIPS 2024): a proposal is dynamic if its ICP-Flow-based velocity (2D translation + yaw) is ≥ 0.50 m/s. — [UNION, arXiv 2405.15688](https://arxiv.org/abs/2405.15688)
  - LISO uses 1 m/s residual flow for moving points. — [LISO](https://arxiv.org/abs/2403.07071)
- **Pedestrians vs poles.**
  - MS3D: use only dynamic pedestrian tracks (s2e > 2 m) as labels in early rounds; static pedestrian tracks are deleted (`USE_STATIC_PED_TRACKS: false`). — [MS3D++ §4.4.3](https://arxiv.org/abs/2308.05988); [code](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/ms3d_utils.py#L163)
  - 3DAL treats all pedestrian tracks as dynamic for refinement. — [3DAL App. E](https://arxiv.org/abs/2103.05073)
  - UNION recovers static mobile objects without motion: DINOv2 embeddings of each proposal's projected points are clustered, and appearance clusters containing ≥ 5% dynamic proposals are declared "mobile". Static members of those clusters (parked cars, standing pedestrians) become labels; background clusters (poles, buildings) do not. — [UNION §3.3–3.4](https://arxiv.org/abs/2405.15688)
- **Occlusion-aware continuation.**
  - Immortal Tracker: never terminate; output only matched alive states, so coasting predictions are not false positives. — [Immortal Tracker](https://arxiv.org/abs/2111.13672)
  - DetZero: birth 1, death ∞, then delete un-updated redundant boxes. — [DetZero App. C](https://arxiv.org/abs/2306.06023)
  - CTRL: bidirectional extension cuts totally missed vehicle ground truths 2.05% → 0.48%, relying on a learned re-scorer. — [CTRL Table 4](https://arxiv.org/abs/2304.12315)
  - Offline-Poly STWO: reconnect non-overlapping fragments by forward-backward motion extrapolation; examples include a motorcycle after long occlusion. — [Offline-Poly §IV-B](https://arxiv.org/abs/2602.13772)
  - MS3D: extrapolate only confident dynamic tracks; propagate static boxes with a 0.98-per-frame score decay. — [MS3D++ §4.3.3](https://arxiv.org/abs/2308.05988)
  - LISO: coast unmatched tracklets for only 1 step. — [LISO](https://arxiv.org/abs/2403.07071)
- **Two-stage association.** Low-score boxes (above T_l) may extend existing tracks; only high-score boxes (above T_h) may start tracks (T_l 0.1, T_h 0.5 for WOD pedestrians). — [SimpleTrack §4.4](https://arxiv.org/abs/2111.09621)

### Inferences
- WATO has no detector score. Its evidence fields can stand in for confidence in these rules: n_sources, lift_confidence, motion_score and frac_persistent per frame; track_life and point counts per track. For example, "≥ N frames with n_sources = 2 or lift_confidence ≥ τ" plays the role of MS3D's "≥ N_pos detections above s_pos". The Offline-Poly dual rule then reads "drop iff short AND low mean evidence".
- For pedestrians, which fragment and are confused with poles, a combination is plausible:
  - Two-stage association with GIoU (tolerant for small boxes).
  - MS3D's "dynamic only" rule for LiDAR-only pedestrian tracks.
  - Camera-lifted class plus DINOv2 appearance (UNION-style) to admit static pedestrians that the lifted source supports.
- For occluded cars boxed on their occluder: a nested or overlapping-box check (DetZero overlap ratio) and size consistency against the track's robust size would flag frames where the box jumps to the occluder. This is inferred; no paper reports this exact failure.

### Gaps
- No paper found that publishes explicit *size-consistency* rejection thresholds (e.g., per-frame size deviation from the track median). Offline-Poly, LISO and OYSTER replace sizes with a robust track size rather than filtering on it.
- No study found of pedestrian-fragment merging beyond generic re-ID (Offline-Poly STWO, BiTrack recovery).

---

## 5. Training-free per-track shape and size estimation: point aggregation in the box frame, registration/ICP, a shared size from the best frames, partial views

### Takeaway
Training-free systems do not regress size from aggregated points. They take a robust statistic of per-frame box sizes over the track (OYSTER: top r = 95–100% percentile of length and width; LISO: 90th percentile; Offline-Poly: softmax-score-weighted mean of the Top-K most confident frames) and re-anchor the center at the corner nearest the ego (OYSTER, Offline-Poly), which corrects one-sided boxes on partially seen cars. For static objects, MS3D fuses per-frame boxes over a window with KDE. Aggregating points across frames (3DAL, DetZero) gives the biggest gains, but in the cited works it feeds learned regressors. 3DAL found that aligning dynamic points by noisy boxes hurts.

### Cited Findings
- **OYSTER (CVPR 2023).** "Due to the tightest-box-fitting nature of the seed clustering labels, the bbox sizes … are smaller for partially observed objects, especially for those farther away". For tracklets of length ≥ q = 6, length and width are set to the top r percentile (r = 100% Pandaset, 95% AV2) of all lengths and widths over the tracklet. The center is then recomputed from "the bbox corner that is the closest to the ego-vehicle … most observed and likely to be the most reliable", keeping the original heading. — [OYSTER §3](https://arxiv.org/abs/2311.02007)
- **LISO.** Box dimensions of all boxes in a track are set to the 90th percentile of observed dimensions. Heading is aligned to the smoothed track direction. Hyper-parameters were "tuned visually on two sample snippets from nuScenes". — [LISO §3.2](https://arxiv.org/abs/2403.07071)
- **Offline-Poly.** "A fundamental property of rigid objects is the temporal invariance of physical size." One size per tracklet: the confidence-weighted (softmax) average of the Top-K most reliable observations. The center is adjusted by corner-centric alignment, since the "L-shaped inflection point provides a more reliable geometric reference than the inferred box center". Corner alignment gives +3.4% MOTA for rigid objects. — [Offline-Poly §IV-D, Table VII](https://arxiv.org/abs/2602.13772)
- **MS3D static KBF.** Score-weighted kernel-density fusion of centroid, dimensions and heading over a rolling 10–16-frame window per static track. — [MS3D code](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/ms3d_utils.py#L371)
- **3DAL static (learned regressor, training-free choice of frame).** Merge all frames' points in world coordinates and transform them into the frame of the highest-score box. Highest-score > average > random initial box: Acc@0.8 56.92 vs 53.96 vs 52.46. — [3DAL App. I](https://arxiv.org/abs/2103.05073)
- **3DAL dynamic.** Registering context points by aligning detector-box centers and headings ("align & refine", 83.33/60.69) underperformed unaligned points (83.79/61.95) and the joint model (85.67/65.77). — [3DAL Table 7](https://arxiv.org/abs/2103.05073)
- **DetZero GRM (learned).** Points of all frames are transformed into each frame's local box coordinates and merged "irrespective of their original origin". A point-to-surface encoding of the proposal is added, t views predict sizes, and the predictions are averaged. Geometry refinement alone gives +1.92 vehicle APH (L1, IoU 0.7). — [DetZero §3.3.1, Table 5](https://arxiv.org/abs/2306.06023)
- **Auto4D (IROS 2021, learned).** Factorizes a trajectory into a fixed "object size in 3D that's fixed through time for rigid objects" (an object-size branch aggregating partial observations) and a motion-path branch. Reports "a 25% reduction of human annotation efforts". — [Auto4D, arXiv 2101.06586](https://arxiv.org/abs/2101.06586)
- **UniLiPs.** Each cluster gets a minimum enclosing cuboid with an enforced minimum size and PCA yaw; trajectory smoothing (spline) follows. No aggregated shape model. — [UniLiPs §3.3](https://arxiv.org/abs/2601.05105)

### Inferences
- A training-free WATO size step could combine these:
  1. Robust size per track: Top-K frames by point count or evidence, or a high percentile of l, w, h.
  2. Corner anchoring at the ego-nearest corner (OYSTER/Offline-Poly).
  3. For static tracks: merge all frames' points in the world frame. SLAM poses are already in poses.parquet, so no registration is needed if the object truly doesn't move. Fit one box (L-shape / minimum-area rectangle) in the frame of the best-viewed observation.

  Step 3 is training-free 3DAL static labeling without the PointNet. Its accuracy is unmeasured in the literature found.
- For far parked cars with about 15 points on one side, the world-frame aggregate over a pass-by should expose more sides, since the ego's viewpoint changes. Percentile sizes alone cannot fix a car that is only ever seen from one side; a class-conditioned size prior (e.g., 3DAL's template sizes 4.8×1.8×1.5 m for cars) would be the fallback. 3DAL's templates are cited from its App. F; using them as a prior is my inference.
- ICP shape registration for dynamic objects is not supported by these sources. ICP-Flow appears only as a motion estimator (UNION) and a weak box baseline (UniLiPs: 7.2 mAP). Given 3DAL's alignment result, a learned or heavily regularized method would be needed before trusting dynamic-object point accumulation.

### Gaps
- No paper found that quantifies a purely geometric static-object box fit on aggregated multi-frame points against ground truth. All accurate aggregated-points refiners were learned.
- No measured comparison of percentile vs Top-K vs aggregated-point sizing.

---

## 6. Stitching overlapping chunks and long sequences in offline tracking

### Takeaway
3D auto-labeling papers process whole sequences (WOD ~20 s; nuScenes 20 s) and do not discuss chunking. The cleanest documented stitching recipe comes from 2D MOT (SUSHI, CVPR 2023): overlapping windows (512 frames with 256 overlap), then a bipartite match of tracks across the overlap with cost 1 − IoU_track (the fraction of shared detections), and ∞ when no detection is shared. Full-length context matters for refinement: CTRL loses 3.0 vehicle APH when tracks are cut at 50 frames, and 3DAL and LabelFormer improve monotonically with more context. So stitch before refining, not after.

### Cited Findings
- **SUSHI clip stitching (2D, CVPR 2023).** Videos are processed in overlapping sliding windows of 512 frames with 256-frame overlap: clips (1,512), (257,768), and so on. Tracks T_A and T_B from adjacent windows are compared only over the overlap. Because both are built over the same detections, IoU(T_A, T_B) = |T_A ∩ T_B| / |T_A ∪ T_B| counts shared boxes. Cost = 1 − IoU if they share at least one box, otherwise ∞, and a min-cost bipartite matching assigns IDs. Gaps are filled by linear interpolation at inference. — [SUSHI, arXiv 2212.03038, App.](https://arxiv.org/abs/2212.03038)
- **Track length capping.** Breaking WOD tracks into pieces of < 50 / < 100 / < 200 frames gives L2 APH vehicle 77.1 / 78.6 / 80.1 and pedestrian 77.6 / 78.0 / 78.3. "The unbounded input length offers CTRL a significant boost." — [CTRL Table 5a](https://arxiv.org/abs/2304.12315)
- **3DAL context.** Static Acc@0.8 is 55.02 with ±5 frames vs 56.92 with all frames; dynamic is 64.51 vs 65.77. — [3DAL Table 8](https://arxiv.org/abs/2103.05073)
- **LabelFormer context.** Accuracy "steadily increases with more temporal context" (windowed variants M6–M8 vs the full model M9). — [LabelFormer Table 3](https://arxiv.org/abs/2311.01444)
- **MS3D.** Static box propagation is clipped to the sequence bounds (`seq_len`), so there is no cross-sequence stitching. — [MS3D ms3d_utils.py L426+](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/ms3d_utils.py#L426)
- **Offline-Poly re-identification (within a sequence).** Fragments are joined by forward-backward extrapolation and assignment (threshold ≈ 0.9 best). This is the same operation needed to join tracks across a chunk seam when no detections are shared. — [Offline-Poly §IV-B, §V-E](https://arxiv.org/abs/2602.13772)

### Inferences
- WATO's neighbouring chunks are decoded independently, so overlapping proposals are not literally "the same detections". In the overlap, proposals for the same sweep can be paired by `(lidar_id, reference_timestamp_ns)` plus box IoU above a threshold, then SUSHI's shared-fraction IoU computed per track pair. Unmatched seam tracks can fall back to Offline-Poly-style motion extrapolation.
- With 6 s of overlap per side, each neighbour pair shares 18 s (3 × chunk_overlap). That is far more than SUSHI's relative overlap, so track-level matching on shared frames should be well conditioned. This is an inference; no source measured it.
- Because refinement benefits from full-length tracks (CTRL, 3DAL, LabelFormer), the order should be: stitch tracks across chunks first, refine on the stitched track, then assign each instant to one chunk at the midpoint, as `src/tracking/README.md` plans.

### Gaps
- No 3D auto-labeling paper found that addresses chunked or overlapping processing or seam artefacts; no published overlap-length guidance for LiDAR offline tracking.

---

## 7. Label quality without ground truth: calibration, agreement-based reliability, uncertainty for student weighting or review routing

### Takeaway
No 3D-box-specific, training-free label-quality estimator with validated calibration was found. In practice, offboard pipelines use:
- Track-level evidence: track length and the number of confident frames (MS3D, OYSTER, LISO).
- Ensemble agreement fused by KDE (MS3D KBF).
- Learned re-scoring (DetZero's confidence model, CTRL), which needs ground truth.

The closest non-learned signals are generative point-likelihood label uncertainty (Feng et al.) and disagreement between two estimators (UA3D). UA3D's per-coordinate loss down-weighting of uncertain pseudo boxes gave +6.9 AP_BEV on nuScenes. LMD-style post-hoc meta-classifiers find real annotation errors: 43 of the top 100 flagged nuScenes boxes were wrong.

### Cited Findings
- **Scores are miscalibrated across domains.** "High confidence scores may not correlate with an accurate box localization or object class". Accurate cross-domain boxes can score as low as 0.3. MS3D's answer is temporal evidence (≥ N_pos confident frames per track) and multi-round self-training to recalibrate. Its future work proposes "quantifying the uncertainty of pseudo-labeled frames and replacing the pseudo-label of uncertain frames with human-annotated labels". — [MS3D++ §4.3.3, §4.4, §6](https://arxiv.org/abs/2308.05988)
- **Ensemble agreement (MS3D KBF).** Boxes from multiple detectors are fused by KDE separately over centroid, size, heading and score, with per-detector-set weights. The fused score reflects agreement. — [MS3D++ §4.2](https://arxiv.org/abs/2308.05988)
- **Learned re-scoring.**
  - DetZero's confidence model: a TP/FP classification branch (negative below IoU τ_l, positive above τ_h) and an IoU-regression branch, combined as a geometric mean. — [DetZero §3.3.3](https://arxiv.org/abs/2306.06023)
  - CTRL's track-centric module assigns proper confidence to extrapolated boxes. High-confidence FPs are measured at the score threshold that gives 50% recall, to "calibrate the different confidence distributions between models". — [CTRL §5](https://arxiv.org/abs/2304.12315)
- **Points-in-box as a score surrogate.** 3DAL's human-vs-auto AP study ranks human boxes by "number of points in boxes as human label scores". — [3DAL §5.2](https://arxiv.org/abs/2103.05073)
- **Label uncertainty from a generative LiDAR model (arXiv 2012.12195, 2020).** Infers per-box label uncertainty "from LiDAR point clouds based on a generative model". It represents a probabilistic box as a spatial uncertainty distribution, proposes Jaccard IoU (JIoU) as an uncertainty-aware metric, and uses the label uncertainty in the loss of a probabilistic detector. — [Feng et al., Labels Are Not Perfect](https://arxiv.org/abs/2012.12195)
- **UA3D (arXiv 2408.00619, 2024).**
  - Setting: unsupervised 3D detection trained on clustering pseudo-boxes.
  - Uncertainty signal: the disagreement between a primary and an auxiliary detection branch, per box coordinate.
  - Use: each coordinate's loss weight is reduced when its uncertainty is high.
  - Gains: +6.9% AP_BEV and +2.5% AP3D on nuScenes; +4.1% and +2.0% on Lyft. — [UA3D](https://arxiv.org/abs/2408.00619)
- **LMD / LidarMetaDetect (arXiv 2306.07835, 2023).** A post-processing quality estimator added to any pre-trained LiDAR detector, giving "a significant increase of statistical reliability in separating true from false predictions". As an annotation-error finder: "43 out of the top 100 proposals of our method indicate, in fact, erroneous annotations" on nuScenes test. — [LMD](https://arxiv.org/abs/2306.07835)
- **Other soft-weighting work (search summaries only, not read in full).**
  - Side-aware localization uncertainty per 3D box side, with soft weights in semi-supervised 3D detection. — [arXiv 2312.10390](https://arxiv.org/html/2312.10390v1)
  - Learned pseudo-label selection plus soft supervision (2025). — [arXiv 2509.23880](https://arxiv.org/pdf/2509.23880)
  - Uncertainty-aware mean teacher for source-free 3D domain adaptation. — [arXiv 2109.14651](https://arxiv.org/pdf/2109.14651)
- **Annotator-reliability (Dawid–Skene-like) models for boxes (search summary only).** Bayesian Detector Combination models each annotator's box translation and scale errors as Gaussians and aggregates boxes and class labels with per-annotator reliability. It targets 2D crowdsourced annotation, not 3D sources. — [arXiv 2407.07958](https://arxiv.org/pdf/2407.07958)
- **Box timing errors (search summary only).** Systematic temporal misplacement in 3D box annotations (box at the reference timestamp vs per-point LiDAR capture times), when injected at the observed magnitudes, degraded Argoverse 2 metrics by more than typical gaps between SOTA detectors (WACV 2026). — [arXiv 2601.14038](https://arxiv.org/html/2601.14038v1)

### Inferences
- WATO already has two independent sources per box (`lidar_mos`, `lifted`) plus `fused`. A Dawid–Skene-style or Bayesian Detector Combination-style model is a natural training-free reliability estimator, with each source treated as an "annotator" with a latent per-source TP rate and per-source position and size error variance. It can be fit by EM on agreement within tracks, with no ground truth; the identifiability caveats of two-annotator models apply. No 3D precedent was found, so this is a proposal, not a finding.
- A simple, explainable per-track quality score could combine:
  - track length in seconds;
  - the fraction of frames with both sources;
  - motion-state consistency (low centroid variance for static tracks; smoother residual for dynamic tracks — the LISO/Offline-Poly smoothing residual is a free localization-noise estimate);
  - size dispersion around the robust size;
  - points in the box (3DAL);
  - nested-box overlap (DetZero).

  It should be calibrated later against a small hand-labelled WATO set, LMD-style. The MS3D++ and UA3D evidence suggests feeding it as a soft per-box (or per-coordinate) loss weight to the student rather than as a hard threshold.
- Routing to human review: rank tracks by low quality score × high impact (long, near, class car or pedestrian), as LMD did with its top-100 error proposals.

### Gaps
- No training-free, ground-truth-free calibration method validated for 3D auto-labels was found. Learned quality heads (DetZero, CTRL, LMD) all need some ground truth.
- No 3D-box Dawid–Skene application was found. Bayesian Detector Combination and REC✓D ([arXiv 2508.06556](https://arxiv.org/html/2508.06556v2), image-focused) were only seen in search summaries.
- The papers seen in search summaries only (side-aware uncertainty, Bayesian Detector Combination, WACV 2026 timing errors, adaptive pseudo-label selection) were not read in full; their numbers are unverified here.
