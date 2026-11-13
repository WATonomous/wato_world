# YOLO confidence threshold and mask tracking

Researched 2026-11-11. External sources below are first-party documentation and
source code. Recommendations are **unmeasured**; this note does not establish an
optimal threshold or change configuration.

## What the threshold does

The repo pins `ultralytics==8.4.175` in
[`docker/requirements/perception_2d.txt`](../../docker/requirements/perception_2d.txt).
[`YoloDetector.detect()`](../../src/perception_2d/src/wato_perception_2d/models/yolo_detector.py)
passes `conf` directly to `model.predict()` and copies `boxes.conf` into each
`Detection.score`; there is no local calibration step.

In the pinned version, the end-to-end output branch retains rows with
`score > conf_thres`, filters classes, applies `max_det`, and returns before
IoU-based NMS. The predictor selects this branch using the model's `end2end`
property. Consequently, for the current end-to-end YOLO26x path, changing
`detection.yolo.iou` does **not** remove overlapping boxes. It is not a substitute
for adjusting confidence or the track birth gate. This conclusion comes from
the pinned implementation, rather than assumptions about other YOLO versions.
Sources: [v8.4.175 NMS implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.175/ultralytics/utils/nms.py#L65-L74),
[v8.4.175 detection predictor](https://github.com/ultralytics/ultralytics/blob/v8.4.175/ultralytics/models/yolo/detect/predict.py#L47-L58).

Ultralytics describes confidence as a model score used to accept or rank
predictions. Raising its cutoff generally trades recall for precision; it can
remove faint, small, and occluded objects as well as false positives. A `0.5`
score is not evidence of a measured 50% success rate on WATO: that interpretation
requires calibration on representative held-out data. Nor does a high mAP
establish calibration, because ranking metrics depend on score order.
Source: [Ultralytics confidence glossary](https://www.ultralytics.com/glossary/confidence).

## Why the downstream effect is larger than fewer boxes

The WATO and default configs currently use `conf: 0.25`, redetection every two
images, `birth_score: 0.0`, `support_keyframes: 5`,
`support_min_area: 0.01`, `reprompt_score: 0.8`, and `backfill_frames: 20`.
See [WATO config](../../src/perception_2d/config/perception_2d.wato.yaml) and
[`_Support` / tracking loop](../../src/perception_2d/src/wato_perception_2d/models/sam2_tracker.py).

| Use of a retained detection | Effect of raising global `conf` |
| --- | --- |
| Prompt for a new masklet | Removes some bad seeds, but also real objects whose strongest box is below the cutoff. |
| Support for an existing masklet | Removes keep-alive evidence. A mask whose bounding box covers at least 1% of the image can end after five unsupported keyframes from a full counter, nominally ten images with redetection every two. Smaller masks are exempt from this particular support decay. |
| Class vote | Removes evidence from the score sums used to choose the masklet class. The stored track score is the mean of supporting detections for the winning class, not necessarily its birth score. |
| Re-prompt correction | The separate 0.8 requirement remains the immediate score gate throughout the suggested 0.25–0.6 sweep. Track lifecycle changes can still affect whether correction happens. |
| Reverse backfill | A later accepted seed may recover up to twenty earlier images, subject to stop rules. This cannot recover an object that never gets an accepted seed, and recovery is not guaranteed. |

SAM2 is a promptable segmenter with video memory. Its video prompt API takes
coordinates/masks and an object ID; it does not take the detector's semantic
class. In this repo the detector supplies classes, while SAM2/DAM4SAM propagates
the prompted region. **Inference:** a coherent, persistent mask is therefore
not independent proof that its seed was a real car/person or had the correct
class. Segmenter quality scores should not be substituted for semantic
precision. Sources: [Meta SAM2 description](https://github.com/facebookresearch/sam2#readme),
[Meta video prompt API](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_video_predictor.py#L161-L229),
[local prompt and label assignment](../../src/perception_2d/src/wato_perception_2d/models/sam2_tracker.py).

Existing repo measurements already motivate caution: the four-clip support
study found real distant cars first boxed at 0.25–0.4 and then only intermittently;
confirmation rules kept only 26–48% of their real keyframe-masks. These are
earlier observations, **not a controlled global-confidence sweep**.
Source: [perception_2d measurements, detector support study](../perception_2d_measurements.md#masks-on-a-full-bag-2026-10-31).

## Unmeasured experiment recommendation

1. Sweep global `conf` over **0.25, 0.35, 0.4, 0.5, 0.6**, fixing weights,
   image size, tracker settings, inputs, and camera coverage. Start with 0.35
   as a modest global ablation; none of these candidates is an established
   improvement. Use complete
   pipeline reruns for lifecycle effects; filtering existing masks by their
   final score does not reproduce a higher detector threshold.
2. Measure box precision `TP/(TP+FP)` and recall `TP/(TP+FN)` against labelled
   frames at a specified matching IoU, stratified by class, camera, size, and
   occlusion. For masks, assess false-positive duration/area, object coverage,
   fragmentation, and first/last observed time; also inspect downstream lifted
   labels. Counts alone cannot identify accuracy. Precision, recall, F1, and
   AP measure different things: AP integrates a precision–recall curve.
   Source: [Ultralytics metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).
3. Read operating-point curves, rather than assuming the validation summary's
   precision/recall describes `conf=0.25`: the pinned metrics code reports
   precision and recall at the smoothed mean-F1 maximum.
   Source: [v8.4.175 metric selection](https://github.com/ultralytics/ultralytics/blob/v8.4.175/ultralytics/utils/metrics.py#L805-L817).
4. If false births dominate but weak boxes are useful support, compare keeping
   detector `conf=0.25` with a separate stricter birth policy. The existing
   `birth_score` is a **tentative/confirmation threshold**, not a hard rejection
   threshold: weaker detections still start tentative tracks and need three
   consecutive supported keyframes under current defaults. Test its loss of
   distant-object recall explicitly. A true hard birth cutoff with low-score
   support preserved would require a distinct policy/code change. Neither
   alternative has been measured here.

## Local artifact analysis (2026-11-11)

Counted all accepted `birth` events in the six per-camera `mask_quality/*.json`
files for each `nusc0061_new` and `nusc1100_new` arm under
`data/artifacts/eval_runs/2026-11-07_mask_policy/`. These are scene chunks
`0000`, generated on November 7 with YOLO26x confidence 0.25, DAM4SAM,
`birth_gate: untracked_area`, and 20-frame backfill. The manifests confirm
these settings. There were 824 / 289 tested births, of which 573 / 168 passed
the birth gate. Accepted events record the original seed score separately
from the final, averaged masklet score.

| Cutoff | Baseline accepted seeds excluded, scene 0061 | Scene 1100 |
| --- | --- | --- |
| 0.25 | 0 / 573 (0%) | 0 / 168 (0%) |
| 0.35 | 141 / 573 (24.6%) | 55 / 168 (32.7%) |
| 0.40 | 194 / 573 (33.9%) | 74 / 168 (44.0%) |
| 0.50 | 288 / 573 (50.3%) | 104 / 168 (61.9%) |
| 0.60 | 364 / 573 (63.5%) | 117 / 168 (69.6%) |

**These percentages are not false-positive reductions or recall losses.**
They count baseline seed scores at or below the proposed cutoff. A rerun can
seed the same object later, accept births the baseline gate refused, change
detector support and masks, and alter downstream identity. No global-threshold
rerun or new ground-truth accuracy measurement was performed for this audit.
The JSON [seed audit](yolo_confidence_seed_audit.json) preserves exact input
paths, effective configs, threshold counts, and per-class counts.

Low scores preferentially select smaller boxes here: at 0.35, median seed-box
area below/above the cutoff is 933 / 2,107 px² on 0061 and 1,812 / 6,048 px²
on 1100. Size is an observed association, not proof that these seeds are real
or distant. At that cutoff, 58 of 175 car seeds on 0061 and 32 of 97 on 1100
are excluded; class-wide uniform thresholds therefore merit a class/size
breakdown.

The earlier WATO PANO_SS study documented 122 accepted seeds: 56 at 0.25–0.4,
23 at 0.4–0.6, 19 at 0.6–0.8 and 24 at ≥0.8 (subtract refused births from
tested births in the published bins). Thus about 46% began below 0.4 in that
one camera. This is also a seed distribution, not an accuracy measurement.
Historical hood/rack false positives scored 0.25–0.42, but the current ego
mask and detector-support policy already address many of these; a stricter
cutoff's marginal benefit must be measured on the current baseline.
Source: [existing WATO measurements](../perception_2d_measurements.md).

For the controlled sweep, rerun perception through label refinement in
separate roots. Use existing `eval_proposals.py`, `eval_tracks.py`, and
`eval_refined.py` for downstream 3D precision/recall, ID switches and
fragmentation at the same 2 m matching distance, broken down by range.
Those scripts measure 3D boxes; evaluating raw detector precision needs
labelled 2D boxes and a specified IoU match separately. WATO requires manually
reviewed or labelled data before reporting a genuine precision/recall curve.
Start with 0.35 against 0.25, then consider 0.4 if the measured recall cost is
acceptable. Leave the shipped configuration unchanged until that comparison.
