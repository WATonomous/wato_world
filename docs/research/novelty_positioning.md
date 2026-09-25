# Research Positioning: Prior Art and Where Novelty Could Come From

**Written**: 2026-09-24. **Status**: notes, not a plan.

How wato_world's multi-stage, multi-source design compares to published
auto-labelers, and which parts could be a research contribution rather than
integration work.

**Short version:** multiple stages and multiple proposal sources are not novel
on their own. Every stage has close prior art, and VESPA (2026) matches the
camera side almost step for step. Novelty has to come from how the sources are
combined (per-point provenance already sets this up) and from problems our rig
forces on us that benchmark datasets hide.

---

## Where each stage already exists

| Work | What it does | What it doesn't do |
|---|---|---|
| UniLiPs (arxiv 2601.05105) | Moving points = scan/map disagreement after IWU; HDBSCAN → Kalman → spline-smoothed boxes; camera labels lifted to LiDAR (Eq. 1) | Boxes appear to come only from moving points, so parked cars get none. One LiDAR. Box results are on their own highway dataset |
| SAM4D (arxiv 2506.21547) | Trained promptable model that segments camera + LiDAR jointly; its data engine auto-labels *masks* | Outputs masks, not boxes. Needs training |
| VESPA (CVPR 2026 Findings) | GroundingDINO + SAM → project onto LiDAR → DBSCAN + L-shape → DINOv2 tracking → LLM size priors | No confidence or provenance per box. Authors report it is "highly sensitive to image resolution" (4.72 mAP on their own dataset) |
| ZOPP (NeurIPS 2024) | GroundingDINO + SAM + DeAOT across cameras; parallax-occlusion filter; point-completion network; L-shape fitting | Objects must be found by a camera first |
| UNION (NeurIPS 2024) | HDBSCAN + ICP-Flow find movers; static objects kept if they look like movers (DINOv2) | No text-named classes |
| MS3D++ | Ensemble of pretrained detectors, kernel-density box fusion, temporal refinement | Detectors are its only source |
| SAL-4D (CVPR 2025) | Video-segmentation tracklets + CLIP tokens lifted into 4D LiDAR | Segmentation, not boxes |

Mapped onto our pipeline:

- `perception_2d` + `semantic_lifting` + planned DINOv2 tracking ≈ VESPA / ZOPP
- `lidar_preprocessing` Steps E/F ≈ UniLiPs IWU + Chen et al.
- `label_refinement` ≈ LabelFormer

A reviewer will read "all of these combined" as integration.

---

## What is already distinct in the code

Real, but adaptations. On their own they read as engineering unless they move
an end-to-end number.

- **Recall and precision kept separate, with every heuristic's verdict kept
  per point** (`source_bits` in
  `lidar_preprocessing/.../motion_proposals/_core.py`). UniLiPs thresholds at
  τ_s and Chen relabels hard; we keep soft cluster features and decide
  downstream.
- **IWU adapted for sparse, multi-LiDAR rigs.** On nuScenes HDL-32E, the
  paper's decay rule tested per pixel evicts 54% of the map; the shipped
  windowed rule evicts 7% at 2× enrichment (`lidar_mos_guidance.md`). A
  reproducibility note, not a refutation — UniLiPs' map may be denser.
- **Parked and moving objects both get boxes** (via SLF and the LiDAR
  detector). UniLiPs only boxes movers.
- **Asynchronous-sensor rigor.** Poses are looked up at each image's own
  timestamp; using the sweep pose instead misplaces a point 30 m away by 17 cm
  median (`semantic_lifting_design.md`).

---

## Candidate contributions (ranked)

### 1. Label-free source reliability → calibrated per-box confidence

What the multi-source design uniquely enables. Our sources are heterogeneous:
LiDAR geometry bits, zero-shot camera models, detectors trained on other
datasets, SLF fits. Treat each as a noisy labeling function and fit a label
model (Dawid–Skene / Snorkel style) that learns per-source accuracy from
agreement patterns, with no ground truth.

- MS3D++ fuses only detectors; VESPA and UniLiPs output boxes with no
  confidence.
- The hard, publishable part: sources aren't independent. Under
  `--seg union`, `SEG_DYNAMIC` is built from AW + MF-MOS evidence, so naive
  agreement double-counts. The label model has to handle dependencies.
- Payoff: confidence-weighted student training, and a human review queue
  (single-source boxes first).
- Evidence needed: calibration (ECE) against nuScenes GT; student AP with
  weighted vs hard labels; leave-one-source-out ablations. The ablations are
  cheap because `ProposalRow.provenance` already exists
  (`wato_common/schemas.py`).

### 2. Continuous-time track fitting across unsynchronized sensors

Fit one shape per track plus a spline trajectory. Evaluate each camera mask at
that image's timestamp and each LiDAR point at its own measurement time.

- Combines SLF (per-frame, synced KITTI), LabelFormer's "size shared, pose per
  frame" (but training-free), and UniLiPs' spline (LiDAR only).
- Fixes the Δt·v error for moving objects that `semantic_lifting_design.md`
  currently accepts ("Dynamic-point handling").
- Nearest competitors: Gaussian-splatting scene reconstruction (Street
  Gaussians, OmniRe) that optimizes actor poses. Position against them as a
  cheap, training-free labeler.
- Caveat: nuScenes hides most of this — its cameras fire as the LiDAR sweeps
  past them. Convincing evidence needs some hand-labeled WATO tracks.

### 3. LiDAR → camera re-prompting (closing the loop)

Project Step F moving clusters (`motion_score > 1`) into images as SAM2
prompts to recover objects GroundingDINO missed; the resulting masklets then
classify the class-less `lidar_mos` boxes.

- UniLiPs only sends semantics into geometry (C(m)); SAM4D prompts across
  modalities but is a trained model.
- Clean experiment: degrade images (resolution, night) and show graceful
  degradation where VESPA collapses.
- Weaker alone (LiDAR-prompted SAM exists in various forms); strongest as the
  robustness section of #1.

### Not recommended as a headline

Self-training LabelFormer on our own pseudo-labels (`labelformer_guidance.md`
Option B). It tends to relearn the bootstrap's errors, and the self-training
line (OYSTER, CPD) already covers that ground.

---

## Decision to make first: which comparison class

A LiDAR detector pretrained on labeled data in `proposal_generation` changes
which papers we are compared against. If CenterPoint is trained on nuScenes
labels, we can't claim annotation-free and can't fairly evaluate on nuScenes.

- **Annotation-free**: no supervised detector. Compare against VESPA, UNION,
  AnnofreeOD on nuScenes (all report there).
- **Cross-domain**: detectors trained only on other datasets (Waymo / KITTI /
  ONCE), evaluated on nuScenes + WATO — the MS3D++ protocol. Candidate #1 fits
  best here: nobody has fused domain-shifted detectors, zero-shot VFMs and
  geometry.

`proposal_generation`, `tracking` and `label_refinement` are still stubs, so
all three candidates live in code not yet written. Build `proposal_generation`
around per-source reliability from the start rather than NMS-then-fuse.

---

## References

- UniLiPs — https://arxiv.org/abs/2601.05105
- SAM4D — https://arxiv.org/abs/2506.21547
- VESPA — https://arxiv.org/abs/2507.20397 (CVPR 2026 Findings)
- ZOPP — https://arxiv.org/abs/2411.05311
- UNION — https://arxiv.org/abs/2405.15688
- MS3D++ — https://arxiv.org/abs/2308.05988
- SAL-4D — https://arxiv.org/abs/2504.00848
- Fusion4DAL — https://link.springer.com/article/10.1007/s11263-025-02370-1
- AnnofreeOD — https://openaccess.thecvf.com/content/ICCV2025/html/Sun_AnnofreeOD_Detecting_All_Classes_at_Low_Frame_Rates_Without_Human_ICCV_2025_paper.html
- Drones as Annotators (related-work survey) — https://arxiv.org/abs/2609.06819
