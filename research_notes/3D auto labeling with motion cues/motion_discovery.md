# Motion cues for label-free 3D object discovery and auto-labeling (estimation, persistence, downstream roles, false-mover rejection)

Scope note: sources read November 2026. Publication years are given per method. Where the assignment's framing differs from the sources, that is flagged:
- DRIFT is NeurIPS 2023, not ICLR 2023 (PDF header: "37th Conference on Neural Information Processing Systems (NeurIPS 2023)") — [DRIFT](https://arxiv.org/pdf/2310.19080).
- SeMoLi is semi-supervised: its edge classifier is trained on labelled moving objects — [SeMoLi](https://arxiv.org/html/2402.19463).
- MS3D++ is multi-source domain adaptation from detectors pretrained on labelled datasets. It is not label-free — [MS3D++](https://arxiv.org/pdf/2308.05988).
- OYSTER uses temporal consistency (tracklet length), not motion — [OYSTER](https://arxiv.org/pdf/2311.02007).
- The ICP-Flow repo says "This version requires a GPU" and "The CPU-only version has not been released yet" — [ICP-Flow repo](https://github.com/yanconglin/ICP-Flow).

## Q1. Per-object motion estimation offline without labels: scene flow vs cluster-then-track; which training-free flow methods suit offline CPU / modest GPU; accuracy on sparse LiDAR and pedestrians; how SeMoLi, LISO, Najibi and UNION get movers from flow

### Takeaway
Every strong published pipeline estimates motion per cluster by rigid registration of the cluster's shape across frames (ICP-Flow, NSFP++, UNION's modified ICP-Flow), or by flow fed into tracking. None uses centroid displacement. Their thresholds are absolute: 0.5 m (UNION), 1 m residual flow (LISO), 0.2 m/s static pre-filter (Najibi, SeMoLi). Clustering per-point flow alone performs very poorly (AP ≈ 0.01–0.03 on Waymo). Gains come from tracking, temporal filtering, shape registration and then self-training. On compute, the best test-time-optimised flows are GPU-heavy: EulerFlow takes 24 h per sequence, Floxels ~9 min per sequence on a T4, ICP-Flow ~3.3 s per frame pair on GPU. Even the best of them recover pedestrian motion worst (dnEPE ≈ 0.2 vs ≈ 0.1 for cars), and older ones (NSFP, ZeroFlow) capture < 30% of pedestrian motion.

### Cited Findings

**Baseline being improved (Chen et al. 2022, arXiv 2201.04501)**
- Chen labels an instance moving "if its accumulated trajectory is larger than its maximum side length of the associated bounding box". All points inside a moving box are then labelled moving — [Chen et al.](https://arxiv.org/pdf/2201.04501).
- Chen Table I (IoU_MOS, label generation): Octomap-based 13.6, Removert 15.7, ERASOR 19.1, Pfreundschuh's 34.5, Yoon's 15.1, full pipeline 74.2. LMNet trained on these auto labels gets 54.3 IoU vs 58.3 with manual labels, and 62.3 with extra auto-labelled KITTI-raw data — [Chen et al.](https://arxiv.org/pdf/2201.04501).

**ICP-Flow (Lin & Caesar, CVPR 2024): training-free cluster-then-register flow**
- Pipeline:
  - Ego-motion from given poses, or KISS-ICP.
  - Patchwork++ ground removal.
  - HDBSCAN on the fused non-ground points of both scans (min_cluster_size = 20; the top 200 clusters by size).
  - Pairing inside a 3.33 m × 3.33 m window (τz = 0.1 m), equal to 120 km/h over 0.1 s.
  - Histogram-of-translations initialisation with 0.1 m bins, then point-to-point ICP with a 0.1 m inlier threshold.
  - A pair is rejected if inlier ratio r < 0.2 or mean distance d > 0.2 m. Hungarian matching was only marginally better than argmin.
  - [ICP-Flow](https://arxiv.org/html/2402.17351)
- Runtime per scan pair: ICP-Flow 3.3 s (Waymo) and 3.5 s (AV2); distilled feed-forward net 21 / 29 ms; NSFP 90 / 26 s; Chodosh et al. 93 / 35 s; FastNSF 0.5 s. Hardware is "the same device" as training (4× V100 + Xeon W-2245). The paper does not split CPU from GPU — [ICP-Flow](https://arxiv.org/html/2402.17351).
- Dynamic-foreground EPE / Acc-S / Acc-R:
  - Waymo: 0.0799 m / 76.10 / 88.53.
  - AV2: 0.1653 m / 48.61 / 70.70. Chodosh et al. are better on AV2 dynamic at 0.1311 m.
  - nuScenes (32-beam, ~25k points per scan): 0.1445 m / 49.59 / 66.02.
  - The distilled FNN loses dynamic accuracy, e.g. nuScenes dynamic 0.1850 m / 21.52 / 45.35.
  - [ICP-Flow](https://arxiv.org/html/2402.17351)
- Stated limitations [ICP-Flow](https://arxiv.org/html/2402.17351):
  - Over- and under-segmentation by ground removal and clustering.
  - Errors when similar objects are near each other.
  - Fast objects leaving the range.
  - ICP "struggles to match sparse clusters".
  - No one-to-one matching.
  - The rigid assumption fails on buses and trucks.
  - Geometry only.
- Mean dynamic normalised EPE on the AV2 2024 challenge: ICP-Flow 0.3309 — [Floxels](https://arxiv.org/html/2503.04718).

**Najibi et al. (ECCV 2022) — "Motion Inspired Unsupervised Perception and Prediction"**
- Pre-flow static removal: ground by RANSAC plane, then a point is marked static if its Chamfer distance to earlier frames is under a threshold, set small "to have a high precision in this step (i.e. 20 cm/s)". Without this, flow on "walls, buildings, trees" adds noise — [Najibi et al.](https://arxiv.org/pdf/2210.08061).
- NSFP++ splits dynamic points into connected components and fits flow per cluster:
  - Local consistency regulariser α = 0.1.
  - Expanded box query to catch fast movers, then pruning.
  - [Najibi et al.](https://arxiv.org/pdf/2210.08061)
- NSFP++ vs NSFP on WOD val: EPE3D 0.017 vs 0.455 m; Acc5 95.05 vs 23.65%. Speed-bucket mIoU is 0.586 vs 0.244; the 0–3 m/s bucket is 0.989 vs 0.657. Najibi says NSFP gives "false negative flows over pedestrians" and "excessive false positive flows over static objects (e.g., buildings)" — [Najibi et al.](https://arxiv.org/pdf/2210.08061).
- Proposals keep points with |flow| ≥ |f|min, then run DBSCAN separately on locations and on flows and intersect the two. Points share a cluster only if close in both space and motion, which splits close-by objects. Box heading comes from the cluster's mean flow. The tracker advances boxes with flow instead of relying on Kalman velocity — [Najibi et al.](https://arxiv.org/pdf/2210.08061).
- Detector trained on the auto labels (PointPillars, WOD, 3D mAP L1 at IoU 0.4):

  | Auto labels | 3D mAP L1 |
  |---|---|
  | No flow + clustering | 4.7 |
  | No flow + AML (tracking, registration) | 9.6 |
  | Unsupervised flow + clustering | 30.4 |
  | Unsupervised flow + AML | 42.1 |
  | Supervised flow + clustering | 30.8 |
  | Supervised flow + AML | 49.9 |

  [Najibi et al.](https://arxiv.org/pdf/2210.08061)

**LISO (Baur et al., ECCV 2024)**
- Ground removed with JCP, ego-motion from KISS-ICP, flow from self-supervised SLIM (BEV range extended to 120 × 120 m) — [LISO](https://arxiv.org/html/2403.07071).
- Keeps points with residual flow > 1 m, then DBSCAN in 6-D (xyz + flow) with ε = 1.0 and minPts = 5. Boxes are dropped if l/w > 4.0, l·w < 0.35 m² or l·w·h < 0.5 m³ — [LISO](https://arxiv.org/html/2403.07071).
- Flow-based tracker run forward and backward with a 1.5 m matching threshold (OYSTER uses 5 m). It drops tracks shorter than 4 steps or with median confidence < 0.3. Tracks longer than 3 m are jerk-smoothed. The stated goal is "high precision (and potentially low recall)" — [LISO](https://arxiv.org/html/2403.07071).
- Baselines on WOD (AP@0.4, movable, BEV / 3D): DBSCAN on scene flow 0.026 / 0.010; RSF 0.030 / 0.020 — [LISO](https://arxiv.org/html/2403.07071).

**SeMoLi (Seidenschwarz et al., CVPR 2024; semi-supervised)**
- Motion cues [SeMoLi](https://arxiv.org/html/2402.19463):
  - Removes points with |v| < 0.2 m/s, using Chamfer distance between t and t+4.
  - Per-point 25-step trajectories from a Neural-Prior trajectory optimiser, which takes ~7 min per timestamp on a 12 GB GPU.
- A message-passing network over a kNN graph classifies edges, followed by correlation clustering. It is trained on labelled moving instances — [SeMoLi](https://arxiv.org/html/2402.19463).
- Pseudo-label F1 @0.4 on Waymo:

  | Clustering | Motion | F1 @0.4 |
  |---|---|---|
  | DBSCAN | estimated | 10.4 |
  | DBSCAN++ | estimated | 16.2 |
  | SeMoLi (10% labels) | estimated | 53.1 |
  | DBSCAN++ | ground truth | 69.0 |
  | SeMoLi (10% labels) | ground truth | 73.5 |

  With estimated motion, clustering quality dominates; with ground-truth motion, simple clustering works well — [SeMoLi](https://arxiv.org/html/2402.19463).

**UNION (Lentsch, Caesar, Gavrila, NeurIPS 2024)**
- RANSAC ground fit with a 5 cm inlier threshold; points > 30 cm above the plane count as non-ground. HDBSCAN on 15 aggregated non-ground scans (±7), with min cluster size 16 and cluster_selection_epsilon 0.50 m — [UNION](https://arxiv.org/html/2405.15688).
- Motion: each proposal's points are split by time step ("undoing the aggregation"). A modified ICP-Flow then estimates 2-D translation + yaw only, assuming objects move on the ground plane. A proposal is dynamic if it moved ≥ 0.50 m — [UNION](https://arxiv.org/html/2405.15688).
- The paper reports no per-stage runtime; experiments ran on 6 A100 80 GB GPUs — [UNION](https://arxiv.org/html/2405.15688).

**Training-free / self-supervised scene flow: cost and accuracy**

| Method | Year / venue | Compute | Mean dyn. normalised EPE (AV2 2024) |
|---|---|---|---|
| EulerFlow | ICLR 2025 | 24 h per AV2 sequence, 1× V100 | 0.130 (ties as best on pedestrians) |
| Floxels-13 | CVPR 2025 | 9.28 s/frame, ~24 min per sequence, T4 | 0.154 |
| Floxels-5 | CVPR 2025 | 3.52 s/frame, ~9 min per 156-scan sequence, T4 | 0.168 |
| SeFlow | ECCV 2024 | 48 ms inference; trained 30 h on 8 A100 | 0.309 |
| ICP-Flow | CVPR 2024 | 3.3–3.5 s per pair (GPU) | 0.3309 |
| NSFP | 2021 | 26–90 s/frame | 0.422 |
| ZeroFlow XL 5x | ICLR 2024 | 21–29 ms inference after distillation | 0.439 |

- EulerFlow details:
  - Test-time optimisation of a space-time ODE with a neural prior; it is "not pretrained on any datasets".
  - Losses: truncated Chamfer over ±3 frames plus cycle consistency (α = 0.01).
  - "optimizing EulerFlow for a single Argoverse 2 sequence takes 24 hours on one NVIDIA V100 16GB GPU".
  - The largest gains are on small classes: pedestrians and wheeled VRUs on AV2, pedestrians and cyclists on Waymo.
  - [EulerFlow](https://arxiv.org/html/2410.02031)
- Floxels details:
  - 0.5 m voxel flow grid, multi-scan distance-transform loss with constant-velocity propagation (5 scans default).
  - A DBSCAN cluster-consistency loss (ε = 0.5, min_points = 4, deliberately over-segmenting); removing it causes a large drop.
  - Timing on a T4 as in the table. The paper claims ~60–140× faster than EulerFlow. On nuScenes val its per-frame time (4.35 s) is close to NSFP-8 (3.43 s) and FastNSF-8 (5.93 s).
  - [Floxels](https://arxiv.org/html/2503.04718)
- AV2 test per-class dynamic normalised EPE [Floxels](https://arxiv.org/html/2503.04718):

  | Method | Car | Other vehicle | Pedestrian | Wheeled VRU | Mean |
  |---|---|---|---|---|---|
  | EulerFlow | 0.093 | 0.141 | 0.195 | 0.093 | 0.130 |
  | Floxels-13 | 0.112 | 0.213 | 0.195 | 0.096 | 0.154 |
  | Floxels-5 | 0.119 | 0.194 | 0.243 | 0.113 | 0.168 |

- Pedestrian evidence from Khatri et al. (ECCV 2024) [Khatri et al.](https://arxiv.org/html/2403.04739):
  - "fewer than 1% of points are from pedestrians and other vulnerable road users", vs ~15% from cars and other vehicles, so threeway EPE hides pedestrian failure.
  - NSFP and ZeroFlow XL 5x describe "less than 30% and 20% of pedestrian motion, respectively".
  - TrackFlow (supervised detector + AB3DMOT) gets pedestrian dnEPE 0.397 and mean 0.287.
- SeFlow (ECCV 2024) [SeFlow](https://arxiv.org/html/2407.01702):
  - Supervision: DUFOMap ray-casting dynamic labels plus HDBSCAN on dynamic points for a cluster loss.
  - CPU preprocessing averages "50ms/frame" for DUFOMap and "500ms/frame" for HDBSCAN.
  - AV2 test threeway EPE 0.0628, dynamic foreground 0.1525, at 48 ± 4 ms on an RTX 3090.
  - Failures: "pedestrians near static structures", distant sparse objects "easily ignored by both the voxelization process and clustering", and false flow where ground is not fully removed.
- ZeroFlow (ICLR 2024) [ZeroFlow](https://arxiv.org/html/2305.10424):
  - Distils NSFP pseudo-labels into FastFlow3D; 29.33 ms inference.
  - NSFP labelling of the AV2 train split "required a total of 753 hours of NVidia Turing generation GPU time".
  - "ZeroFlow inherits the biases of its pseudo-labels".
- UniFlow (arXiv Nov 2025, abstract and search snippet only) claims near-parity with EulerFlow at a fraction of the runtime, via a supervised multi-dataset model. It is not label-free — [UniFlow](https://arxiv.org/pdf/2511.18254).

**Sparse LiDAR**
- LISO calls nuScenes, a 32-layer sensor, the hardest dataset for self-supervised methods. LISO-CP gets 0.109 AP / 0.224 NDS there, vs supervised CenterPoint 0.484 AP — [LISO](https://arxiv.org/html/2403.07071).
- MODEST without ephemerality "not able to extract enough signals to train a decent detector on nuScenes where LiDAR is much sparser than Lyft" — [MODEST](https://arxiv.org/pdf/2203.15882).

### Inferences
- WATO's `motion_score` (net centroid displacement ÷ max box side) is Chen's criterion. Every later method replaces centroid motion with a registration-based or flow-based estimate. Registration (ICP with a histogram initialisation, accepting only r ≥ 0.2 and d ≤ 0.2 m) estimates the transform of the shape, not the visible centroid. A cluster whose visible part grows or shrinks as the ego passes should register to near-zero translation, or fail the inlier test, rather than show displacement. This directly targets the "visibility drift fakes motion" finding.
- For a CPU-only stage, ICP-Flow's algorithm is the most portable choice: HDBSCAN (already in Step F) plus per-pair histogram initialisation plus small ICPs. The official code needs a GPU, so this means a reimplementation (e.g. numba or Open3D ICP). Its per-pair cost on GPU is 3.3 s at ~200 clusters, so CPU time on WATO's ~71 clusters per frame is unknown. Test-time-optimised flows (EulerFlow, Floxels, NSFP) are impractical on CPU. A modest GPU running Floxels at ~9 min per 156-frame sequence would be the cheapest high-accuracy option if a GPU stage were allowed.
- Use absolute thresholds for small objects. UNION's absolute 0.5 m, LISO's 1 m residual flow, and SeMoLi's and Najibi's 0.2 m/s static filter all avoid normalising by box size. A size-normalised score penalises nothing for large objects but makes fragmented pedestrians (tiny boxes, short tracks) noisy.
- Najibi's location-∩-flow DBSCAN and LISO's 6-D (xyz + flow) DBSCAN are cheap ways to split adjacent movers once per-cluster motion exists. Both need a per-point or per-cluster velocity, which Step F lacks today.
- Because pedestrian motion is the weakest point even for the best flows, a geometry-only stage should not be expected to find pedestrians reliably. Camera masklets are the more promising seed; see Q4.

### Gaps
- No source gives CPU-only runtimes for ICP-Flow, Floxels or UNION's motion stage.
- No source evaluates any of these flows on VLP-16/VLP-32C rigs. nuScenes (32-beam) is the closest proxy.
- Let-It-Flow, VoteFlow and other 2024–2025 unsupervised flows were not checked.
- ICP-Flow's per-class (pedestrian) AV2 numbers exist only in challenge figures. EulerFlow says it "at least halv[es] the error on every class" vs ICP-Flow, but exact values were not retrievable from text.
- Najibi's |f|min value and DBSCAN parameters are in the supplementary, which was not read.

## Q2. Multi-scan aggregation of movers (UniLiPs and others): does it fix fragmentation of small or far movers?

### Takeaway
Multi-frame aggregation is used, but in three different ways:
- Aggregate everything, then split by time to estimate motion (UNION, 15 scans).
- Aggregate only static-confirmed points for shape (CPD, 11 frames, moving points removed from non-current frames).
- Aggregate a few frames of moving candidates (UniLiPs, 3 scans, ≥ 2 neighbours within 1 m; ViLGOD, HDBSCAN on xyz + PP-score + Δt).

None publishes a small- or far-object ablation of the aggregation itself. Small and far movers stay the main failure everywhere:
- LISO pedestrian AP is 0.029 BEV.
- SeFlow names distant sparse objects as a failure.
- MODEST pedestrian recall needs self-training to rise from 10.2 to 42.5.

The documented fixes for far and sparse objects are temporal shape registration along a track (Najibi, ViLGOD, CPD prototypes) and detector self-training, not denser per-frame clustering.

### Cited Findings
- UniLiPs (arXiv Jan 2026) [UniLiPs](https://arxiv.org/html/2601.05105):
  - Marks as moving the scan points with no correspondence in the IWU-refined map, requiring "at least 2 other moving candidates in a neighborhood of 1 meter".
  - Clusters three pose-aligned consecutive scans of moving points with HDBSCAN; no eps or min-points values are given.
  - Fits minimum enclosing cuboids with a minimum size and PCA yaw, tracks with a constant-velocity + yaw Kalman filter, then spline-smooths each trajectory.
  - It does not discuss small or distant objects, pedestrians, or vegetation and wall false positives. Detection is evaluated on one highway dataset.
- UniLiPs Table 5, pseudo-box mAP / NDS on its highway set: ICP-Flow 7.2 / 11.4, LISO 21.1 / 40.9, UniLiPs 31.0 / 45.2. Ablation: without spline smoothing 23.9 / 33.2; without IWU 11.7 / 30.8 — [UniLiPs](https://arxiv.org/html/2601.05105).
- UniLiPs detector trained on 20% GT + 80% pseudo, BEV / 3D AP at ±70 m: UniLiPs 29.19 / 25.33, oracle 32.44 / 30.13, ICP-Flow 9.44 / 3.20 — [UniLiPs](https://arxiv.org/html/2601.05105).
- A search summary lists UniLiPs as 3DV 2026; the lab page was not opened — [UniLiPs project](https://light.princeton.edu/unilips).
- UNION aggregates ±7 non-ground scans before HDBSCAN (min cluster 16, ε 0.50 m), then recovers motion by splitting the proposal's points per time step. The paper does not discuss smearing of movers in the aggregate — [UNION](https://arxiv.org/html/2405.15688).
- CPD (CVPR 2024) [CPD](https://arxiv.org/html/2404.16493):
  - Multi-Frame Clustering over 2n+1 frames, with 11 frames best. PP-score identifies moving points, which are removed from the non-current frames before concatenation.
  - Prototypes are built from high-quality complete objects, those with a CSS score > 0.8.
  - These prototypes replace the size of sparse or far labels and re-locate them using the densest boundary points.
  - The paper says MODEST "often misses distant, sparse objects" and OYSTER finds them "with inaccurate sizes and positions".
- CPD results on WOD val, 3D AP L2 [CPD](https://arxiv.org/html/2404.16493):

  | Class | IoU | CPD | OYSTER |
  |---|---|---|---|
  | Pedestrian | 0.3 | 18.01 | 3.52 |
  | Pedestrian | 0.5 | 13.22 | 0.14 |
  | Vehicle | 0.7 | 32.13 | 14.10 |

- ViLGOD (arXiv Aug 2024) [ViLGOD](https://arxiv.org/html/2408.03790):
  - Clusters non-ground points over n aligned frames with HDBSCAN on (x, y, z, PP-score, Δt), each frame subsampled by 1/n.
  - Drops segments with < 10 points, > 1 m above ground, or < 0.5 m tall.
  - Association is greedy within 1 m, relaxed to 5 m (with a 30% point-count tolerance) for fast objects.
- Najibi's shape registration aggregates each track's partial views by sequential ICP with a grid-search initialisation. The track's views together are "often close to amodal boxes" — [Najibi et al.](https://arxiv.org/pdf/2210.08061).
- SeMoLi reports pseudo-boxes that are "modal (tight around visible points)" and too small, which hurts strict-IoU results. It inflates boxes to a minimum size of (1, 1, 2) m on Waymo and (0.75, 0.75, 1.75) m on AV2 — [SeMoLi](https://arxiv.org/html/2402.19463).
- Pedestrian recall from motion-seeded pipelines:
  - LISO pedestrian AP@0.4 on WOD is 0.029 / 0.009 (BEV / 3D), vs supervised 0.513 / 0.413 — [LISO](https://arxiv.org/html/2403.07071).
  - SeMoLi pedestrian 3D-IoU recall is 41.8 on Waymo (below DBSCAN++ at 46.3) and 42.5 on AV2 — [SeMoLi](https://arxiv.org/html/2402.19463).
  - MODEST max pedestrian recall on Lyft is 10.2 at R0 and 42.5 at R10; supervised on Lyft is 53.8 — [MODEST](https://arxiv.org/pdf/2203.15882).
- M-detector post-processes occlusion "event" points as follows [M-detector](https://pmc.ncbi.nlm.nih.gov/articles/PMC10771495/):
  - DBSCAN on 0.3 m voxel centres removes isolated event voxels.
  - Each cluster's box is expanded to 2× its size.
  - Event voxels then grow recursively into neighbouring non-ground voxels within that box.
  - This region growing is a documented way to recover the rest of a fragmented mover from a few seed points.

### Inferences
- UniLiPs' 3-scan, ≥ 2-neighbours-within-1 m rule is a density filter on seeds. It removes isolated false movers but will also remove a far pedestrian with a few returns per scan. It helps precision, not small-object recall.
- For fragmentation, the most transferable fixes for WATO are:
  - Grow from moving seeds into adjacent non-ground points (M-detector style, bounded by an expanded box). This turns a fragment into a full object without lowering seed precision.
  - Accumulate each track's points in its own object frame via per-frame registration (Najibi, ViLGOD). This yields amodal size, so a 1 m-tall "wall-like" fragment can be judged on its accumulated shape.
  - Use CPD-style class prototypes (median complete-object size per class) to regularise sparse or far boxes.
- Three-frame clustering with Δt as a feature (ViLGOD) is a cheap way to give HDBSCAN more points for a slow pedestrian. With 3 LiDARs on WATO, a frame_sync frame already merges ~3 sweeps, so the marginal gain is unknown.

### Gaps
- No source ablates aggregation window size against small or far mover recall. UniLiPs does not report it; UNION gives no per-class class-agnostic recall.
- Exact HDBSCAN parameters for UniLiPs are only in its supplementary, which was not read.

## Q3. Persistence / ephemerality across repeated traversals (MODEST PP-score, DRIFT): applicability when the vehicle loops the same route

### Takeaway
PP-score is the normalised entropy of neighbour counts within r = 0.3 m across traversals. Clusters are dropped when their 20th-percentile PP-score exceeds 0.7. Measured gains:
- On Lyft, AP_BEV@0.25 (0–80 m) rises from 30.9 to 57.3 after 10 self-training rounds, vs a no-ephemerality baseline.
- Used as a filter during self-training, PP-score prevents drift.

PP-score needs ≥ 2 localised traversals of the same place. OYSTER found MODEST with only one traversal (short-term repeated views) much weaker (Pandaset AP@0.3 22.8 vs 43.5). ViLGOD computes PP-score over adjacent frames of a single drive. PP-score detects ephemeral objects, including parked cars that left between loops, which motion cannot. That makes loops on the same route a real recall source for static mobile objects, if poses can be put in one frame.

### Cited Findings
- PP-score definition [MODEST](https://arxiv.org/pdf/2203.15882):
  - N_t(q) = number of points within r of q in traversal t, and P(t;q) = N_t / Σ N_t′.
  - τ(q) = H(P(·;q)) / log T, with τ = 0 if no traversal has neighbours.
  - High τ means persistent background.
- MODEST supplementary parameters [MODEST](https://arxiv.org/pdf/2203.15882):
  - Aggregates one scan every 2 m over [0, 70] m ahead and uses "as many traversals as possible (≥ 2)", with r = 0.3 m.
  - Graph with K = 70 and r′ = 2.0 m; DBSCAN ε = 0.1, min_samples = 10.
  - Filter α = 20th percentile, γ = 0.7; clusters need ≥ 10 points and box volume in [0.5, 120] m³.
- MODEST results, AP_BEV / AP3D @0.25 [MODEST](https://arxiv.org/pdf/2203.15882):

  | Setting | 0–80 m | 50–80 m |
  |---|---|---|
  | Lyft, no PP-score (MODEST-PP R10) | 30.9 / 27.3 | — |
  | Lyft, MODEST R0 | 42.5 / 37.9 | 8.9 |
  | Lyft, MODEST R10 | 57.3 / 55.1 | 27.0 |
  | Lyft, MODEST R40 | 64.4 / 62.7 | 47.1 |
  | Lyft, supervised | 69.5 / 69.1 | — |
  | nuScenes, MODEST R10 | 11.8 / 6.6 | — |
  | nuScenes, supervised | 22.2 / 18.2 | — |

- PP-score filtering during self-training matters: 100% seed + 100% self-training gives 48.9 / 47.8 without it and 57.3 / 55.1 with it. The paper says "the improvement brought by PP score filtering cannot be compensated by additional rounds of self-training" — [MODEST](https://arxiv.org/pdf/2203.15882).
- MODEST's motivation for self-training: seed boxes miss parked cars "parked throughout all traversals". The detector then "is able to identify objects (like parked cars) that were missed in the ephemerality computation, because these seemingly static objects are nevertheless similar in shape to other moving objects" — [MODEST](https://arxiv.org/pdf/2203.15882).
- DRIFT (NeurIPS 2023) [DRIFT](https://arxiv.org/pdf/2310.19080):
  - Replaces slow self-training with reward-ranked fine-tuning. The reward combines a shape-prior reward (Gaussian prototypes over typical box sizes), an alignment reward using PP-score dynamic points, and common-sense filters.
  - "With only 9 hours of training, it outperforms both baselines that have been trained for days" (MODEST takes about 5 days).
- DRIFT results, mAP @IoU 0.5 over 0–80 m [DRIFT](https://arxiv.org/pdf/2310.19080):

  | Dataset | DRIFT | MODEST | Supervised (in-domain) |
  |---|---|---|---|
  | Lyft | 45.3 (120 ep) | 39.6 (600 ep) | 60.4 |
  | Ithaca365 | 35.1 (30 ep) | 27.1 (300 ep) | 51.5 |

- OYSTER's single-traversal test of PP-score, "multiple observations over a short period of time (i.e., the length of the snippet)", AP@IoU 0.3 [OYSTER](https://arxiv.org/pdf/2311.02007):

  | Dataset | PP score | PP + init-train | MODEST (1 traversal) | OYSTER |
  |---|---|---|---|---|
  | Pandaset | 6.4 | 28.4 | 22.8 | 43.5 |
  | AV2 | 5.5 | 24.4 | 20.6 | 35.4 |

  OYSTER: "MODEST (1 traversal) relies very heavily on multiple traversals".
- ViLGOD computes PP-scores from neighbours "across adjacent frames" within one drive and uses them as a clustering feature. It labels a track static if its boxes overlap its largest box or it lacks consistent motion, with α = 20% and δ = 0.7 on PP-scores — [ViLGOD](https://arxiv.org/html/2408.03790).
- CPD also uses PP-score, to identify moving points before multi-frame aggregation — [CPD](https://arxiv.org/html/2404.16493).

### Inferences
- WATO's `frac_persistent` resembles a within-drive PP-score. A true cross-traversal PP-score needs at least two passes of the same place minutes or more apart: a ring-road loop within one recording, or different recordings of the same route. That is when parked cars, waiting buses and parked bicycles change. Within one pass, a parked car is persistent by construction, so motion and short-term persistence cannot label it.
- Cross-traversal PP-score needs both passes in one metric frame. WATO's eidos poses are per bag; within one recording looping the same road, the SLAM frame is shared, though drift should be checked. Across recordings, a map-to-map registration (e.g. ICP of the two `global_static_map`s) would be needed. MODEST notes it "assume[s] good localization in the training set" and cites 1–2 cm RTK or 10 cm MCL.
- The γ = 0.7 at the 20th-percentile filter is cheap and CPU-only. It is a natural second veto on Step F gated tracks: a "mover" whose points are persistent across loops is a false mover, e.g. wall-sliding or vegetation.
- MODEST's nuScenes result (AP 11.8 vs 57.3 on Lyft) warns that a 32-beam sensor gives much weaker ephemerality seeds. WATO's centre VLP-32C sits at roughly that density.

### Gaps
- No source reports PP-score accuracy as a function of traversal count, time gap or localisation error.
- Not checked: Hindsight (ICLR 2022) and Rote-DA (NeurIPS 2022) by the same group, which use past traversals as features or adaptation signals.
- Whether WATO recordings contain enough repeat passes for this is a project measurement, not a literature question.

## Q4. Downstream roles of motion in published pipelines (seeds for self-training, appearance look-alikes, class gating, static/dynamic refinement switch, camera re-prompting), with gains

### Takeaway
In every published pipeline, movers are a high-precision seed, not the final label set. The large gains come from what the movers unlock:
- **Appearance generalisation to static look-alikes.**
  - UNION: AP 14.2 → 39.5 on nuScenes.
  - Najibi open-set: vehicle AP 0 → 65.4 with no vehicle labels.
- **Self-training.**
  - MODEST: 42.5 → 64.4 AP_BEV over 40 rounds.
  - LISO: +0.081 AP.
  - ViLGOD: 0.363 → 0.564 movable BEV AP.
- **Motion-gated class decisions.** MS3D++ takes pedestrians only from dynamic tracks because static poles look like pedestrians.
- **A static/dynamic switch for box refinement.**
  - Static: fuse boxes over history.
  - Dynamic: smooth trajectories and register shapes.
  - Smoothing adds 7.1 mAP in UniLiPs and +0.013–0.020 AP in LISO.

No published pipeline found re-prompts a camera segmenter with LiDAR movers.

### Cited Findings
**Seeds → appearance generalisation (training-free variant)**
- UNION [UNION](https://arxiv.org/html/2405.15688):
  - Computes DINOv2 ViT-L/14 features of projected LiDAR points, averaged per proposal.
  - Runs K-means (K1 = 20) over all proposals and keeps appearance clusters with ≥ 5% dynamic proposals as "mobile".
  - The 5% threshold is chosen from "an uptick at around 5 %" in a histogram, tuned "visually on 10 frames" of nuScenes. No sensitivity analysis over K is given.
- UNION class-agnostic results on nuScenes val (CenterPoint trained on pseudo-labels) [UNION](https://arxiv.org/html/2405.15688):

  | Labels | AP | NDS |
  |---|---|---|
  | HDBSCAN | 13.8 | 15.7 |
  | + motion estimation | 14.2 | 19.8 |
  | + appearance clustering (UNION) | 39.5 | 31.7 |
  | OYSTER | 9.1 | 11.5 |
  | LISO | 10.9 | 13.9 |
  | Supervised, 1% / 10% / 100% labels | 27.8 / 61.2 / 78.7 | — |

- UNION multi-class [UNION](https://arxiv.org/html/2405.15688):
  - Pseudo-classes come from a second K-means over mobile objects; "UNION-Xpc" means X pseudo-classes.
  - Each is mapped to a real class by cosine similarity to prototypes built from "three example instances from the nuScenes training dataset", described as "negligible supervision".
  - UNION-05pc: mAP 25.1, pedestrian AP 44.2, vehicle 31.0, cyclist 0.0.
  - Size-prior mapping gives HDBSCAN + SP pedestrian AP 0.4 and UNION + SP 3.7.
  - Stated limitation: rare mobile classes can merge into background appearance clusters and be discarded.
- UNION claims to avoid iterative self-training, which "often runs 5–10 sequential rounds in prior methods" — [UNION](https://arxiv.org/html/2405.15688).
- Najibi open-set leave-one-out (WOD, 3D AP) [Najibi et al.](https://arxiv.org/pdf/2210.08061):
  - Trained only on VRU labels: vehicle AP 0.0, mAP 44.4. Adding auto labels (AML) gives vehicle AP 65.4 and mAP 77.1.
  - Trained only on vehicle labels: VRU AP rises from 0.0 to 20.8 and mAP from 48.8 to 59.2.
  - Trajectory prediction with VRU-only human labels: minADE 9.53 → 2.15 and minFDE 22.31 → 5.55.
- UP-VL (Najibi et al., ICCV 2023; arXiv 2309.14491) [UP-VL](https://arxiv.org/html/2309.14491v1):
  - Clusters points on location + NSFP++ flow + OpenSeg vision-language features. ε_sf = 1.0 m/s gives moving-only labels; ε_sf = 0 covers all motion states.
  - Moving-only, class-aware: auto labels Veh 57.5 / VRU 29.8; detector with distillation Veh 76.9 / VRU 28.6.
  - All motion states: auto labels Veh 30.2 / VRU 14.7; detector without distillation mAP 27.6, with distillation 35.8.
  - Static objects are much harder to label from these cues than movers.
- ViLGOD (arXiv Aug 2024) [ViLGOD](https://arxiv.org/html/2408.03790):
  - Classifies each tracked cluster with CLIP ViT-B/16 on depth renders of the points from 4 views.
  - Waymo class-agnostic movable BEV / 3D AP: 0.363 / 0.323 vs LISO 0.292 / 0.211. Moving 0.280 / 0.240; static 0.327 / 0.311.
  - After self-training: 0.564 / 0.538. Pedestrian 0.388 / 0.359 vs LISO-CP 0.055 / 0.037.
  - Ablation, Waymo pedestrian BEV: corner alignment 4.9 → 8.0 (per paper text); class-label refinement roughly doubles it (to 16.3).

**Seeds → self-trained detector (geometry only)**
- LISO WOD ablation, CenterPoint, AP@0.4 BEV / 3D [LISO](https://arxiv.org/html/2403.07071):

  | Configuration | BEV / 3D |
  |---|---|
  | No tracker, no self-training | 0.177 / 0.086 |
  | OYSTER-style tracking | 0.255 / 0.104 |
  | LISO tracker | 0.279 / 0.209 |
  | + self-training | 0.360 / 0.290 |
  | + track optimisation | 0.292 / 0.211 |
  | + both | 0.380 / 0.308 |
  | Ground-truth ego-motion and flow | 0.411 / 0.339 to 0.423 / 0.349 |

  - Self-training is single-frame and receives no flow input, "so it learns appearance rather than motion". Weights are reset every 2 regenerations.
  - Final LISO-CP: movable 0.292 / 0.211, still objects 0.208 / 0.140.
  - Stated limitation: no class labels.
- MODEST: the detector trained on seeds recovers parked cars missed by ephemerality. Self-training lifts 50–80 m AP_BEV@0.25 from 8.9 to 47.1. Numbers in Q3 — [MODEST](https://arxiv.org/pdf/2203.15882).
- OYSTER (CVPR 2023) [OYSTER](https://arxiv.org/pdf/2311.02007):
  - Near-range (< 40 m) DBSCAN seeds, with zero-shot extension to long range via CNN translation equivariance.
  - Tracklet-length "temporal consistency" filtering, then "Track, Refine, Retrain, and Repeat".
  - Pandaset class-agnostic AP @0.3 / 0.5 / 0.7: 43.5 / 29.5 / 18.1.
  - Short-tracklet filtering improves results (ablation M3 → M4).
- CPD: self-training with prototype guidance raises WOD L2 AP@0.7 from 24.04 to 32.13 — [CPD](https://arxiv.org/html/2404.16493).
- SeMoLi (10% labels): pseudo-label-trained detector reaches moving-only AP@0.4 of 57.5 with 0% GT and 66.2 with 10% GT, vs 89.0 supervised. All-objects AP without GT is 19.5. The model "biases detections toward nearby moving regions, reducing recall on static objects" — [SeMoLi](https://arxiv.org/html/2402.19463).

**Motion as a class gate and as the static/dynamic refinement switch**
- MS3D++ (domain adaptation with pretrained detectors) [MS3D++](https://arxiv.org/pdf/2308.05988):
  - "Pre-trained detectors commonly misclassify pole-like objects as 'Pedestrians'", with confidences "around 0.3-0.6", like real pedestrians.
  - "While pole-like objects and pedestrians may appear similar when static, the distinction becomes clearer when the object starts moving". It therefore uses only dynamic pedestrian tracks as pseudo-labels.
  - Static pedestrians are added only after self-training calibrates confidence.
- MS3D++ tracks are classified static or dynamic "based on predefined thresholds of begin-to-end distance and box centre variance" [MS3D++](https://arxiv.org/pdf/2308.05988):
  - Static vehicles fuse boxes over H historical frames (KBF).
  - Dynamic vehicles are inter- and extrapolated only for confident tracks.
  - A high IoU threshold for the static tracker is "critical in minimising false positive tracks of parked vehicles that are classified as 'dynamic' due to inaccurate box localization".
- MS3D++ pedestrian pseudo-label quality, Waymo 0–30 m: round 1 P 0.912 / R 0.324; round 4 P 0.846 / R 0.689. Temporal refinement raises vehicle 50–80 m recall from 0.418 to 0.514 — [MS3D++](https://arxiv.org/pdf/2308.05988).
- ViLGOD: static tracks get the median position and majority-vote orientation; dynamic tracks get corner alignment and box refinement. Box refinement adds +6.3 BEV / +7.1 3D AP — [ViLGOD](https://arxiv.org/html/2408.03790).
- Spline or jerk trajectory smoothing for movers:
  - UniLiPs: mAP 23.9 → 31.0 — [UniLiPs](https://arxiv.org/html/2601.05105).
  - LISO: 0.279 → 0.292 BEV AP — [LISO](https://arxiv.org/html/2403.07071).
- UniLiPs also uses moving-object masks to remove dynamic points from the map before semantic and depth labelling — [UniLiPs](https://arxiv.org/html/2601.05105).

**Motion to re-prompt camera segmenters**
- No pipeline was found that prompts SAM or SAM2 with LiDAR movers.
- Nearest work:
  - UniLiPs runs SAM2 on images and removes moving points separately (search summary) — [UniLiPs project](https://light.princeton.edu/unilips).
  - SAM4D (ICCV 2025) is a promptable camera + LiDAR stream segmenter with a "motion-aware" module (search summary only) — [SAM4D](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_SAM4D_Segment_Anything_in_Camera_and_LiDAR_Streams_ICCV_2025_paper.html).
  - An ACM MM 2025 paper fuses SAM-style 2D segmentation with LiDAR at data level, without motion: nuScenes mAP 28.4 — [Ji et al.](https://arxiv.org/abs/2508.20530).

### Inferences
- The strongest training-free lever for WATO is UNION's "movers label appearance clusters" step. Its training-free parts (DINOv2 embedding, K-means, ≥ 5% dynamic fraction) fit perception_2d / semantic_lifting, which already project LiDAR into the cameras. Applied to proposal_generation's `lidar_mos` and `lifted` sources, gated movers would vote on which appearance clusters, and therefore which static masklets, are mobile. That turns ~3.4 gated movers per frame into labels for parked look-alikes.
  - WATO already gets classes from perception_2d, so the marginal value is mostly for objects the detector misses, or as an independent check on detector classes.
- Motion is most useful as a gate where geometry is ambiguous: pole vs pedestrian (MS3D++), and parked vs moving to choose the box-refinement path (fuse a static track's boxes; smooth and register a dynamic track). This argues for keeping `motion_score` and its soft features as track-level attributes that refinement switches on, not just per-frame boxes.
- For self-training, which is in scope as a description, the numbers say movers alone give a precise but recall-poor seed: LISO still-object AP 0.208 vs movable 0.292. Self-training is where static recall comes from.
- The camera-reprompting role is an open opportunity, not a published result. A gated LiDAR mover not covered by any masklet is a natural point or box prompt for SAM2 on the cameras where it projects.

### Gaps
- UNION does not report how many proposals are dynamic or how robust its 5% rule is to false dynamics. Its claim to be "robust against objects falsely labeled as dynamic" is unsupported in the paper.
- No paper quantifies motion re-prompting of 2D segmenters.
- MS3D++ static/dynamic thresholds ("predefined thresholds") are in its code and config, not the text read here.

## Q5. How published methods reject false movers (wall sliding, vegetation, visibility / occlusion drift)

### Takeaway
Published defences are layered:
- Remove the static majority before motion estimation: ground, plus Chamfer < 20 cm/s (Najibi), |v| < 0.2 m/s (SeMoLi), or ray-casting free-space (DUFOMap, used by SeFlow).
- Require registration quality: ICP inlier ratio ≥ 0.2 and residual ≤ 0.2 m (ICP-Flow).
- Require temporal consistency: tracks ≥ 4 steps with median confidence ≥ 0.3 (LISO); tracklet-length filtering (OYSTER); 3-scan neighbour support (UniLiPs).
- Apply shape or common-sense box filters: l/w ≤ 4, l·w ≥ 0.35 m², volume ≥ 0.5 m³ (LISO); 0.5–120 m³ and ≥ 10 points (MODEST); height ≥ 0.5 m and ≤ 1 m off the ground (ViLGOD).
- Check map or persistence consistency: M-detector rejects events near stationary map points; PP-score γ = 0.7 (MODEST).

Pure flow methods have no free-space reasoning. EulerFlow states moving shadows create false flow, the same mechanism as WATO's visibility drift. No paper names "wall sliding"; the closest stated mechanism is the aperture problem on flat surfaces (Najibi).

### Cited Findings
- Najibi removes ground because "Ground is usually captured as a flat surface for which predicting local motion is not possible due to the aperture problem". It removes more static regions by Chamfer distance to earlier frames (< 20 cm/s) because flow on "walls, buildings, trees, etc." adds noise. Separating static points "substantially reduces false positive flow predictions on static objects" — [Najibi et al.](https://arxiv.org/pdf/2210.08061).
- EulerFlow limitation: no ray-casting geometry, so "moving shadows can produce false flow (e.g., on a stationary table)". The bird track was lost when LiDAR returns vanished — [EulerFlow](https://arxiv.org/html/2410.02031).
- DUFOMap principle: "points observed inside this region at another time have to be dynamic" for voxels observed empty. A candidate void voxel is confirmed only if all 26 neighbours are intersected or hit [DUFOMap](https://arxiv.org/html/2403.01449):
  - Settings: voxel 0.1 m, d_s = 0.2 m, d_p = 1.
  - Runtime: 0.062 s per 64-channel scan and 0.019 s per 16-channel scan on an i9-12900KF.
  - Static / dynamic accuracy: KITTI 00 SA 97.96 / DA 98.72; sparse 16-channel set SA 99.64 / DA 83.00.
  - Stated: "When the LiDAR data is sparse, our conservative model for classifying void regions leads to lower dynamic accuracy", and "for a big bus moving slowly, part of the points may not be classified as dynamic".
- SeFlow uses DUFOMap's ray-cast dynamics to decide which points get a dynamic Chamfer loss vs a static zero-flow loss. It reports false flow when "ground points are not completely removed" and failures on "pedestrians near static structures" — [SeFlow](https://arxiv.org/html/2407.01702).
- M-detector (Nature Communications, 2024) [M-detector](https://pmc.ncbi.nlm.nih.gov/articles/PMC10771495/):
  - Three occlusion tests against recent depth images: perpendicular motion, and motion away from or toward the sensor along the beam.
  - "an event point should not lie in the neighborhood of any stationary map points".
  - Points from newly revealed areas "would not occlude or be occluded by any points in previous frames" and are labelled static. This is the explicit guard against visibility-change false positives.
  - KITTI frame-out IoU 0.746 vs LMNet-8* 0.635.
  - 33.6 ms per nuScenes frame on an i7-10700 CPU, and 2–4 µs per point.
- ICP-Flow match acceptance: reject if inlier ratio < 0.2 or mean distance > 0.2 m. The histogram initialisation votes on translation in 0.1 m bins — [ICP-Flow](https://arxiv.org/html/2402.17351).
- LISO box filters drop l/w > 4.0, l·w < 0.35 m², l·w·h < 0.5 m³. Tracks need ≥ 4 steps and median confidence ≥ 0.3 — [LISO](https://arxiv.org/html/2403.07071).
- MODEST common sense: ≥ 10 points, box volume in [0.5, 120] m³, not under the ground or floating. The PP-score filter (α = 20th percentile > γ = 0.7 → drop) is "highly effective in removing false positives" during self-training — [MODEST](https://arxiv.org/pdf/2203.15882).
- MS3D++: a high-IoU static tracker avoids parked cars "classified as 'dynamic' due to inaccurate box localization" — [MS3D++](https://arxiv.org/pdf/2308.05988).
- UP-VL drops boxes whose points are > 99% "background" by cosine similarity to background text embeddings (ε_bg = 0.02). This is a semantic veto that covers vegetation and buildings — [UP-VL](https://arxiv.org/html/2309.14491v1).
- ViLGOD drops segments < 0.5 m tall or > 1 m above ground, and calls a track static if its boxes overlap its largest box or it "lacks consistent motion" — [ViLGOD](https://arxiv.org/html/2408.03790).
- UNION, UniLiPs and ICP-Flow do not discuss vegetation, wall or ego-induced false dynamics — [UNION](https://arxiv.org/html/2405.15688), [UniLiPs](https://arxiv.org/html/2601.05105), [ICP-Flow](https://arxiv.org/html/2402.17351).

### Inferences
- **Visibility drift.** A centroid moves when the visible part changes; a registered shape does not. The published fix is registration (ICP-Flow, Najibi) plus free-space evidence (DUFOMap, M-detector). A cheap test fits WATO's existing AW log-odds grid: a true mover leaves behind space that was observed occupied and later observed free. Require that a gated track's earliest box footprint was later ray-carved free. Static structure revealed as the ego passes never yields such evidence, matching M-detector's "newly revealed points are static" rule.
- **Wall sliding.** A planar or linear fragment can slide along its own plane under ICP or centroid tests: the aperture problem Najibi cites for ground. Three defences:
  - LISO's l/w > 4 and l·w < 0.35 m² box filters directly remove thin wall-like boxes.
  - A degeneracy check, rejecting motion that lies along the cluster's dominant planar or linear axis or whose ICP Hessian is near-singular, is a standard registration safeguard. It is not stated in these papers.
  - A cross-traversal PP-score or free-space veto catches the remainder.
- **Vegetation** is handled in the literature by persistence (PP-score), map consistency (M-detector) or semantics (UP-VL's background text), not by motion thresholds. WATO's camera classes (`cls` None ⇒ no lifted instance) are already a semantic veto. A fragment with no camera class and persistent points across loops is the likeliest false mover.
- **The "far, 22 points, no class" half of WATO's gated tracks.** The literature's answer is to leave them as low-confidence seeds and let self-training or appearance clustering decide (MODEST, UNION), rather than tightening motion gates. Tightening gates is also what kills pedestrians.

### Gaps
- No paper reports a "wall sliding" failure by name, or a measured false-mover rate by cause (vegetation vs walls vs visibility).
- No source evaluates ray-casting free-space vetoes combined with cluster tracking on sparse 16/32-beam multi-LiDAR rigs. DUFOMap's 16-channel test is semi-indoor.
- The M-detector accuracy figures for nuScenes and Waymo are only in figures not retrievable as text.
