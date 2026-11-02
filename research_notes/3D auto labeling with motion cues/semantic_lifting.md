# Semantic lifting: how offline / zero-shot 3D auto-labelers move 2D semantics onto LiDAR and form instances

Scope note. Papers covered, with year and venue where confirmed: ZOPP (arXiv 2411.05311, 2024), UniLiPs (arXiv 2601.05105, Jan 2026; the Princeton project page lists 3DV 2026), VESPA (arXiv 2507.20397, Jul 2025, no venue yet), SAL "Better Call SAL" (arXiv 2403.13129, 2024; the arXiv page names no venue, and ECCV 2024 is my recollection), SAL-4D "Zero-Shot 4D Lidar Panoptic Segmentation" (arXiv 2504.00848, Apr 2025), SAM4D (arXiv 2506.21547, Jun 2025), Najibi et al. UP-VL (ICCV 2023), and from 2024 to 2026 SLF (ECCV 2024), LiSe (arXiv 2407.08569, 2024), UNION (arXiv 2405.15688), LeAP (arXiv 2502.03901, 2025), LOSC (arXiv 2507.07605, 2025; search listings say 3DV 2026), YoCo (arXiv 2502.19698, 2025), Jo et al. "Devil is in the Details" (arXiv 2501.09485, 2025) and CATS-V2V (arXiv 2511.11168, 2025) for timing. Each bullet under "Cited Findings" states what the paper says. Anything under "Inferences" is mine.

Mapping onto wato_world's four known problems, given as a quick index (details below):
- **Few points on far objects / one-sided boxes:** VESPA box inflation to class priors; SLF shape-prior fitting; Najibi and ZOPP track-level ICP accumulation; LiSe 2D-derived boxes beyond 10 m; VESPA tracking.
- **Mask bleed onto background:** DBSCAN largest-component (VESPA with a class-width eps, SAL ensemble-replace, LeAP camera-distance clustering, SAM4D vote-rate main cluster). Mask erosion hurts.
- **Occludee getting occluder's points:** ZOPP parallax filter (depth-ratio 0.25, kernel 15); SLF depth-sorted occlusion maps; VESPA mask NMS/trim; SAM4D multi-frame vote rate.
- **v·Δt shift:** CATS-V2V per-object target-time compensation; Jo et al. per-moving-cluster ICP registration to camera time. None of the zero-shot labeling papers handles it explicitly.

## Q1. How each method lifts masks to points: visibility tests, depth sources, erosion and boundary handling

### Takeaway
Most published zero-shot labelers do plain frustum projection with no z-buffer or occlusion test (VESPA, SAL, SAL-4D, LeAP, UP-VL, LiSe, YoCo). They rely on 3D clustering after lifting to cut bleed and occlusion leakage. Only UniLiPs (LiDAR min-pooled depth plus 0.5 m), ZOPP (parallax depth-ratio filter), SAM4D (ray casting into a 4D voxel reconstruction) and SLF (depth-sorted mask occlusion maps) reason about visibility explicitly. wato_world's Eq. 1 test plus Depth Anything plus a 2 m largest-group filter is already at or above the state of the art. Erosion of SAM masks measurably hurts in two independent papers.

### Cited Findings

**UniLiPs (2026)**
- The visibility test is Eq. 1: a point is visible if D(u,v) ≤ min(D(𝒩(u,v))) + 0.5. Depth comes from projecting the LiDAR scan into the label map with P_cam. The neighborhood (min-pool kernel) size is not stated in the paper — [UniLiPs](https://arxiv.org/html/2601.05105)
- Per-pixel labels come from OneFormer ("stacking the most recurrent (Argmax) class from each OneFormer model per-pixel logit maps"). SAM2 supplies object masks on down-sampled frames — [UniLiPs](https://arxiv.org/html/2601.05105)
- Table 8b ablates the lifting module "sequentially" (Algorithm 1, then accumulation, then occlusion mask). Full method versus each step:
  - KITTI mIoU: 64.9 / 60.7 / 59.4 / 57.0
  - nuScenes mIoU: 58.0 / 55.2 / 54.9 / 50.1
  - nuScenes category-mIoU: 65.2 / 64.9 / 62.6 / 55.3
  - The occlusion mask is therefore worth about +2.4 mIoU (KITTI) and +4.8 mIoU (nuScenes) if the removals are cumulative. Source: [UniLiPs](https://arxiv.org/html/2601.05105)

**ZOPP (2024)**
- Lifting is plain projection (q_i = K(p_i^C)). Points inside a mask inherit its instance ID and class. Points from neighbouring views merge by instance ID. Re-projected object points are then cleaned with 3D DBSCAN; eps and minPts are not reported — [ZOPP](https://arxiv.org/html/2411.05311)
- A separate parallax occlusion filter follows (see Q2) — [ZOPP](https://arxiv.org/html/2411.05311)

**VESPA (2025)**
- Detection and masks:
  - Grounding DINO with confidence ≥ 0.3 feeds SAM masks.
  - Mask NMS on intersection-over-area: a mask with IoA > 0.5 against a higher-confidence mask is suppressed, otherwise it is *trimmed*. Source: [VESPA](https://arxiv.org/html/2507.20397)
- Point filtering before projection:
  - Ground is removed by RANSAC planes (global and per angular sector) fitted on 5 aligned sweeps. Points within 30 cm of the plane are dropped (the source renders it "3030 cm").
  - Points beyond 40 m range or above 4 m height are also dropped.
  - "Each 3D point is associated with a segmentation mask if the projection falls inside the mask." Source: [VESPA](https://arxiv.org/html/2507.20397)
- No visibility test. VESPA states the problem directly: "the LiDAR sensor is mounted significantly higher than the cameras," so "the LiDAR returns include 3D points from surfaces that are occluded or invisible from the viewpoint of the camera" whose "2D positions may still fall within the foreground object mask." The fix is cluster denoising (DN), not a depth test — [VESPA](https://arxiv.org/html/2507.20397)
- DN runs DBSCAN in the XY plane with **eps = the class's average object width, queried from an LLM**, keeps the largest connected component, and refits an L-shape box. Table 4 (3-class nuScenes, CenterPoint trained on pseudo-labels):

  | Configuration | mAP | NDS | Car AP |
  |---|---|---|---|
  | Full | 46.76 | 43.47 | 48.9 |
  | Without DN | 27.86 | 26.72 | 23.8 |

  The paper's text says DN is worth "up to 19%", which does not match the table. Source: [VESPA](https://arxiv.org/html/2507.20397)

**SAL (2024)**
- SAM settings: POINTS_PER_SIDE 32, PRED_IOU_THRESH 0.84, STABILITY_SCORE_THRESH 0.86, MIN_MASK_REGION_AREA 100. The mask hierarchy is flattened by area-sorted NMS (IoU 0.01), leaving about 45 masks per scan; about 39 of them transfer to LiDAR — [SAL](https://arxiv.org/html/2403.13129)
- Visibility is a frustum filter only. Pseudo-label PQ is 22.2 without it and 59.3 with it, against 69.0 for GT labels (Table 1) — [SAL](https://arxiv.org/html/2403.13129)
- On boundaries, SAL reports "bleeding edges ... after unprojecting from the cameras to Lidar caused by imperfect calibrations". It calls the transfer "inherently noisy due to imperfect calibration and issues with synchronization and Lidar rolling shutter" — [SAL](https://arxiv.org/html/2403.13129)
- **Erosion hurts.** Table 4 (class-agnostic PQ, semantic oracle):

  | Variant | PQ |
  |---|---|
  | SAM | 46.0 |
  | SAM + Erosion | 42.2 |
  | SAM + DBSCAN (filter) | 46.7 |
  | SAM + DBSCAN (replace) | 48.7 |
  | SAL model | 70.7 |

  The authors attribute the erosion drop to smaller segments failing the 0.5 overlap threshold. The kernel size is not given — [SAL](https://arxiv.org/html/2403.13129)
- DBSCAN refinement:
  - Ground is removed with Patchwork++.
  - DBSCAN runs at six eps values (1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221), giving about 5,413 clusters per scan.
  - A lifted mask is **replaced** by its best cluster if IoU > 0.5. Source: [SAL](https://arxiv.org/html/2403.13129)

**SAL-4D (2025)**
- Table 9 (single-scan label steps, PQ class-agnostic / zero-shot):

  | Step | Class-agnostic PQ | Zero-shot PQ |
  |---|---|---|
  | Original | 48.7 | 27.5 |
  | + flatten in 3D (lift then flatten) | 51.8 | 28.6 |
  | + per-instance DBSCAN refine | 53.6 | 29.7 |
  | + flatten via coverage (IoMin, not IoU) | 55.3 | 29.9 |

  Source: [SAL-4D](https://arxiv.org/html/2504.00848)
- "The best pseudo-label is obtained by only enabling DBSCAN per scan separately." DBSCAN across all scans of a window hurts association, which the authors attribute to dynamic objects (Table 8) — [SAL-4D](https://arxiv.org/html/2504.00848)
- No explicit occlusion reasoning — [SAL-4D](https://arxiv.org/html/2504.00848)

**SAM4D data engine (2025)**
- Builds a VDBFusion voxel 4D reconstruction: the static background in world coordinates, and each foreground object in its own body frame from *pre-annotated 3D boxes* — [SAM4D](https://arxiv.org/html/2506.21547)
- Uses "exhaustive ray casting from the center of each image toward the voxels" to make a pixel-voxel table. Rays go only to frustum voxels, which is its visibility constraint — [SAM4D](https://arxiv.org/html/2506.21547)
- Each voxel's "vote rate" is the fraction of image observations mapping it to the object. DBSCAN runs in BEV and keeps "the main cluster with the highest vote rate while discarding the rest as noise" — [SAM4D](https://arxiv.org/html/2506.21547)
- Cross-modal IoU quality: mean 0.56, 10th percentile 0.24 — [SAM4D](https://arxiv.org/html/2506.21547)

**LeAP (2025)**
- PointPainting-style projection with no z-buffer. A mask's points are clustered **by distance from the camera** and points outside the largest cluster are discarded, because "calibration and masking errors often give background points the label of a foreground object" — [LeAP](https://arxiv.org/html/2502.03901v1)

**UP-VL / Najibi et al. (ICCV 2023)**
- Per-pixel OpenSeg features are unprojected to points by calibration, with no separate occlusion test. The 640-d features are PCA-reduced to 64 — [Najibi et al.](https://arxiv.org/html/2309.14491v1)

**LOSC (2025)**
- Points outside the frustum, or "hidden behind other points due to parallax configurations", get *ignore*. No depth-test details are given — [LOSC](https://arxiv.org/html/2507.07605v2)

**SLF (ECCV 2024)**
- Table 10 (KITTI AP3D Easy / Mod. / Hard), corrupting SAM masks against 87.9 / 88.2 / 83.9 clean:

  | Corruption | AP3D Easy / Mod. / Hard |
  |---|---|
  | Erosion 5×5 | 85.2 / 83.9 / 79.5 |
  | Erosion 9×9 | 82.1 / 78.2 / 72.1 |
  | Dilation 5×5 | 86.3 / 85.6 / 81.2 |
  | Dilation 9×9 | 83.9 / 80.5 / 76.6 |

  Source: [SLF](https://arxiv.org/html/2407.11382v2)

**YoCo (2025)**
- Uses Depth Anything (DAM) for depth smoothing of click prompts. Adding DAM raised mAP 40.19 → 42.13 (Table 3; column headers ambiguous in the source) — [YoCo](https://arxiv.org/html/2502.19698v1)

### Inferences
- wato_world's lifting (Eq. 1 with both a Depth Anything V2 map and a LiDAR z-buffer, then a largest-group filter) is more conservative than VESPA, SAL, SAL-4D, LeAP or UP-VL, which have no depth test. Even so, UniLiPs's occlusion-mask ablation (+2.4 to +4.8 mIoU) supports keeping it.
- The direct analogue of wato's 2 m largest-group filter is VESPA's DN: XY DBSCAN, keep the largest component. Its one refinement is that **eps is class-dependent (≈ class average width)** rather than a fixed 2 m. That would cut bleed for pedestrians and bikes (eps well under 2 m) and avoid splitting trucks.
- SAL's "replace with best DBSCAN cluster if IoU > 0.5" is a cheap upgrade over "filter": +2.0 PQ over filter-only in SAL Table 4. It also completes object points the mask missed.
- SAL-4D's warning is directly relevant to multi-sweep work: run the bleed or cluster filter **per scan**, not on the accumulated cloud, or moving objects smear.
- Do not erode SAM2 masks to fight bleed. Two papers (SAL −3.8 PQ; SLF −2 to −12 AP3D) show erosion costs more than it saves, because far or thin objects lose most of their few points. Bleed is better handled in 3D.

### Gaps
- UniLiPs does not state its min-pool kernel size. ZOPP does not give its DBSCAN eps or minPts. SAL does not give its erosion kernel. I found no official config files for ZOPP, VESPA (code "upon acceptance") or UniLiPs.
- No paper reports a precision/recall trade-off for the visibility tolerance (0.5 m) by range.

## Q2. How occluded objects whose mask receives the occluder's points are handled

### Takeaway
ZOPP is the only labeler with a dedicated parallax filter. It runs a sliding window over each mask's projected points, splits near and far points when the depth ratio exceeds 0.25, and strips far points inside the near points' local rectangle. It gains about +2.6 IoU for vehicles. That filter targets far background leaking into a near mask, which is the inverse of wato_world's "occludee gets occluder edge points" case. For wato's case the transferable ideas are depth-ordered mask ownership (ZOPP's near-to-far processing, SLF's occlusion maps), mask trimming against higher-confidence masks (VESPA), and multi-observation vote rates (SAM4D).

### Cited Findings

**ZOPP parallax occlusion filter (stated in the paper)**
1. Sort object masks by depth, near to far, and process each one.
2. Slide a kernel of size **15**, with horizontal step **10** and vertical step **5** (units presumably pixels), over the projected points.
3. Compute (max(p) − min(p)) / min(p) of the depths in the kernel. If it exceeds **0.25** ("The depth ratio threshold is set to 0.25"), split the points into near and far.
4. Build a local rectangle from the near points. Its left and right edges are the min and max near-point u. With a single near point, the right edge is a pseudo coordinate. Its top is the minimum near-point v and its bottom is the kernel's bottom.
5. Remove far points inside the rectangle from the object and relabel them as background (road, sidewalk).

The paper's Algorithm 1 appears to have a typesetting error (the same call in both branches). Source: [ZOPP](https://arxiv.org/html/2411.05311)

**ZOPP filter results**
- Table 8, segmentation IoU before → after the filter:

  | Class | Before | After |
  |---|---|---|
  | Vehicle | 51.6 | 54.2 |
  | Bicyclist | 47.7 | 49.6 |
  | Pedestrian | 76.1 | 77.3 |
  | Pole | 49.6 | 51.7 |
  | Building | 73.8 | 75.5 |
  | Road | 80.5 | 81.8 |

  Source: [ZOPP](https://arxiv.org/html/2411.05311)
- Table 9, filtering plus point completion together, L2 recall:

  | Class | Before | After |
  |---|---|---|
  | Vehicle | 22.6 | 48.8 |
  | Pedestrian | 34.2 | 46.7 |
  | Cyclist | 11.5 | 22.9 |

  Filtering and completion are not separated — [ZOPP](https://arxiv.org/html/2411.05311)
- Table 7, Vehicle L1 AP: ZOPP 37.56 overall versus 33.42 on occluded objects (an 11.02% drop). CenterPoint drops from 73.04 to 59.30 — [ZOPP](https://arxiv.org/html/2411.05311)

**Other occlusion handling**
- **SLF:** LiDAR points are projected onto each mask, and **each mask's depth is the median depth of its associated points**. Masks are sorted by depth to build occlusion maps, which weight the Dice silhouette loss so occluded parts of a mask don't penalize the fit — [SLF](https://arxiv.org/html/2407.11382v2)
- **VESPA:** mask NMS on IoA. A lower-confidence mask with IoA > 0.5 against a higher one is suppressed; otherwise the overlap is trimmed from it. Occlusion leakage is then handled in 3D by DN (largest XY component with eps = class width) — [VESPA](https://arxiv.org/html/2507.20397)
- **SAM4D:** voxel vote rate over all image observations, then the main BEV cluster by vote rate — [SAM4D](https://arxiv.org/html/2506.21547)
- **SAL-4D:** "flatten" overlapping lifted masklets by **intersection-over-minimum > 0.5 against a larger (by point volume) masklet**, which suppresses the smaller one. This alone was worth +1.7 class-agnostic PQ over IoU-based flattening (Table 9) — [SAL-4D](https://arxiv.org/html/2504.00848)

### Inferences
- **Why ZOPP's filter alone would not fix wato's failure, and could make it worse.** In wato's case the occludee B's mask contains the occluder A's edge points. They pass Eq. 1 because they *are* the nearest surface. Inside B's mask, ZOPP's kernel would label A's edge points as "near" and B's real points as "far", and would delete B's points. This is my reading of ZOPP's rule, not something ZOPP tests.

  ZOPP's near-to-far ordering suggests the intended fix: points already owned by a nearer mask are removed before a farther mask is processed. The paper does not state that explicitly.

- **A wato-specific rule assembled from these pieces (my synthesis, not a published method).**
  1. Process instances near to far by **median point depth** (SLF's mask depth).
  2. Give a point to the nearest mask whose *dilated* footprint contains it. Dilation absorbs boundary error and the v·Δt shift.
  3. Within each instance, apply a ZOPP-style depth-ratio split (0.25 at kernel 15 px is the published value). Drop the *near* cluster when it lies within a few pixels of a nearer instance's mask, and drop the *far* cluster otherwise.
- VESPA-style IoA trimming of overlapping SAM2 masks (owner = higher-confidence or nearer mask) is a cheap 2D-side complement, since SAM2 masklets of an occludee often bleed under the occluder's edge.
- **Multi-frame vote rate (SAM4D) is the strongest generic defence.** A's edge points fall inside B's mask only in some frames, as the parallax geometry changes, so their per-point fraction of "B" votes stays low. wato_world's planned multi-sweep aggregation would get this for free if votes are kept as counts (UniLiPs's (l_ij, n_ij)) rather than a single argmax.

### Gaps
- No paper in scope reports a metric isolating "occludee box pulled onto the occluder". ZOPP's occluded-subset AP is the closest.
- ZOPP's handling of points already claimed by nearer masks, and the units of its kernel, are not specified.

## Q3. Temporal aggregation (multi-sweep accumulation, map-level propagation, 4D tracklets) and whether it helps sparse or far objects

### Takeaway
Temporal aggregation consistently helps, but published gains are mostly in semantic mIoU or association, not range-binned detection. The two kinds of evidence:
- **Point-level gains:** UniLiPs (Algorithm 1 KD-tree propagation +2.8 to +4.2 mIoU, accumulation +0.3 to +1.3), LOSC (10 cm voxel majority vote, +5.9 to +9.5 mIoU after training), LeAP (voxel Bayesian, +2.1 mIoU and 6× more labeled points) and YoCo (2 adjacent frames best, +4.8 mAP; 5 frames worse).
- **Object-level gains, which matter most for sparse or far objects:** track-level ICP accumulation for amodal boxes (UP-VL, ZOPP) and VESPA's tracking (removing it halves mAP).

Every map-level method either assumes static semantics or must exclude moving objects.

### Cited Findings

**UniLiPs (2026)**
- Algorithm 1 builds a KD-tree over map points. It uses a radius neighbourhood r = 0.2 m, Gaussian weights w_ij = exp(−‖p_i − p_j‖²/(2σ²)) with σ = r/2, excludes labels 0 and −1, and sets l_i = argmax_l Σ w_ij over neighbours with label l — [UniLiPs](https://arxiv.org/html/2601.05105)
- Each map point keeps (l_ij, n_ij) label counts, accumulated by "sequentially associating each point label of a LiDAR scan to the correspondent point in the map" — [UniLiPs](https://arxiv.org/html/2601.05105)
- Table 8b, ablated sequentially. Removing Algorithm 1 costs 4.2 mIoU (KITTI, 64.9 → 60.7) and 2.8 (nuScenes, 58.0 → 55.2). Further removing accumulation costs 1.3 / 0.3 more — [UniLiPs](https://arxiv.org/html/2601.05105)
- IWU is a static-probability update weighted by label consistency C = max_j n_ij / Σ_j n_ij, with α = 0.7, τ_s = 0.5 and a 30 cm radius. Table 8a, box quality:

  | Configuration | mAP | NDS |
  |---|---|---|
  | Full | 31.0 | 45.2 |
  | Without spline optimization | 23.9 | 33.2 |
  | Without IWU | 11.7 | 30.8 |

  Source: [UniLiPs](https://arxiv.org/html/2601.05105)
- The paper says aggregation helps "especially in dense regions". It gives no far-range ablation — [UniLiPs](https://arxiv.org/html/2601.05105)

**LOSC (2025)**
- Temporal consolidation aggregates the whole sequence and takes a per-voxel majority vote at **10 cm**, with *ignore* as a full class. Softmax-weighted votes gave "marginal" gain. The method assumes static semantics; errors arise when objects of different classes pass through the same voxels — [LOSC](https://arxiv.org/html/2507.07605v2)
- Table 2, trained-model mIoU (nuScenes / SemanticKITTI):

  | Labels | mIoU |
  |---|---|
  | VLM labels | 40.5 / 22.6 |
  | + temporal consolidation | 46.4 / 32.1 |
  | + augmentation and temporal, combined | 48.0 / 34.2 |

  Source: [LOSC](https://arxiv.org/html/2507.07605v2)
- Table 4, label coverage / label mIoU: VLM 75.8% / 39.3 → temporal 74.1% / 39.9 on nuScenes; 15.4% / 25.5 → 6.7% / 26.3 on SemanticKITTI. Raw label accuracy rises only slightly, and coverage falls because *ignore* wins votes — [LOSC](https://arxiv.org/html/2507.07605v2)

**LeAP (2025)**
- All scans go into one sparse voxel grid with a Bayesian update (multiply by class probabilities and renormalize). The paper says "0.2 mm", presumably 0.2 m. A k = 9 nearest-voxel distance-weighted smoothing follows — [LeAP](https://arxiv.org/html/2502.03901v1)
- Table I (SemanticKITTI mIoU / category mIoU):

  | Labels | mIoU | Category mIoU |
  |---|---|---|
  | Point | 46.8 | 68.6 |
  | Voxel | 48.9 | 71.2 |
  | + 3D-CN, 1 iteration | 57.6 | 81.0 |

  Voxels label "over six times the number of points". The authors note that **moving objects leave "tracks" in the voxel grid**, which they list as a limitation — [LeAP](https://arxiv.org/html/2502.03901v1)

**YoCo (2025)**
- Adjacent frames are pose-transformed into the current frame and voxel soft-voted. Thresholds scale by dist/D, so voxels near the sensor need more votes and higher confidence — [YoCo](https://arxiv.org/html/2502.19698v1)
- Table 4, mAP by number of adjacent frames:

  | Adjacent frames | mAP |
  |---|---|
  | 0 | 47.37 |
  | 1 | 47.52 |
  | 2 | 52.18 |
  | 3 | 51.07 |
  | 5 | 48.17 |

  More frames cause "erroneous votes" — [YoCo](https://arxiv.org/html/2502.19698v1)
- Table 5, voting rule: soft + count + score 52.18 versus hard 48.54 — [YoCo](https://arxiv.org/html/2502.19698v1)

**SAL-4D (2025)**
- SAM2 masklets run over windows. Objects are labeled only if visible in the window's first frame, and later ones are picked up through cross-window association — [SAL-4D](https://arxiv.org/html/2504.00848)
- Table 1 (SemanticKITTI LSTQ):
  - Window 8 without cross-window association: 49.2
  - With it: windows 2 / 4 / 8 / 16 give 50.6 / 51.4 / 51.1 / 50.5
  - Without cross-window association, LSTQ rises with window length: 46.3 (K = 2) → 49.9 (K = 16). Association saturates at 8 frames, while recognition keeps improving.
  - On nuScenes (2 Hz keyframes), window 4 drops LSTQ from 30.0 to 27.6.

  Source: [SAL-4D](https://arxiv.org/html/2504.00848)
- Single-scan zero-shot PQ: SAL 29.9 versus SAL-4D 34.5; mIoU 31.9 versus 39.1. SAL "fails to segment smaller objects" such as cyclists and motorcyclists "due to limited temporal context" (Table 14 caption) — [SAL-4D](https://arxiv.org/html/2504.00848)

**UP-VL / Najibi et al. (ICCV 2023)**
- Per track, ICP shape registration aligns the points across frames. Registered points are aggregated, an amodal box is fit, registration is undone, and each visible box is replaced by the amodal box — [Najibi et al.](https://arxiv.org/html/2309.14491v1)
- Clustering + OpenSeg visible boxes 34.7 mAP versus auto-labels with amodal boxes 43.7 (moving objects, Table 2); 13.1 versus 22.4 for all motion states (Table 3). These comparisons confound other changes — [Najibi et al.](https://arxiv.org/html/2309.14491v1)

**ZOPP (2024)**
- Static objects: points accumulated in global frame by pose, L-shape fit, then outlier removal, FPS subsampling, a second L-shape fit, and transform back to each frame — [ZOPP](https://arxiv.org/html/2411.05311)
- Dynamic objects: per-frame FPS and initial boxes, anchors refined by L-shape fitting, and trajectories smoothed by a linear fit and a Kalman filter in global coordinates — [ZOPP](https://arxiv.org/html/2411.05311)
- Shape completion: a PointNet plus CLIP-text-conditioned completion network, trained on point removal, rotation and transform augmentations with Chamfer loss — [ZOPP](https://arxiv.org/html/2411.05311)
- Vehicle L1 AP by range: 42.31 (0–30 m), 35.14 (30–50 m), 29.89 (50+ m) — [ZOPP](https://arxiv.org/html/2411.05311)

**VESPA (2025)**
- Tracking matches DINOv2 features, uses ICP for velocity, and orients boxes along motion above 0.5 m/s. Table 4: removing TR drops mAP 46.76 → 25.48 and NDS 43.47 → 21.84, the largest single effect. The text says "21%", which does not match the table. The paper does not describe explicit cross-frame box aggregation — [VESPA](https://arxiv.org/html/2507.20397)
- Box inflation (BI): dimensions below LLM class priors are inflated to class means, with the box shifted outward, away from the ego. Without BI, mAP drops 46.76 → 40.68, and **car AP drops 48.9 → 30.9** with ASE 0.363 → 0.766 — [VESPA](https://arxiv.org/html/2507.20397)

**SLF (ECCV 2024, single frame, cars only)**
- Fits a 5-D PCA SDF shape prior (79 Apollo-Car3D models) by Adam (lr 0.1, 150 iterations). The losses are a Dice silhouette loss against the SAM mask, an SDF loss at frustum points, and a ground term — [SLF](https://arxiv.org/html/2407.11382v2)
- Table 4, AP3D Easy / Mod. / Hard:

  | Losses | AP3D |
  |---|---|
  | E_pc only (points only) | 70.5 / 64.8 / 62.9 |
  | E_mask + E_pc | 87.3 / 88.0 / 83.7 |
  | Mask only | 6.9 / 5.9 / 5.6 |

  Source: [SLF](https://arxiv.org/html/2407.11382v2)
- Table 8, beams downsampled: 64 → 87.9 / 88.2 / 83.9; 32 → 81.3 / 77.5 / 77.3; 16 → 80.7 / 76.3 / 70.0. Sparse points cause front/rear flips — [SLF](https://arxiv.org/html/2407.11382v2)

**LiSe (2024)**
- Adds 2D-derived boxes (GroundingDINO → SAM → frustum points → seeded region growing, keeping the largest cluster → tight box) only for objects **> 10 m** from the ego, to avoid conflicts with LiDAR boxes at close range — [LiSe](https://arxiv.org/html/2407.08569)
- Table 3, AP_BEV for "3D only" → "3D + 2D":

  | Range | 3D only | 3D + 2D |
  |---|---|---|
  | 30–50 m | 5.5 | 8.3 |
  | 50–80 m | 1.5 | 3.4 |

  Source: [LiSe](https://arxiv.org/html/2407.08569)
- Table 4, minimum distance for image-derived boxes: all 13.8; > 5 m 15.4; > 10 m 16.2; > 15 m 14.3 (0–80 m AP_BEV) — [LiSe](https://arxiv.org/html/2407.08569)

### Inferences
- **For wato's far-object problem (4–15 points, box on one side), the evidence favours object-level fixes over point-level label propagation.**
  - VESPA BI (+18 car AP), SLF's mask-silhouette constraint (points-only 64.8 → mask + points 88.0 Mod. AP3D) and track-level ICP aggregation (UP-VL, ZOPP static) all add geometry the single sweep lacks.
  - UniLiPs Algorithm 1 and LOSC voting improve per-point labels but don't create new points on the far object.
- SLF's 16-beam result (76.3 Mod. AP3D) is the only number I found close to wato's VLP-16 or VLP-32C density, and it still beats points-only fitting at 64 beams (64.8). That makes SLF the most directly applicable "box from few points + mask" method, and it is already planned as a proposal source. Its prior covers cars only.
- A cheap version of the SLF idea is a class-prior box (VESPA BI). Inflate to class-mean dimensions and **shift away from the ego**, using the observed face as the near face. This directly addresses "a box covers one side".
- Map-level label propagation (UniLiPs Algorithm 1, LOSC, LeAP) is valid for **static** points only. LeAP explicitly reports moving-object "tracks". If adopted, gate it with lidar_preprocessing's dynamic mask (`~mask` trusted static), as UniLiPs does with IWU.
- YoCo's result (2 adjacent frames best, 5 worse) and SAL-4D's per-scan DBSCAN finding argue for **short windows or track-aligned aggregation for objects**, not long raw accumulation.
- Use count-based, distance-aware votes, as in YoCo's "soft + count + score" and UniLiPs's n_ij: keep counts per (point, instance), not a single argmax.

### Gaps
- No paper in scope gives a range-binned ablation showing that *label* aggregation (as opposed to box aggregation or 2D-derived boxes) improves far objects specifically.
- VESPA's text/table mismatch on ablation percentages is unresolved. I used the table values.

## Q4. Merging an object's masks across cameras and across time into one 3D instance

### Takeaway
Cross-camera merging is done with 3D point overlap, at a very loose threshold (SAL: IoU > 0.01) or a strict one (SAL-4D: IoU ≥ 0.5). VESPA adds an image-border rule (leftmost or rightmost 15%) and ZOPP adds appearance plus panoramic horizontal position. Cross-time merging uses either SAM2 within a window plus Hungarian assignment on 3D IoU between windows (SAL-4D), or DINOv2 mask-averaged embeddings with a class gate, a velocity gate and mutual-best matching (VESPA). Cross-camera merging is a small effect (VESPA −1.3 mAP without it). Cross-time association is a large one.

### Cited Findings
- **VESPA MCM:** proposals merge if their classes match and they either share LiDAR points or sit within the leftmost or rightmost 15% of adjacent images. Merged point clouds, features and confidences are combined. Without MCM: mAP 46.76 → 45.50, car AP 48.9 → 45.6 — [VESPA](https://arxiv.org/html/2507.20397)
- **VESPA tracking:** DINOv2 features averaged over the mask area are matched by cosine similarity among same-class proposals. A spatial gate uses an LLM-queried maximum urban velocity, and only mutual best matches are accepted. Velocity comes from ICP of matched clusters — [VESPA](https://arxiv.org/html/2507.20397)
- **SAL:** masks with multi-view IoU above 0.01 are unioned and their CLIP features averaged — [SAL](https://arxiv.org/html/2403.13129)
- **SAL-4D:** instances from different cameras fuse at IoU ≥ 0.5, with a mask-area-weighted feature average. Cross-window association is Hungarian assignment on 1 − 3D IoU over the overlapping frames; matched IDs keep a global ID and CLIP features are averaged over time. Cross-window labels (Table 12) raise S_assoc from 67.2 to 77.2 (LSTQ 50.7 → 53.2) but lower S_cls from 38.3 to 36.6 — [SAL-4D](https://arxiv.org/html/2504.00848)
- **ZOPP:** per view, Grounding-DINO → SAM → DeAOT tracking. Across views, the cost combines appearance (cosine of Grounding-DINO/DeAOT features) and location (views concatenated in panoramic order, horizontal pixel distance normalized per object). No weights are given and there is no ablation of this step — [ZOPP](https://arxiv.org/html/2411.05311)
- **SAM4D:** masklets from different videos with "substantial overlap" in the voxel reconstruction are merged — [SAM4D](https://arxiv.org/html/2506.21547)
- **UNION:** appearance embedding = DINOv2 features at the projected pixels of a cluster's LiDAR points, averaged. DINOv2 ViT-L/14 beat I-JEPA by 12.7 AP / 5.8 NDS when used for clustering (Table 4) — [UNION](https://arxiv.org/html/2405.15688)

### Inferences
- wato_world's "merge masklets that claim the same LiDAR points" is the VESPA/SAL criterion. Two cheap additions are published:
  - VESPA's camera-seam rule (same class plus both masks touching the 15% border of adjacent cameras), for objects split at a seam where few points are shared.
  - SAL-4D's IoMin-based flattening, so a small duplicate is suppressed by a larger one.
- For cross-time and cross-chunk identity (tracking's job), the published zero-shot recipe is DINOv2 mask-averaged embeddings plus a class gate, a velocity gate and mutual-best matching (VESPA), or Hungarian assignment on 3D IoU over overlapping frames (SAL-4D). The 6 s chunk overlap maps naturally onto SAL-4D's "association over overlapping frames".

### Gaps
- No paper ablates the cross-camera merge threshold (0.01 versus 0.5 IoU) or appearance-based cross-camera merging in isolation.

## Q5. Unsynchronized camera/LiDAR timing for moving objects

### Takeaway
None of the zero-shot labeling papers (ZOPP, VESPA, SAL, SAL-4D, SAM4D, UP-VL, UniLiPs) compensates object motion between LiDAR and camera time. They treat the error as noise and absorb it with 3D clustering or denoising. The two concrete remedies found are both per-object: move each moving object's points to the camera timestamp, either by object-level motion compensation (CATS-V2V: recall at IoU 0.5 from 0.39 to 0.59 on one clip) or by per-moving-cluster ICP registration to the camera keyframe (Jo et al.: +3.7 linear probing).

### Cited Findings
- **Timing in the labeling papers themselves:**
  - SAL calls transfer noise "imperfect calibration and issues with synchronization and Lidar rolling shutter" and relies on DBSCAN and model distillation — [SAL](https://arxiv.org/html/2403.13129)
  - VESPA's ablation says DN helps mitigate "calibration and synchronization errors" but gives no mechanism — [VESPA](https://arxiv.org/html/2507.20397)
  - ZOPP, SAL-4D, SAM4D and UP-VL do not discuss it — [ZOPP](https://arxiv.org/html/2411.05311); [SAL-4D](https://arxiv.org/html/2504.00848); [SAM4D](https://arxiv.org/html/2506.21547); [Najibi et al.](https://arxiv.org/html/2309.14491v1)
- **CATS-V2V (2025)** distinguishes two compensations:
  - **Frame-based:** points in each camera's FOV are paired with the nearest camera frame and "compensated to the corresponding camera frame timestamp".
  - **Target-based:** "we compute the average timestamp of all points belonging to each object and associate the object with the nearest camera frame". The object's points are then motion-compensated to that timestamp.
  - Table 4 (one clip, five cameras):

    | Metric | Stamp-based | Frame-based | Target-based |
    |---|---|---|---|
    | Average IoU | 0.3736 | 0.4493 | 0.4623 |
    | Recall at IoU 0.5 | 0.3906 | 0.5766 | 0.5947 |
    | Recall at IoU 0.7 | 0.1172 | 0.2494 | 0.2768 |
    | Centre offset (px) | 61.54 | 50.26 | 49.76 |

  Source: [CATS-V2V](https://arxiv.org/html/2511.11168)
- **Jo et al. (2025), Positive Pair Mining:**
  - Static points align in unsynced data, but "the moving points are mismatched because obtaining projection from moving points to the 2D image is intractable".
  - Remedy: aggregate ±5 sweeps by pose, remove ground with Patchwork++, cluster with HDBSCAN, call a cluster moving if its centre moves by L1 > **0.5 m** between consecutive times, and run **cluster-wise ICP only on moving clusters** to the keyframe (identity for static and ground points).
  - Table S2, nuScenes linear probing: synced-only 41.6, nearest alignment 41.5, PPM 45.2.
  - On dynamic foreground, PPM's flow error is still high: EPE 0.992 and AccS 13.3, versus static 0.102 / 83.0 (Table S9).

  Source: [Jo et al.](https://arxiv.org/html/2501.09485)
- **Other related work (search summaries, not full-text verified):**
  - SST-Calib modifies the projection equation for the sensor delay and estimates velocity from consecutive images — [SST-Calib](https://arxiv.org/pdf/2207.03704)
  - UrbanIng-V2X estimates a worst-case in-frame error of about 0.7 m at 50 km/h with worst-case misalignment — [UrbanIng-V2X](https://arxiv.org/pdf/2510.23478)
  - AsyncBEV learns the residual motion of dynamic objects after ego-motion compensation — [AsyncBEV](https://arxiv.org/pdf/2601.12994)

### Inferences
- wato's measured shift (median 0.1 m, p99 1.1 m) is the same kind of error CATS-V2V corrects. wato already compensates ego motion and per-point time (deskew plus pose at the image's own time), which is CATS-V2V's "frame-based" level. The missing step is "target-based".
- A practical wato version is two-pass:
  1. Lift with the static assumption.
  2. Form instances or tracks and estimate per-instance velocity from lidar_preprocessing's Step F Kalman tracks, or from consecutive-sweep centroid or ICP as in Jo et al. and VESPA.
  3. Shift each moving instance's points by v·(t_cam − t_point) and re-run the mask vote for those points only.

  The 0.5 m-between-sweeps "moving" gate (Jo et al.) and VESPA's 0.5 m/s motion threshold are published starting values.
- Until that exists, absorbing the error the way VESPA and SAL do (a 3D largest-component filter with class-width eps) is the published fallback. It explains why DN is VESPA's second-largest ablation effect.

### Gaps
- No zero-shot 3D labeling paper reports labeling accuracy as a function of camera-LiDAR Δt or object speed.
- CATS-V2V's numbers come from one clip and use annotated boxes. They do not come from an auto-labeler.

## Q6. Geometry feeding back into semantics (LiDAR motion or clusters used to clean or re-prompt masks)

### Takeaway
Feedback exists, but almost entirely in 3D. Cluster replacement of masks (SAL, SAL-4D), main-cluster selection by multi-view vote rate (SAM4D), a cheaply trained 3D consistency network on the top 20% most confident labels (LeAP, +8.7 mIoU), label consistency weighting the static/dynamic map update (UniLiPs IWU, which takes box mAP from 11.7 to 31.0) and motion statistics deciding which appearance clusters are object classes (UNION, +25.3 AP). I found no zero-shot driving labeler that re-prompts SAM or SAM2 from LiDAR motion clusters.

### Cited Findings
- **UNION:** proposals moving at least 0.50 m (ICP-Flow, 2D translation + yaw) are dynamic. K-means (K1 = 20) runs on DINOv2 appearance embeddings, and a cluster is "mobile" if at least 5% of its members are dynamic. Static objects that look like moving ones are thereby recovered. Table 3 (nuScenes class-agnostic AP / NDS):

  | Configuration | AP / NDS |
  |---|---|
  | HDBSCAN | 13.8 / 15.7 |
  | + motion | 14.2 / 19.8 |
  | + appearance clustering | 39.5 / 31.7 |

  Source: [UNION](https://arxiv.org/html/2405.15688)
- **UP-VL (Najibi et al.):** proposals come from NSFP++ scene-flow and spatio-temporal clustering (DBSCAN 1.0 m, moving threshold ε_sf = 1.0 m/s). A point is background if its cosine similarity to any background text embedding is ≥ ε_bg = 0.02, and a box is dropped if more than r_bg = 99% of its points are background. Table 5, mAP:
  - ε_bg: 0.10 → 20.5; 0.05 → 21.9; 0.02 → 22.4; 0.00 → 22.1
  - r_bg: 50% → 14.1; 90% → 19.2; 99% → 22.4

  Source: [Najibi et al.](https://arxiv.org/html/2309.14491v1)
- **UP-VL distillation:** feature distillation into the 3D detector beats direct projection, 27.6 → 35.8 mAP (Table 3) — [Najibi et al.](https://arxiv.org/html/2309.14491v1)
- **UniLiPs IWU:** the static probability is updated with label consistency C. Points within 30 cm of a map point are pushed by r*·(1 + C), the rest by (1 − r*)·(1 − C), with α = 0.7, and points below τ_s = 0.5 are evicted. Moving candidates need "at least 2 other moving candidates in a neighborhood of 1 meter" (one extraction rendered this "22 ... 11 meter"). They are then HDBSCAN-clustered over 3 consecutive aligned scans and fit with cuboids, PCA yaw, a Kalman filter and splines — [UniLiPs](https://arxiv.org/html/2601.05105)
- **LeAP 3D-CN:** a WaffleIron network trained for one epoch on XYZ, using the per-class top 20% most confident voxel labels. It is fused back into the voxel map by Bayesian update, raising mIoU 48.9 → 57.6 (1 iteration) → 58.1 (2 iterations) — [LeAP](https://arxiv.org/html/2502.03901v1)
- **LOSC:** a point keeps its label only if the class agrees across 10 image augmentations, otherwise *ignore*. Label mIoU rises 39.3 → 45.1 on nuScenes at the cost of coverage (75.8% → 63.7%) (Table 4) — [LOSC](https://arxiv.org/html/2507.07605v2)
- **SAL and SAL-4D:** a lifted mask is replaced by its best DBSCAN cluster at IoU > 0.5 (SAL Table 4: 46.0 → 48.7 PQ) — [SAL](https://arxiv.org/html/2403.13129); [SAL-4D](https://arxiv.org/html/2504.00848)
- **YoCo:** LiDAR or BEV clicks are projected as SAM prompts. A clustering step keeps the click's cluster subject to geometric conditions ("Click SAM" 25.36 → "+Cluster" 40.19 mAP, Table 3). Pseudo-labels are replaced by network predictions when score ≥ T_s2 and IoU ≥ T_IoU, worth +3.17 mAP (Table 2). Threshold values are not given — [YoCo](https://arxiv.org/html/2502.19698v1)
- **SALT and Prutsch et al.:** SALT turns LiDAR into pseudo-images and uses 4D-consistent prompting of SAM2. Prutsch et al. (ECMR 2025) run SAM2 on LiDAR range and reflectivity images. Both come from search summaries and were not full-text verified — [SALT](https://arxiv.org/html/2503.23980v1); [Prutsch et al.](https://a-pru.github.io/files/prutsch_llp_ecmr2025.pdf)

### Inferences
- The closest published analogue to "use LiDAR motion clusters to fix semantics" is UNION's rule: an appearance cluster is a mobile class if at least 5% of its members move. wato_world could use Step F motion clusters the same way, as a check on perception_2d classes (for example, flag masklets of "static-only" appearance that sit on moving clusters, or vice versa). Using them as SAM2 *prompts* for objects the 2D detector missed would be a novel step. I found no published ablation for it in driving auto-labeling.
- LeAP's 3D-CN (one-epoch training on the top 20% most confident labels) is the cheapest "trained" feedback with a large reported gain. It is a candidate only if a small 3D network is acceptable, given that the supervised LiDAR detector is deferred.

### Gaps
- I found no published method that re-prompts SAM or SAM2 from LiDAR motion clusters for driving auto-labeling, and no ablation of such a step.
- YoCo's thresholds (T_s, T_vote, D, T_s2, T_IoU) are not published.
