# Turn LiDAR motion into track-level evidence

Published offline auto-labelers don't use LiDAR motion the way wato_world does today. They don't treat movers as a box source. They use them as evidence about tracks: movers seed the label set, decide whether a track is refined as parked or moving, separate classes geometry confuses (pedestrian vs pole), and vouch for parked objects that look like them. The big gains all come after tracking. VESPA loses 21 mAP without its tracker. 3DAL's whole-track boxes for parked cars beat single-frame boxes by 15 points at IoU 0.7. UNION nearly triples AP by letting movers label look-alike objects. Step F's weakness is a familiar one too. `motion_score` is Chen et al.'s 2022 rule, "moved further than its own size", applied to centroid displacement. Every later method replaced it with shape registration, absolute thresholds, box-shape filters and free-space or persistence checks, and those are the defences that catch WATO's thin wall-like tracks, sub-metre fragments and visibility-drift movers. On the camera side, wato_world's lifting is already stricter than most published labelers, which use no depth test at all. Its remaining problems (one-sided far car boxes, bleed, a hidden car boxed on its occluder) are solved in the literature by re-anchoring boxes and sizing them per track, not by better lifting. The main thing to build is therefore `tracking`, as the stage where motion lives. It should associate every source into tracks, count agreement per track, decide parked vs moving per track, keep tracks on accumulated evidence, and refine the two kinds differently. Cheap Step F filters come first, a registration-based motion estimate and a free-space check after, and untested ideas (re-prompting SAM2 from movers, label-free source reliability) last. Two limits remain. No published method measures 3D label quality without ground truth. And geometry-only motion is weakest on pedestrians and far, sparse objects, so for those objects motion should confirm a camera detection, not find one.

## Lifting is already stricter than most published labelers

Most zero-shot labelers lift masks by plain projection: a LiDAR point joins a mask if it lands inside it, with no depth test. VESPA, SAL, SAL-4D, LeAP, UP-VL, LiSe and YoCo all work this way ([VESPA](https://arxiv.org/html/2507.20397); [SAL](https://arxiv.org/html/2403.13129); [LeAP](https://arxiv.org/html/2502.03901v1)). Only four check visibility:
- UniLiPs keeps a point within 0.5 m of the min-pooled LiDAR depth. That is worth **+2.4 mIoU on KITTI and +4.8 on nuScenes** ([UniLiPs](https://arxiv.org/html/2601.05105)).
- ZOPP runs a depth-ratio parallax filter: +2.6 IoU on vehicles ([ZOPP](https://arxiv.org/html/2411.05311)).
- SLF orders masks by the median depth of their points ([SLF](https://arxiv.org/html/2407.11382v2)).
- SAM4D ray-casts into a voxel reconstruction ([SAM4D](https://arxiv.org/html/2506.21547)).

wato_world tests depth against both a Depth Anything map and a LiDAR z-buffer, then keeps each instance's largest point group. That puts it at the strict end of the field.

Instead of depth tests, the field cleans up in 3D, and that step matters a lot. VESPA keeps the largest XY cluster of each lifted instance and refits its box; **removing that step drops mAP from 46.8 to 27.9** ([VESPA](https://arxiv.org/html/2507.20397)). Two details carry over:
- VESPA sets the cluster radius to the class's average width instead of one value for every class. For wato that means about a car's width for cars and well under a metre for pedestrians, instead of 2 m for both. This directly targets the bled car boxes up to 3.28 m tall.
- SAL replaces a lifted mask with its best-matching DBSCAN cluster (IoU > 0.5) instead of only trimming it. That also adds points the mask missed: 46.0 → 48.7 PQ ([SAL](https://arxiv.org/html/2403.13129)).

Two papers show that eroding masks to fight bleed backfires: SAL loses 3.8 PQ, and SLF loses 4 to 10 points of moderate AP3D. Small and far objects lose most of their few points ([SAL](https://arxiv.org/html/2403.13129); [SLF](https://arxiv.org/html/2407.11382v2)). SAL-4D adds that cleanup must run on each scan, not on an accumulated cloud, or moving objects smear ([SAL-4D](https://arxiv.org/html/2504.00848)).

No published method targets the hidden-car failure wato measured. The front car's edge returns fall inside the hidden car's mask and pass the depth test, because they really are the nearest surface there. ZOPP's parallax filter targets the reverse case (far background inside a near mask), and applied naively it would delete the hidden car's own points. Three published pieces do transfer:
- Process masks near to far by median point depth, so the nearer mask claims shared points (SLF's ordering).
- Drop a box that sits mostly inside another box. DetZero does this with an overlap ratio of 0.3 for vehicles ([DetZero](https://arxiv.org/abs/2306.06023)).
- Keep per-point vote counts across frames instead of a single winning label. The edge points fall inside the hidden mask only in some frames; SAM4D uses this through its per-voxel vote rate ([SAM4D](https://arxiv.org/html/2506.21547)).

Two lifting findings bear directly on motion:
- **Timing.** None of the zero-shot labelers corrects for an object moving between LiDAR time and camera time; they absorb the error with 3D clustering. CATS-V2V shows that shifting each object's points to the camera time helps, but on its single clip the gain over per-frame correction (which wato already does) is small: **recall at IoU 0.5 rises from 0.577 to 0.595**. The big jump was from no correction (0.391) to per-frame correction ([CATS-V2V](https://arxiv.org/html/2511.11168)).
- **Feedback from geometry to semantics.** The only published loop that uses motion is UNION's: a group of similar-looking objects is called mobile if at least 5% of its members move. No one has published prompting SAM or SAM2 from LiDAR movers ([UNION](https://arxiv.org/html/2405.15688)).

Accumulating over time helps far objects at the object level (tracking, stacking a track's points). Map-level label voting is safe only for static points: LeAP reports that movers leave "tracks" smeared through its voxel map ([LeAP](https://arxiv.org/html/2502.03901v1)).

## Far boxes are fixed by re-anchoring and track size, not fusion rules

Every labeler in scope fits boxes with Zhang et al.'s L-shape search. OYSTER names what that produces: tight boxes "are smaller for partially observed objects, especially for those farther away" ([OYSTER](https://research-assets.waabi.ai/wp-content/uploads/2023/05/Oyster-paper.pdf); [CPD](https://arxiv.org/html/2404.16493)). wato's lifted car boxes have exactly this problem: median 2.26 × 0.83 m, against 3.96 × 1.68 m where both sources agree.

The cheapest fix is anchor-and-resize: give the box a class or track size, and pin it to the best-seen corner or edge instead of its centroid.
- **CPD** (training-free) lifts Waymo vehicle 3D AP at IoU 0.7 from **4.8 to 24.0**. Its ablation shows where the gain comes from: **moving the box adds 12.7 points of BEV recall; resizing it adds only 3.9** ([CPD](https://arxiv.org/html/2404.16493)).
- **VESPA** inflates small boxes to class means and pushes them away from the ego. That is worth 18 points of car AP ([VESPA](https://arxiv.org/html/2507.20397)).
- **OYSTER** does it per track: a top-percentile size, re-centred from the corner nearest the ego. It lifts AP@0.7 from 13.2 to 18.1 ([OYSTER](https://research-assets.waabi.ai/wp-content/uploads/2023/05/Oyster-paper.pdf)).

There is a catch. OYSTER's distance-to-collision accuracy fell because some boxes were enlarged along the wrong heading. With about 15 points on one face, a single frame can't tell length from width, so heading has to come from somewhere else: track motion, the mask's shape, or a shape prior.

SLF supplies that shape prior. Fitting a car model to both the points and the SAM mask raises KITTI moderate AP3D from 64.8 (points alone) to 88.0, and with 16 beams it still reaches 76.3 ([SLF](https://arxiv.org/html/2407.11382v2)). It covers cars only and confuses front and rear on sparse returns, so it complements track heading rather than replacing it.

Published fusion methods offer less than they seem to. MS3D's kernel-density box fusion beats NMS and weighted box fusion by 2 to 4 AP3D, mostly at range (10.4 against 7.6 beyond 50 m). But it assumes **more than four boxes within 2 m** of each other, which it gets from many detectors and test-time augmentation ([MS3D](https://arxiv.org/html/2304.02431)). With wato's two sources per frame it reduces to picking one box. The parts worth keeping are small rules:
- choose, don't average, the centroid and heading;
- group boxes by centroid distance rather than IoU;
- weight each box by score times its source's weight ([MS3D code](https://github.com/darrenjkt/MS3D/blob/main/pcdet/utils/box_fusion_utils.py)).

LiSe, the main published LiDAR-plus-camera fuser, simply takes the union, and admits image boxes only beyond 10 m ([LiSe](https://arxiv.org/html/2407.08569)). No paper estimates a source's reliability from the data; MS3D++ sets source weights by hand ([MS3D++](https://arxiv.org/abs/2308.05988)). Fusion4DAL is a trained detector, and its full text is paywalled ([Fusion4DAL](https://link.springer.com/article/10.1007/s11263-025-02370-1)).

Two facts change how fusion should work for wato:
- **The two sources box different objects by design.** UniLiPs, which Steps E and F resemble, boxes only moving points, so like lidar_mos it never boxes a parked car. UNION boxes parked cars only because movers vouch for their appearance ([UniLiPs](https://arxiv.org/html/2601.05105); [UNION](https://arxiv.org/html/2405.15688)). That is why per-frame pairs are scarce.
- **The most-used rejection rule is persistence, and it belongs to tracking.** Published pipelines drop short tracks: OYSTER below six frames, MS3D below 7 to 10 ([OYSTER](https://research-assets.waabi.ai/wp-content/uploads/2023/05/Oyster-paper.pdf); [MS3D++](https://arxiv.org/abs/2308.05988)).

Both point the same way: count agreement per track. A track counts as two-source if lidar_mos matches it in some frames and lifted in others. No auto-labeler rejects a box because LiDAR rays pass through it; free-space reasoning appears only as an input to a detector ([WYSIWYG](https://arxiv.org/pdf/1912.04986)).

## Strong pipelines split parked from moving per track, then refine each differently

Offline trackers share a few habits that need no training:
- **Tracks never die during occlusion**, but only frames with a matching detection are output. This avoids **96% of the vehicle ID switches caused by ending tracks too early** ([Immortal Tracker](https://arxiv.org/abs/2111.13672)).
- **Weak boxes can extend a track, but only strong boxes can start one** ([SimpleTrack](https://arxiv.org/abs/2111.09621)).
- **The tracker runs forward and backward and fuses the two passes.** DetZero's tracker alone raises track-level recall (most of a true track covered by one predicted track) by 16 points over an AB3DMOT-style tracker ([DetZero](https://arxiv.org/abs/2306.06023)).

Offline-Poly packages fragment re-linking, a "drop only if short *and* weak" filter, one size per track and corner alignment into a pipeline written in NumPy with no learning ([Offline-Poly](https://arxiv.org/abs/2602.13772)).

The parked-vs-moving decision is made per track, from the track's own box centres, not per cluster.
- **3DAL** fits a linear rule on two features, centre variance and first-to-last distance, for tracks with at least seven boxes. It gets **over 99% accuracy on vehicles**. It treats every pedestrian as moving, because "even the static ones tend to move their arms and heads" ([3DAL](https://arxiv.org/abs/2103.05073)).
- **MS3D** hard-codes the rule: a track is static if its first-to-last distance is under 1 m and its mean xy variance is under 0.1 m², with 2 m in place of 1 m for pedestrians ([MS3D code](https://github.com/darrenjkt/MS3D/blob/ffed761a6846183966cc38d3144c90d91446fa4a/pcdet/utils/tracker_utils.py#L77-L90)).

MS3D adds four safeguards that wato needs ([MS3D++](https://arxiv.org/abs/2308.05988)):
- A second, strict tracker (BEV IoU 0.7) for parked cars, because jittery boxes otherwise break a parked car into short "moving" pieces.
- A static track that overlaps a moving track's trail is relabelled as moving.
- A track becomes a label only after a few confident detections, and that label then applies to the track's whole life.
- In early rounds only moving pedestrian tracks are kept, because poles get the same scores as pedestrians.

Refinement then splits by state:
- **A parked object gets one box.** 3DAL's whole-track boxes reach **82.3% accuracy at IoU 0.7, against 67.2% for single-frame boxes**. MS3D's 16-frame fusion lifts AP3D from 38.7 to 43.3 ([3DAL](https://arxiv.org/abs/2103.05073); [MS3D](https://arxiv.org/html/2304.02431)). 3DAL fits the box in the frame of the best observation, which beats an averaged frame (Acc@0.8 56.9 against 54.0).
- **A moving object gets one size, a pose per frame, and a smoothed trajectory.** UniLiPs's spline smoothing adds **7.1 mAP** (23.9 → 31.0). Offline-Poly's sliding-window fit edges out Kalman, RTS and Gaussian-process smoothing ([UniLiPs](https://arxiv.org/html/2601.05105); [Offline-Poly](https://arxiv.org/abs/2602.13772)).

One warning from 3DAL: stacking a moving car's points by aligning its noisy boxes did worse than not aligning them. So don't accumulate a moving object's points without real registration ([3DAL](https://arxiv.org/abs/2103.05073)). The learned refiners (DetZero at +7.7 APH, CTRL, LabelFormer) show how much a later trained stage could add, but they need labels ([DetZero](https://arxiv.org/abs/2306.06023); [LabelFormer](https://arxiv.org/abs/2311.01444)). CTRL and DetZero find that splitting parked from moving doesn't help a learned refiner (79.8 vs 80.1 APH). A training-free pipeline still needs the split, because the two refinement procedures are different ([CTRL](https://arxiv.org/abs/2304.12315)).

Two more points complete the downstream picture.
- **Stitch chunks before refining.** No 3D labeler handles chunked sequences, but refinement wants long tracks: CTRL loses **3.0 vehicle APH when tracks are cut at 50 frames instead of 200** ([CTRL](https://arxiv.org/abs/2304.12315)). SUSHI's 2D recipe matches tracks across the overlap by the share of detections they have in common ([SUSHI](https://arxiv.org/abs/2212.03038)). That maps onto wato's 6 s shared span, with detections matched by timestamp.
- **Label quality.** No published method estimates 3D label quality without ground truth. Pipelines use track evidence (length, number of confident frames), agreement between detectors, or a learned re-scorer ([MS3D++](https://arxiv.org/abs/2308.05988); [DetZero](https://arxiv.org/abs/2306.06023)). The evidence on how to use a quality score is clearer than on how to measure one. UA3D down-weights uncertain box coordinates in the student's loss and gains **6.9 AP_BEV** on nuScenes. LMD's after-the-fact quality scores found 43 real annotation errors among the top 100 boxes it flagged in nuScenes ([UA3D](https://arxiv.org/abs/2408.00619); [LMD](https://arxiv.org/abs/2306.07835)). Dawid–Skene-style label models have been applied only to human annotators, so label-free source reliability for 3D is still open.

## Step F measures centroid drift, a cue later methods abandoned

`motion_score` is Chen et al.'s 2022 test: an object is moving if it travelled further than its box's longest side ([Chen et al.](https://arxiv.org/pdf/2201.04501)). Every strong method since measures motion by registering a cluster's shape across frames instead:
- **ICP-Flow** aligns cluster pairs, starting from a histogram of candidate translations and refining with ICP. It rejects a match whose inlier ratio is under 0.2 or whose mean residual is over 0.2 m ([ICP-Flow](https://arxiv.org/html/2402.17351)).
- **UNION** uses a version restricted to ground-plane motion and calls a proposal dynamic once it has moved a fixed 0.5 m ([UNION](https://arxiv.org/html/2405.15688)).
- **Najibi et al.** estimate flow per cluster ([Najibi et al.](https://arxiv.org/pdf/2210.08061)).

The reason is the failure wato measured. A cluster's centroid moves when the visible part of a structure changes as the ego drives past; a registered shape does not. The ratio fails in both directions. A 1.6 × 0.1 m wall fragment whose visible part slid 3 m passed it. A pedestrian whose track breaks into short pieces scored a median of 0.34 and failed it.

Motion on its own is not the answer either. Clustering per-point scene flow gets 0.026 BEV AP on Waymo. In Najibi's ablation, flow plus clustering scores 30.4 mAP, and adding tracking and registration raises it to 42.1 ([LISO](https://arxiv.org/html/2403.07071); [Najibi et al.](https://arxiv.org/pdf/2210.08061)).

Published defences against false movers are layered, and wato has only some of the layers:
- **Remove static points first.** wato's Step B does this.
- **Require a clean registration** (ICP-Flow's thresholds above).
- **Require tracks to persist.** LISO needs at least four steps and a median confidence of 0.3.
- **Require boxes that look like objects.** LISO drops boxes with length over 4 times width, a footprint under 0.35 m², or a volume under 0.5 m³. ViLGOD drops segments under 0.5 m tall ([LISO](https://arxiv.org/html/2403.07071); [ViLGOD](https://arxiv.org/html/2408.03790)).
- **Reject points the map already explains.** M-detector labels points in newly revealed space as static, which is an explicit guard against visibility-drift movers. It runs at 34 ms per nuScenes frame on a CPU ([M-detector](https://pmc.ncbi.nlm.nih.gov/articles/PMC10771495/)).

The wall fragment's length is 16 times its width, four times over LISO's limit. No paper names "wall sliding". Najibi cites the related aperture problem (motion along a flat surface can't be observed) as the reason to remove such surfaces before estimating flow.

wato's measurements line up with this literature:
- **About 5% of clusters pass the gates** (3.4 of 71 per frame).
- **The camera-confirmed half of the ~970 gated tracks**, mostly cars within 30 m, is what the literature calls a high-precision seed, and it works.
- **The thin and sub-metre tracks** are exactly what shape filters exist to delete.
- **The far half** (median 37 m, 22 points) sits where every geometric method fails. ICP-Flow's authors report that ICP "struggles to match sparse clusters", and half of the objects CTRL misses entirely have fewer than five points ([ICP-Flow](https://arxiv.org/html/2402.17351); [CTRL](https://arxiv.org/abs/2304.12315)). The literature doesn't tighten motion gates for these tracks; tightening is what kills pedestrians. It keeps them as low-confidence candidates for appearance, camera or self-training evidence to settle ([MODEST](https://arxiv.org/pdf/2203.15882); [UNION](https://arxiv.org/html/2405.15688)).
- **Only 24 pedestrian tracks is expected.** The best test-time scene flow has twice the error on pedestrians that it has on cars (0.195 vs 0.093 normalised EPE). NSFP captures under 30% of pedestrian motion. LISO's motion-seeded pedestrian AP is 0.029 ([Floxels](https://arxiv.org/html/2503.04718); [Khatri et al.](https://arxiv.org/html/2403.04739); [LISO](https://arxiv.org/html/2403.07071)). For pedestrians, the camera should find them and motion should confirm them.

Above all, published pipelines use motion in more places than boxes. Movers:
- **vouch for parked look-alikes.** UNION goes from 14.2 to 39.5 AP, and Najibi's open-set detector reaches 65.4 vehicle AP without any vehicle labels ([UNION](https://arxiv.org/html/2405.15688); [Najibi et al.](https://arxiv.org/pdf/2210.08061));
- **seed self-training, which is where parked-object recall comes from.** LISO reaches 0.208 AP on still objects against 0.292 on moving ones ([LISO](https://arxiv.org/html/2403.07071));
- **separate pedestrians from poles** ([MS3D++](https://arxiv.org/abs/2308.05988));
- **choose the refinement path and give movers their heading.** VESPA takes heading from motion above 0.5 m/s ([VESPA](https://arxiv.org/html/2507.20397));
- **supply the velocity for correcting each object's camera-LiDAR timing** ([CATS-V2V](https://arxiv.org/html/2511.11168)).

Today Step F fills only part of the first role, and only one frame at a time.

Compute is not a blocker if registration runs only where it is needed. ICP-Flow's official code needs a GPU and takes 3.3 s per frame pair for about 200 clusters ([ICP-Flow repo](https://github.com/yanconglin/ICP-Flow)). Test-time scene-flow methods are far heavier: EulerFlow takes 24 h per sequence, Floxels about 9 minutes per sequence on a T4 ([EulerFlow](https://arxiv.org/html/2410.02031); [Floxels](https://arxiv.org/html/2503.04718)). The portable option is a CPU reimplementation that registers only the gated track candidates, a few per frame instead of 71. Its CPU runtime has not been measured.

## Ten changes, ranked by the problem each solves

The order puts cheap fixes first, then the `tracking` stage that most published gains depend on, then upgrades to the motion estimate, then ideas no paper has tested. The thresholds are published starting values from other sensors, not tuned for WATO, and any limit stated in frames should be restated in seconds for WATO's rates.

| # | Problem it solves | Change | Published basis | Stage |
|---|---|---|---|---|
| 1 | Thin wall-like tracks and sub-metre fragments pass the gates | Add box-shape gates: length/width ≤ 4, footprint ≥ 0.35 m², volume ≥ 0.5 m³. Failing clusters keep their rows and soft features; they just aren't gated. | LISO, ViLGOD, MODEST | Step F |
| 2 | Few frames have both sources (pairs / lidar_mos 0.11–0.56); per-frame motion is noisy | Write `tracking` over all proposal sources. Tracks never die but output only matched frames; weak boxes extend tracks and strong boxes start them; run forward and backward passes. Count source agreement per track. Drop a track only if it is both short and weak; label it once it has a few strong frames (two sources, or a confident lift), and apply that label to its whole life. | VESPA (−21 mAP without tracking), MS3D++, Offline-Poly, Immortal Tracker | tracking |
| 3 | "Moving" is a per-cluster ratio, and jittery parked cars look like movers | Decide parked vs moving per track from its box centres: first-to-last distance < 1 m and xy variance < 0.1 m² means parked (2 m for pedestrians). Add a strict-overlap tracker for parked cars, and relabel a parked track that crosses a mover's trail as moving. Step F's `motion_score` becomes one input among several. | 3DAL (> 99%), MS3D code | tracking |
| 4 | Far cars are boxed on one side; boxes jitter frame to frame | Refine by state. Parked: one box from the whole track's points in world coordinates, fitted in the best-seen frame. Moving: one robust size per track, anchored at the corner nearest the ego, heading from motion, smoothed trajectory. Stitch tracks across chunks first. | 3DAL (+15 pts), CPD, OYSTER, UniLiPs (+7.1 mAP), Offline-Poly, CTRL | tracking → label_refinement |
| 5 | Poles pass as pedestrians; few pedestrian tracks | Let the camera find pedestrians. Accept a LiDAR-only or weakly supported pedestrian track only if it moves a fixed distance. Don't expect Step F to find pedestrians. | MS3D++, Khatri et al., LISO | tracking |
| 6 | Visibility drift fakes motion | Replace centroid displacement with registration of each gated track's points across frames. Accept only clean fits (inlier ratio ≥ 0.2, residual ≤ 0.2 m) and use a fixed distance threshold (~0.5 m). A failed fit means "motion unknown", not "moving". | ICP-Flow, UNION, Najibi et al. | Step F (gated tracks only) |
| 7 | Structure revealed as the ego passes reads as motion | Check free space per track: the space a mover leaves behind must later be seen through. Reuse Step B's ray casting. Measure it before shipping: the per-voxel "void once" rule already flooded the dynamic cloud on WATO. | M-detector, DUFOMap | Step F |
| 8 | Labels carry no confidence, so the student can't weight them and reviewers can't triage | Score each track from evidence already on hand: duration, share of two-source frames, registration and smoothing residuals, spread of box sizes, points in the box, overlap with nested boxes, `n_sources` and `frac_persistent`. Calibrate the score on a small hand-labelled WATO set and use it as a soft loss weight. Fit a label model later. | UA3D (+6.9 AP_BEV), LMD, 3DAL, MS3D++ | label_refinement, student_training |
| 9 | Class-less movers (far, ~22 points) and objects the detector missed | Project gated movers that no mask covers into the cameras as SAM2 prompts (unpublished). Separately, let movers vouch for parked look-alikes by appearance, as UNION does. | UNION (+25 AP); no published re-prompting | perception_2d, semantic_lifting |
| 10 | A moving object's points land beside its mask | Shift each moving instance's points by its track velocity times the camera-LiDAR time gap before the mask vote. Expect a small gain over today's per-frame correction. | CATS-V2V, Jo et al. | semantic_lifting (second pass) |

The research also argues against several things:
- **Don't tighten the `motion_score` thresholds.** The far half needs a different kind of evidence, and pedestrians would be lost first.
- **Don't erode masks to fight bleed.**
- **Don't stack a moving object's points by aligning its noisy boxes.**
- **Don't output boxes extrapolated past a track's ends without re-scoring them.** CTRL's extension paid off only because a trained re-scorer cleaned it up, and MS3D decays such boxes' scores instead ([CTRL](https://arxiv.org/abs/2304.12315); [MS3D++](https://arxiv.org/abs/2308.05988)).
- **Don't run kernel-density fusion on two boxes.**

Three cheap fixes outside Step F come from the same papers:
- size the lifted-instance cluster radius to the class's width (VESPA);
- cap car box height (CPD's vehicle size window stops at 3 m) ([CPD](https://arxiv.org/html/2404.16493));
- filter out boxes nested inside another box, for hidden cars (DetZero) ([DetZero](https://arxiv.org/abs/2306.06023)).

Semantics can also flow back into motion cheaply. UniLiPs weights its static-map update by how consistently each map point is labelled. wato's IWU still runs with that term at zero, and now that semantic_lifting exists it can be filled in. No paper measures that term on its own: UniLiPs only reports removing IWU entirely, which costs 31.0 → 11.7 mAP ([UniLiPs](https://arxiv.org/html/2601.05105)). If WATO recordings revisit the same road, a persistence score across passes is a cheap extra veto on false movers. MODEST's results warn that it is weaker on sparse LiDAR: 11.8 AP on nuScenes against 57.3 on Lyft ([MODEST](https://arxiv.org/pdf/2203.15882)).

## Conclusion

The research changes what Step F is for. Its per-frame boxes are its least valuable output. Its value is as track-level evidence: the switch between parked and moving refinement, the check on pedestrians, the heading for movers, and the velocity for timing correction. That changes how Step F should be judged. The question is no longer how many gated clusters per frame are right, but how often a track's parked-or-moving verdict matches a person's. Answering it needs a small hand-labelled WATO set. The quality score needs the same set for calibration, so building it early pays off twice.

The open ground is also where wato's design is unusual. Label-free source reliability and re-prompting SAM2 from LiDAR movers have no published precedent, and both become tractable only once `tracking` exists, because agreement and coverage can then be counted per track instead of per frame. Two questions remain that only measurement on WATO can settle. First, does registration-based motion hold up at VLP-32C and VLP-16 density? No source tests these sensors; nuScenes's 32-beam LiDAR is the closest proxy, and it is consistently the hardest dataset in these papers. Second, does a training-free single-box fit on a parked car's accumulated points come close to the learned refiners? No paper has measured that.
