# SAL-style scan cluster refinement

Research checked 2026-11-06. Scope: the geometric refinement in SAL's pseudo-label engine, with implementation choices explicitly separated below.

## Verified source behavior

SAL builds a DBSCAN cluster ensemble from each scan after ground removal using Patchwork++. Algorithm 1 builds that ensemble before its camera loop: its input is the scan, rather than each lifted instance's selected points. [Appendix A.1](https://arxiv.org/html/2403.13129#A1.SS1)

| Item | SAL description |
| --- | --- |
| Density thresholds | `1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221` |
| Match | Best point-mask IoU across the ensemble |
| Acceptance | IoU exceeds `0.5` |
| Accepted output | Replace the lifted mask with the matched cluster |
| Rejected output | Keep the original lifted mask |
| Preserved | Number of masks and associated CLIP features |

These values and rules come from [Appendix A.1 and Table A.1](https://arxiv.org/html/2403.13129#A1.SS1). The paper calls these values density thresholds; it does not explicitly specify `min_samples`, radius units, backend, tie-breaking, or a cluster-exclusivity policy. Union belongs to cross-camera fusion, not cluster replacement. Algorithm 1 refines before cross-camera insertion/merging, although the surrounding prose describes fusion first. [Appendix A.1](https://arxiv.org/html/2403.13129#A1.SS1)

The [authors' repository](https://github.com/nv-dvl/segment-anything-lidar/tree/aed1c4ab6214186b91e7e3a56ad6e45f578a7e02) contains only `README.md` and `LICENSE` at checked commit `aed1c4ab6214186b91e7e3a56ad6e45f578a7e02`; its README says code release is pending. Therefore no released implementation was available there to resolve these omissions. Treat any chosen `min_samples` as a project parameter, not a verified SAL default.

## Consequences for the proposed change

The following are deductions from point-set matching, not additional claims about SAL's implementation.

A cluster bank built from the full non-ground scan can contain points that never entered an instance's lifted support. Replacing support with such a cluster therefore permits both expansion and removal. Merely rerunning a largest-component filter inside the lifted subset cannot introduce any omitted points.

That ability does not guarantee object completion. Let the lifted support be `L`, the candidate cluster be `C`, and `I = |L ∩ C|`. The score is `I / (|L| + |C| - I)`. If a clean lifted subset contains only 30 of a complete cluster's 100 points, its IoU is `0.3`; the `0.5` gate rejects completion. The fallback deliberately leaves the original 30 points. Conversely, a cluster can pass the gate while including neighboring-object points when those points are a sufficiently small fraction of the union. The matching objective has no semantic boundary oracle.

Ground filtering also affects candidate availability: points removed from the candidate scan cannot be recovered through replacement. DBSCAN noise points need not belong to any candidate. The fallback can retain such points when they already occur in the original support.

## Implemented behavior

[`frame_points.load`](../../src/proposal_generation/src/wato_proposal_generation/frame_points.py) now retains every non-ground return, preserving `(sweep_id, point_index)` and the original semantic/motion columns. Previously its union of lifted labels and gated motion-cluster points excluded most unlabeled surfaces. Missing ground masks stop the run; road points are not silently admitted. Ground masks were already required upstream by lifting.

[`scan_refinement.refine`](../../src/proposal_generation/src/wato_proposal_generation/scan_refinement.py) builds six DBSCAN partitions once per frame and matches each original instance independently when `lifted.support_method: scan_clusters` is selected. It retains the strict replacement/fallback rule. No mask is deleted by this matcher. Seeds now come from independent `lifted_candidates` claims: several hypotheses can claim the same return and match the same cluster. Final proposal ownership is resolved separately after refinement; see [deferred 3D mask ownership](deferred_3d_mask_ownership.md).

[`lifted.proposals`](../../src/proposal_generation/src/wato_proposal_generation/lifted.py) uses the refined support for geometry and fusion membership. Existing height caps, the five-point floor, grounding, fragment rejection, and nested-box rejection remain separate postprocessing. The five-point floor applies after matching, allowing a three-point seed to recover a five-point cluster. The height cap is also applied after matching; the earlier outline's proposal to prefilter heights was not adopted because it would change SAL's seed IoU. A selected component entirely above a class cap can consequently still produce no box. That limitation remains explicit rather than being hidden inside geometric matching.

The shared `largest_group` helper remains unchanged for parked-track refinement. Existing moving-cluster/lifted fusion still unions matched geometry, providing an additional recovery path. `label_refinement.gather` already loads full non-ground scans and crops them to track regions, so it does not require a new persisted membership format.

## Assumptions and choices

| Choice | Reason and scope |
| --- | --- |
| Euclidean XYZ distances; six values interpreted as radii in metres | A conventional spatial DBSCAN interpretation for metric LiDAR coordinates. SAL leaves metric, XYZ/XYZI, parameter name, and units unstated; this is an assumption, not a verified author setting. |
| `min_samples: 5`, including self | SAL omits it. Five is the [standard scikit-learn default](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html); it is configurable and is not presented as a SAL default. |
| Existing SciPy backend | [Exact `cKDTree` queries](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html), `p=2`, query approximation `eps=0`, need no new dependency or lock change. SciPy's `eps` here is search approximation, not the clustering radius. |
| Core-only expansion; stable border assignment | Standard DBSCAN: borders never propagate connectivity. Core components are discovered in input order; ambiguous borders attach to the first adjacent component, following the [official implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_dbscan_inner.pyx) and [ordering convention](https://scikit-learn.org/stable/modules/clustering.html#dbscan). |
| Exact tree rebuilds for efficiency | Once half a search tree's cores are assigned, rebuild it over the unassigned cores. Every original return still contributes to density counts and IoU. This avoids repeatedly materializing dense visited neighborhoods without voxelizing, sampling, limiting neighbors, or approximating distances. |
| Equal-IoU ties keep the first published scale, then first component | SAL does not specify ties. Stable scale and point order make repeated runs deterministic. |
| Refine merged independent instances in proposal generation | `lifted_candidates` preserves competing point-instance claims while retaining lifting's cross-camera identities. SAL Algorithm 1 instead refines per-camera masks before fusion; that difference remains explicit. |
| Use the existing synchronized frame on multi-LiDAR data | Proposals and motion clusters already use that frame; single-LiDAR data uses one sweep. WATO's configured synchronization tolerance is 25 ms. Ego deskew does not compensate object motion between sensors; no adjacent-time accumulation or new compensation model is introduced. |
| Preserve original mean seed confidence; count retained seeds separately | SAL preserves the associated semantic token. Our corresponding semantic evidence stays with the original seed; cluster-only returns do not acquire camera votes. `lift_seed_point_count` counts original labeled returns retained in final support, while `lidar_point_count` counts total geometry. Tracking's strong-lift gate uses the former. |
| Keep box gates with survivor-only suppression | Height, fragment and nested thresholds remain separately configured repository behavior. Final acceptance compares against surviving owners only; a rejected candidate cannot erase a later hypothesis. |
| Keep `largest_group` as the default; select `scan_clusters` explicitly | The measured comparison below loses precision and recall. Keeping the old default avoids silently promoting an unvalidated change. SAL matching remains intact as a separate selectable path, rather than gaining unpublished gates. |

Under `support_method: scan_clusters`, `lifted.refinement.enabled: false` retains original support for ablation and never substitutes legacy cleanup for failed matches. The `largest_group` path retains its radius fields and confidence averaging over final support; both paths now use independent candidates and deferred ownership. Older proposal files remain readable without the new seed-count column; their original total labeled count supplies legacy tracking evidence. New source rows always write the separate count, and fused rows keep evidence on their members.

`proposal_refinement.parquet` records every instance/frame match: support method, best scale/component and IoU, status, seed/refined/addition/removal counts, final emission, and final counts when emitted. Best-candidate metadata is retained even for a rejected match. It is component-local diagnostics, written before the proposals completion marker and included in the manifest. It does not export per-point membership or independently verified point accuracy.

## Validation and remaining limits

Final pinned-image checks: `./watod test proposal_generation -q` passed 60 proposal tests and 195 common tests; `./watod test tracking -q` passed 14 tracking tests and the same 195 common tests. Each image skipped the optional `test_progress` module because `tqdm` is absent. Ruff and `git diff --check` passed. No dependency or lockfile changes were needed.

Behavioral tests cover unlabelled-point recovery and contamination removal; strict threshold equality; original fallback including noise/disconnected support; multiscale selection and ties; self-counting and coincident-return multiplicity; sparse-border chains versus dense-core chains; cross-scale overlaps; missing ground; stable sweep point indices; original-confidence preservation; small-seed recovery before the box floor; artifact round trips and legacy compatibility; and tracking's seed evidence gate. A staged full pipeline verifies static recovery through input loading, fitting, audit output, and unmatched fusion.

Existing fragment/nested-filter tests run with refinement disabled to isolate those postfilters. The adjacent-car/pedestrian fixture also explicitly runs the new full path: gaps of 0.05 and 0.1 m lose the pedestrian after car-support expansion and fragment rejection; 0.3 m keeps it. This is an observed synthetic regression, not a claim that DBSCAN guarantees neighboring-object separation. No unpublished veto was added to make that case pass.

Strict IoU rejects severe missing-support cases; ground misclassification and DBSCAN noise limit available recovery. Dense bridges can merge objects, contaminated seeds can choose background, and fully unobserved surfaces still require shape priors or temporal completion. Box-level ground truth can evaluate proposals, but point precision/recall requires independent point annotations. Broader tracking/refinement replay and multi-rig quality evaluation remain necessary before claiming an end-to-end improvement.

## Frozen-input comparison, 2026-11-06

This comparison predates independent masks and deferred ownership. It does not measure their combined effect; rerun the upstream perception and lifting stages for that evaluation.

It compares the lifted-box implementation at commit `c65a2e6` with the SAL path on exactly the same upstream labels, world returns, ground grid, classes, and box postfilters.

For `NuScenes_v1_0_mini_scene_0061_0_mcap`, chunk 0000, the comparison uses all 37 ground-truth instants with a valid nearby scan within 25 ms. Scores are lifted-only, class-agnostic center matching at 2 m using `wato_common.eval.boxes`, with unseen/out-of-taxonomy GT ignored as in the existing evaluator. These results are not fused-proposal, tracking, or final-refinement scores.

| Lifted-only metric | Existing largest group | SAL-style scan clusters |
| --- | ---: | ---: |
| Recall | 32.29% | 30.54% |
| Precision | 70.77% | 59.29% |
| Matched cared-for boxes | 351 / 1,087 | 332 / 1,087 |
| False positives | 145 | 228 |
| Median 3D IoU on matches | 0.116 | 0.119 |
| Car recall | 52.50% | 50.00% |
| Pedestrian recall | 25.14% | 23.92% |
| Truck recall | 53.98% | 49.56% |

The new matcher replaced 392 of 1,353 instance/frame supports, adding 4,177 returns and removing 2,387. It emitted 629 lifted boxes versus 556 previously, including predictions matched to ignore-GT. Median IoU is computed on each method's own matches, so its small increase is not a paired demonstration of surface recovery. Point counts are likewise not point-precision measurements.

For WATO's `sensors_2026_07_07_23_01_24_2_with_eidos_mcap`, chunk 0000, eight uniformly spaced synchronized frames were inspected (reference sweeps 11, 189, 369, 549, 727, 907, 1087, 1267). The matcher replaced 24 of 216 supports, adding 1,390 returns and removing 52; it emitted 75 lifted boxes versus 73. No independent ground truth was available for these WATO samples, so these counts do not establish improved geometry or ownership.

On the host environment (Python 3.10, NumPy 2.2.6, SciPy 1.15.3), median clustering/matching time was 4.25 s per selected nuScenes scan (maximum 14.21 s) and 0.54 s per selected WATO frame (maximum 0.67 s). These timings exclude loading and box fitting, can vary with concurrent host work, and are not measurements of the pinned container or full-chunk runtime.

**Decision:** retain `largest_group` as the shipped default and expose the complete SAL-style path through `lifted.support_method: scan_clusters`. The measured precision/recall regression does not support default promotion. The strict fallback, merged-mask adaptation, scale/density transfer between sensors, and interaction with existing duplicate filters are plausible contributors that require separate controlled ablations; this comparison does not isolate their causes. No unpublished guard was added to improve these numbers.
