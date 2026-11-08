# Deferred 3D mask ownership

Implemented 2026-11-06. Competing masks now survive perception and lifting;
proposal generation decides ownership after each hypothesis sees 3D geometry.

## Source and scope

[SAL-4D §3.2.1](https://arxiv.org/html/2504.00848#S3.SS2.SSS1) lifts overlapping
masklets, refines them geometrically, and then flattens them.
[Appendix A.1.1](https://arxiv.org/html/2504.00848#A1.SS1.SSS1) orders masklets
by temporal point volume and suppresses at strict intersection-over-minimum
above 0.5. This implementation adopts the ordering of lifting, refinement and
ownership, and that coverage threshold. It ranks support within a synchronized
proposal frame rather than accumulating a temporal window. It does not
implement SAL-4D's complete 4D masklet engine or cross-window association.

## Data path

1. **Perception saves independent masks.** Ego-vehicle exclusion and per-mask
   island cleanup remain. Pixel-logit ownership and overlap-based track
   duplicate, collapse and containment pruning are removed. Lost-object and
   detector-support lifecycle still bound tracking. The masklet table records
   `mask_semantics="independent"`; readers reject legacy owned-mask output.
2. **Lifting preserves every visible claim.** Projection and physical visibility
   are shared per point/camera, then each covering mask supplies a claim.
   Cached masks use cropped rectangles with image origins, keeping memory
   proportional to their rectangles rather than full images per object.
   Same-camera competition does not supply cross-camera identity evidence.
3. **A sparse artifact carries the hypotheses.**
   `semantic_lifting/lifted_candidates/<sweep_id:06d>.npz` contains aligned
   `point_indices`, `instance_ids`, and `confidences`. Each point-instance pair
   occurs once; one point can occur for several instances. Cross-camera claims
   of one merged instance are deduplicated. The instance table includes
   hypotheses that win zero provisional labels. `lifted_labels` remains a
   provisional unique view for diagnostics and existing consumers.
4. **Proposal refinement uses candidates.** Frame points keep one row per
   original `(sweep_id, point_index)`, plus independent seed sets over those
   rows. Both radius cleanup and the opt-in scan-cluster matcher operate on
   each seed independently. The matcher shares its full non-ground DBSCAN bank
   across hypotheses and can return overlapping refined support.
5. **Ownership commits only for accepted geometry.** Height-cap support first.
   Visit candidates by descending support cardinality, breaking ties by
   instance ID. Suppress a candidate whose IoM with an accepted hypothesis's
   full support exceeds `lifted.max_support_iom` (default 0.5). Otherwise
   propose its unowned returns, require the final point floor, fit that box,
   and check nested/fragment gates against accepted final boxes. Commit its
   points only if every gate passes. Accepted owners never disappear in a
   later filter. A rejected candidate neither owns points nor suppresses
   another candidate. Final lifted memberships are disjoint.
6. **Motion class votes use surviving camera evidence.** The proposal pipeline
   rebuilds its semantic view from surviving owners' retained original seeds,
   then classes motion proposals. A rejected provisional winner cannot class
   an unmatched motion box. Cluster-acquired geometry creates no camera votes.

Candidate confidence retains the existing class-agreement meaning. Each
camera contributes its strongest masklet weight per class; duplicates do not
multiply camera evidence. Scan refinement preserves mean original seed
confidence; radius cleanup averages the candidate's retained seed confidence.
`lift_seed_point_count` counts retained original seeds and `lidar_point_count`
counts final geometry. Neither is read from a competing provisional owner.

## Regenerating existing artifacts

Old owned PNGs cannot reconstruct the losing claims. Run `perception_2d`,
`semantic_lifting`, and `proposal_generation`, in that order, with `--force`,
using the same bag, chunk selection and sensor profiles. Rerun downstream
tracking and refinement to consume the new proposals. Missing mask semantics
or candidate files raises a rerun instruction. No stored recording was
regenerated as part of this code change.

`largest_group` remains the shipped support method. Deferred ownership applies
to both methods; selecting `scan_clusters` remains an explicit refinement
experiment. The previous frozen-mask scan-refinement comparison predates
independent masks and deferred ownership and does not measure this change.

## Validation and remaining limits

Host validation passed 544 tests across common, perception, lifting, proposals,
tracking and label refinement, with three optional tests skipped. Ruff lint,
formatting and whitespace checks passed for changed Python files.

Regressions cover saved overlapping PNGs, a lower-scoring hypothesis that wins
zero provisional labels, visibility filtering, bounded camera evidence,
same-camera identity separation, candidate artifact validation, multi-scanner
point keys, shared-bank refinement, strict IoM ties, partial overlap and final
box fitting, rejected-owner recovery, and fragment/nested filter cascades.
The staged proposal regression removes every real object's provisional label
and verifies its complete candidate support still produces the correct box
and motion class.

These tests establish recoverability and ordering, not real-data accuracy.
Large contaminated hypotheses can still outrank smaller correct ones; geometry
does not identify the true object automatically. Refinement still merges
cross-camera identity before matching, and moving actors are not compensated
between cameras/scanners. The unique lifting artifact remains provisional;
it is not rewritten from proposal ownership. Motion/lifted box fusion still
uses a union of matched sources and is not a panoptic point-label export.
Removing early overlap pruning can increase active tracks and stored masks;
CPU/GPU runtime and precision/recall need a new frozen-input evaluation.
