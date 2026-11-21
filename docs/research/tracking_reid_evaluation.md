# Tracking and ReID evaluation — 2026-11-11

> Since 2026-11-12 tracking has an appearance consumer (off by default) and
> perception_2d a new descriptor; the measurement that followed this
> evaluation is `docs/tracking_measurements.md` ("Appearance").

The current tracker has **no appearance-based ReID consumer**. It extracts and
saves DINOv2 descriptors in perception, but association, within-chunk re-linking,
and cross-chunk linking use geometry and lifted-instance IDs. This is a missing
capability, rather than evidence of corrupted feature files. See the sourced
[feature-path inspection](dino_reid_feature_path.md).

## Scope and reproducibility

This is an evaluation, not a behavior change. No production code, configuration,
existing track, mask, proposal, or label was changed. The deterministic feedback
loop was a replay of the current tracker/default configuration against cached
proposals, followed by the existing evaluator. No bug-fix hypothesis or regression
test was needed for this request; the existing MOT metric tests passed (8 tests).

nuScenes inputs are the November 7 `nusc0061_new`, `nusc1100_new`, and
`nusc0061x2_new` arms under
`data/artifacts/eval_runs/2026-11-07_mask_policy/`. These have the newer mask
birth gate and backfill, whereas the primary artifact root's nuScenes tracks
date from November 3. Replaying current tracking avoids describing either older
track output or historical documentation numbers as today's implementation.
Upstream perception, lifting and proposals were **not** regenerated.

For WATO, rescored the existing November 11 bag-level tracks from
`sensors_2026_07_07_23_01_24_2_with_eidos_mcap`. All eleven tracking manifests
record the current tracking YAML's SHA256:
`7f2fdeb1b6c8e0b0842ab9e839f4a5edcc5e3872811788ca54f11c4f4faee0ec`.
This verifies configuration identity, not that the entire dirty source tree was
identical at production time. WATO tracking was not replayed in this audit.

The analysis helper follows the existing `link_arm.py` isolated-root approach.
It links inputs, copies only LiDAR index metadata to translate `/data/artifacts`
URIs for host reads, calls `track_chunk()` / `link_and_summarize()`, and writes
new tracks in its own root. It then runs the real `eval_tracks.py` with
`PYTHONHASHSEED=0`. All original
point clouds and source artifacts stay read-only.

Reproduction from repository root:

```bash
PYTHONPATH=src/common/src:src/tracking/src python3 data/artifacts/eval_runs/2026-11-11_tracking_audit/run_audit.py nusc0061
PYTHONPATH=src/common/src:src/tracking/src python3 data/artifacts/eval_runs/2026-11-11_tracking_audit/run_audit.py nusc1100
PYTHONPATH=src/common/src:src/tracking/src python3 data/artifacts/eval_runs/2026-11-11_tracking_audit/run_audit.py nusc0061x2
PYTHONPATH=src/common/src:src/tracking/src python3 data/artifacts/eval_runs/2026-11-11_tracking_audit/run_audit.py wato --saved
PYTHONPATH=src/common/src:src/tracking/src python3 data/artifacts/eval_runs/2026-11-11_tracking_audit/loss_audit.py
PYTHONPATH=src/common/src:src/tracking/src python3 data/artifacts/eval_runs/2026-11-11_tracking_audit/feature_audit.py
PYTHONPATH=src/common/src:src/tracking/src python3 -m pytest -p no:anyio src/common/tests/test_eval_mot.py -q
```

`--saved` also rescores each nuScenes arm's existing tracks for comparison.
The scripts and complete per-event/per-object results are in the local
[audit directory](../../data/artifacts/eval_runs/2026-11-11_tracking_audit/).
The compact [audit JSON](tracking_reid_audit.json) retains aggregate results,
source hashes and feature-integrity measurements; per-run provenance in that
directory records proposal hashes and effective replay configuration.

## Identity continuity

Evaluation uses the repository's class-agnostic MOT matching at 2 m BEV center
distance, at 37 / 38 annotated instants (approximately 2 Hz), with owned chunks
only. Classes outside the taxonomy and GT boxes with no non-ground LiDAR support
are ignored. These are two small nuScenes scenes, not held-out WATO accuracy
or official benchmark results.

| Current tracker replay | 0061 | 1100 | 0061 in two chunks |
| --- | --- | --- | --- |
| ID switches (CLEAR IDSW) | 25 | 7 | 25 |
| Unique GT identities affected by IDSW | 14 | 7 | 13 |
| IDF1 | 0.411 | 0.667 | 0.415 |
| Association recall (AssRe) | 0.481 | 0.780 | 0.485 |
| Association precision (AssPr) | 0.925 | 0.974 | 0.924 |
| HOTA at this matching threshold | 0.372 | 0.648 | 0.374 |
| CLEAR interruptions (Frag) | 52 | 32 | 51 |
| Objects ever matched by the frame evaluator | 43 | 26 | 43 |
| Of these, objects in one output ID | 28 (65.1%) | 16 (61.5%) | 29 (67.4%) |
| Mean output tracks per matched object | 1.51 | 1.42 | 1.51 |
| Track-frame recall / precision | 33.5% / 73.8% | 61.8% / 81.9% | 33.6% / 73.7% |

Use the JSON for full precision. The two-chunk arm links 11 of 13 seam-crossing
objects into one identity (84.6%). Its 25 switches include three whose old/new
observations come from different chunks. It is the same scene, not a third
independent sample.

Rescoring the saved November 7 tracks on **the same proposals** gives 40 / 19
IDSW and IDF1 0.378 / 0.623 on 0061 / 1100. The current replay reduces switches
and raises IDF1; it does not eliminate fragmentation. This comparison changes
tracking relative to saved artifacts, not the YOLO confidence threshold.

### What the switch flags establish

The current evaluator labels 20/25 and 5/7 switches as `early_termination`:
the old track has **no output row** at the switch timestamp. The remaining
5/25 and 2/7 are labelled `wrong_association`, meaning it does have an output
row then. These are descriptive heuristics, not proof of the causal mechanism
or that an internal track was deleted. Today `max_coast_s: null` lets internal
tracks coast until chunk end, but only matched observations are output.

In 0061, 16 of those 20 no-old-output switches transfer to a track already
older than 0.5 s; in 1100, 2 of 5 do. Therefore the issue is not simply a new ID
being born at each reappearance. Concurrent tracks, geometric changes and
matching ambiguity also warrant inspection. Three events in each scene have
the evaluator's duplicate-birth flag. None of these replay switches is flagged
as detector-only birth or geometry-source change.

By class group, switches are 6 vehicle / 12 pedestrian / 7 two-wheeler in 0061,
and 3 / 2 / 2 in 1100. Nearby same-class objects are a useful ReID evaluation
target, but 2 m class-agnostic matching and partial boxes can themselves make
identity assignment ambiguous. A switch count alone is not visual proof of
one object's identity being hijacked.

Sources: [`switch_causes()`](../../src/tracking/scripts/eval_tracks.py#L139),
[MOT implementation](../../src/common/src/wato_common/eval/mot.py),
[tracker output](../../src/tracking/src/wato_tracking/tracker.py),
[continuity-related defaults](../../src/tracking/config/tracking.yaml).

## Missing track observations

A separate coverage audit uses the existing score-ordered frame matcher,
which differs from CLEAR's identity-preserving assignment. An interior gap is
one or more cared-for GT annotations missed between two successful matches,
with either the same or a different output identity. It measures observation
coverage, not only ID changes, and not every high-frequency camera/LiDAR frame.

| Coverage, current replay | 0061 | 1100 |
| --- | --- | --- |
| Cared-for GT objects | 79 | 27 |
| Objects never matched | 36 | 1 |
| GT object-observations missed | 723 / 1,087 | 236 / 617 |
| Ever-matched objects with an interior gap | 30 / 43 (69.8%) | 13 / 26 (50.0%) |
| Interior gap events | 51 | 30 |
| Median time between hits bounding a gap | 1.10 s | 1.50 s |
| Ever-matched objects missing at last cared-for annotation | 31 / 43 | 13 / 26 |

The final row is a coverage tail, not proof of early internal termination.
These measurements mix proposal availability, localization quality and
association failures. They do not isolate how many missing observations a
ReID change could recover. No labelled instant had zero output tracks, so
the existing frame evaluator's skip on an entirely empty prediction frame
does not change these runs' recall totals. At longer range recall is especially
low: 7.6% / 5.3% at 30–50 m, versus 72.2% / 85.4% inside 15 m.

## WATO and feature integrity

The saved current WATO run has 1,379 tracks. Of the 2,016 lifted instances
present in kept tracks, 58.8% appear in one track and **41.2% in more than
one**, averaging 1.70 tracks per instance. Median owned observations per track
are 18, p90 130.2. This is a fragmentation proxy: SAM2/lifted IDs can themselves
split or drift, entirely missing objects do not enter the denominator, and
shared-instance association already favors this proxy. With no `gt_boxes`
for WATO, a true WATO IDSW, loss rate or ReID accuracy cannot be reported.

Inspected **all 6,105 WATO descriptors**, plus all 573 / 168 descriptors from
the nuScenes source arms. Every recorded file exists and loads as finite,
nonzero float32 `(1024,)`. No missing or malformed vector was found. WATO
median L2 norm is 32.29 (range 25.96–46.09); these are not unit vectors.
All 92,228 saved WATO track rows have a null `dino_feature_path`, consistent
with source inspection showing no appearance consumer. Integrity does not
establish same-identity/different-identity discrimination.

Extraction uses mask bounding-box RGB crops, retaining background, then a
square center crop and a CLS descriptor. Whole-masklet aggregation gives every
sample equal weight, without quality filtering or L2 normalization. Wide/tall
objects can be truncated; drift and occlusion can contaminate the mean. These
are inferred risks to evaluate before connecting descriptors to ReID, not
measured causes of the current geometry-only switches. A cosine comparison
can normalize at comparison time; merely normalizing stored means would not
add a consumer or undo identity contamination. Details and primary sources:
[DINO feature-path note](dino_reid_feature_path.md).

## Priorities suggested by the evidence

1. Establish labelled same/different-identity descriptor pairs around the
   recorded switches, occlusion gaps and camera transitions. Compare current
   descriptors with aspect-preserving, foreground-aware and quality-selected
   variants, measuring hard-negative confusion before choosing a similarity
   cutoff. A whole-masklet mean uses future observations; distinguish offline
   linking from causal association when evaluating it.
2. Evaluate an opt-in appearance-assisted association/re-linking path with
   geometry/class gates and retained baseline behavior. Define a shared
   artifact reader and connect masklet descriptors to proposal/track identity;
   currently neither that reader nor the association input exists. Validate
   IDSW, IDF1, coverage and false merges together. Geometry-only coasting
   already fails to retain continuity through many gaps; increasing track
   lifetime further does not address the missing appearance signal.
3. Keep the YOLO sweep separate. Start with 0.35 against 0.25, using the same
   inputs and independent artifact roots; measure range-specific recall and
   gap/ID metrics alongside false positives. DINO currently cannot rescue a
   weak object excluded by a higher confidence threshold. See the
   [confidence analysis](yolo_confidence_threshold.md), whose seed-count audit
   is explicitly not a measured precision–recall curve.

No ReID implementation, feature-quality experiment, YOLO confidence rerun,
or claim of an optimal threshold is included in this evaluation.
