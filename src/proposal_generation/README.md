# proposal_generation

3D box proposals from what upstream already produces: lidar_preprocessing's
Step F moving-object clusters, semantic_lifting's labelled instances and,
when `fusion_detector.enabled`, detection_3d's BEVFusion boxes; one box per
source and one fused box per object, per frame. CPU only. A LiDAR-only
detector is a later source; SLF (fitting shapes to camera masks) runs per
track in label_refinement.
Measurements behind the defaults: `docs/proposal_generation_measurements.md`.

## How a proposal is made

Per frame — lidar_preprocessing's frame_sync tick: a `lidar_cc` sweep and the
corner sweeps matched to it on WATO, one sweep on a single-LiDAR bag:

- **`lidar_mos`**: each Step F cluster passing the gates (`motion_score > 1`,
  `n_sources ≥ 2`, `frac_persistent < 0.5`; `docs/lidar_preprocessing_measurements.md`)
  is one box: Step F's, with its bottom lowered to the ground grid (it stopped
  at Step F's 0.25 m height floor). Shape gates judge the cluster by its Step F
  track's median box (`track_hint_id`): a track longer than 4 × its width is a
  wall strip and is dropped; a track under 0.35 m² or 0.5 m³ keeps a box only
  on frames where a lifted label classes it (a pedestrian is that small). It
  takes a class when ≥ 30% of its points carry camera seed evidence retained
  by surviving 3D lifted owners, by a confidence-weighted vote; otherwise
  `cls` is None. Rejected hypotheses and cluster-acquired points do not cast
  class votes. Its Step F features are
  kept as evidence (`motion_score`, `n_sources`, `frac_persistent`,
  `track_life`).
- **`lifted`**: `lifted.support_method` chooses the support algorithm. The
  default `largest_group` keeps the largest radius-connected group of each
  independent candidate's points (2 m, with optional class overrides).
  Set **`scan_clusters`** to use each instance as a seed for SAL-style scan
  refinement. Build six DBSCAN partitions of the full non-ground frame,
  including stationary/unlabelled returns, and choose the cluster with the
  highest point-set IoU for each original seed. Replace support only at IoU
  **> 0.5**; otherwise keep the original seed, including its disconnected and
  noise points. Replacement can add missed returns and remove contamination;
  it is neither a union nor a largest-component fallback. Matching is
  independent per instance, including competing claims to the same return,
  so refined support may overlap. Points above
  the class height cap (cars: 3 m over ground) are then removed from box
  support. Candidates are visited by descending support size (ties use
  ascending instance ID). A candidate is suppressed at point-set IoM strictly
  above `max_support_iom` (0.5) with an accepted owner; otherwise contested
  points go to the earlier owner. With ≥ 5 remaining points it is boxed
  (`fit_bev_box`, bottom to ground) and checked against accepted owners.
  A fragment: ≥ 80% of its points within 0.3 m of the
  points of a box with ≥ 5× as many, any class (a small mask beside a big one
  claims a few of the big object's edge returns, which box as a thin column
  beside it). A nested box: ≥ 30% of its footprint inside a same-class box
  with more points (a car hidden behind another, boxed on its occluder's edge
  returns). Only accepted boxes suppress later candidates; a rejected box
  consumes no points. Boxes are fit to their final exclusive support.
  SAL evidence: `lift_confidence`, the ORIGINAL seed's mean confidence,
  and `lift_seed_point_count`, its labelled returns retained in the final
  support. The total `lidar_point_count` includes geometric additions. Added
  points receive no new camera votes or semantic labels.
  The radius method averages its own candidate's confidence over final support.
- **`fusion_detector`** (`fusion_detector.enabled`): each detection_3d box with
  score ≥ 0.08 (the car's `confidence_threshold`), outside the car's 2.5 m
  ego zone, of a class in `fusion_detector.class_map` (car, truck, bus,
  motorcycle, bicycle, pedestrian; trailers, construction vehicles, barriers
  and cones are not relabelled into them). Its box is the model's — centre,
  size, heading (made undirected here; the directed one stays in
  `detections_3d`) — not refit to points; `lidar_point_count` counts the
  frame's non-ground points inside it, `score` is the model's, `cls` the
  mapped class.
- **`fused`**: a lidar_mos and a lifted box sharing ≥ 5 LiDAR points, ≥ 30%
  of the smaller one's, are one object (greedy, most shared first, one to one):
  a box fit to both point sets, with the lifted class. Every other box is
  fused on its own, class-less if it has none, so each object either source
  found has one fused row. A fusion_detector box then joins the object it
  shares the most points with by the same rule, one to one; the object takes
  the detector's box (refitting would discard the sides it predicts), keeps
  the lifted class if any (else the detector's) and averages the members'
  scores. A detector box no object matched is fused on its own when
  `fusion_detector.add_unmatched`. `member_proposal_ids` lists the source rows.

Scores are placeholders until per-source reliability is measured: lidar_mos
`min(1, motion_score / 2)`, lifted `lift_confidence`, fused the members' mean.
Boxes are yaw-only and undirected (heading mod π). Neighbouring chunks overlap
by 6 s and both label the shared span; tracking links them.

The opt-in refinement follows [SAL Appendix A.1/Table A.1](https://arxiv.org/html/2403.13129#A1.SS1):
six published thresholds in their original order, strict IoU acceptance,
replacement, and unchanged-seed fallback. Implementation choices the paper
does not specify are Euclidean XYZ radii in metres, `min_samples: 5` (counting
self), and deterministic scale/input-order ties. DBSCAN expands only through
core points; a border point does not propagate connectivity. The implementation
uses the existing SciPy dependency without voxelization, downsampling, or
temporal accumulation. Refinement uses this pipeline's already merged instance
seeds and existing synchronized multi-LiDAR frames; SAL Algorithm 1 instead
refines per-camera masks before cross-camera fusion on a scan. These are
adaptations, not a claim of exact SAL reproduction. Ownership follows
[SAL-4D's lift-before-flatten ordering](https://arxiv.org/html/2504.00848#S3.SS2.SSS1),
using support cardinality in one frame instead of temporal volume. The
coverage threshold is SAL-4D Appendix A.1.1's strict IoM > 0.5; residual
exclusive assignment and survivor-only box gates are local choices.
Source verification,
choices, and limitations: [design note](../../docs/research/sal_scan_cluster_refinement.md).
It remains opt-in because a frozen nuScenes comparison reduced lifted-proposal
precision and recall; recovered points alone did not establish a net gain.

Strict IoU cannot expand a clean seed to a cluster twice its size or larger.
Dense chains and neighboring surfaces can still merge; matching can favor
background when the seed is contaminated. Existing height/fragment/nested-box
filters are separate box postprocessing, not SAL refinement, and can reject a
refined instance. Synthetic adjacent-pedestrian tests explicitly cover this
interaction. No semantic veto, class-specific radius, expansion cap, or
one-to-one cluster allocation is added to the matcher. Final ownership is a
separate step after matching.

## Run

`ingest`, `lidar_preprocessing` (with its `proposals` step) and
`semantic_lifting` must have run first, and `detection_3d` when
`fusion_detector.enabled`.

Regenerating old owned-mask outputs requires running `perception_2d`, then
`semantic_lifting`, then `proposal_generation` with `--force` for the same
bag/chunk and profiles. The old PNGs cannot reconstruct losing claims; missing
independent-mask metadata or candidate files raises a rerun instruction.

```bash
./watod run proposal_generation --bag <bag_id> [--chunk <chunk_id>] [--force] \
    --config /ws/src/proposal_generation/config/proposal_generation.wato.yaml
```

Chunks whose `proposals.parquet` exists are skipped unless `--force`. Each
chunk gets `manifest_proposal_generation.json` (wato_common.manifest): config,
build provenance, and hashed inputs — the per-sweep files through their
producers' manifests and indexes. The earlier 8–21 s WATO chunk measurement
predates full-scan refinement and is not its runtime estimate. Existing outputs
are skipped; use `--force` to regenerate them with refinement.

## Viewers

```bash
./watod run proposal_generation viz --bag <bag_id> --chunk <chunk_id> \
    --config <same config> --view 3d|bev|camera [--show sources|fused] [--every N]
```

Under `chunks/<chunk_id>/viz/`:

- `--view 3d` → `proposals_<show>.html`: each frame's non-ground points (all
  inside a box, a sample of the rest) and box wireframes over grey context,
  with a slider over frames (`wato_common.viz.point_cloud_html`; wheel zooms,
  drag pans, "follow" keeps the frame centred). Every 5th frame by default;
  36 MB for a WATO chunk.
- `--view bev` → `proposals_bev_<show>/<sweep_id>.png`: top-down, 50 m around
  the reference LiDAR, every 10th frame.
- `--view camera` → `proposals_cam_<show>/<cam>/<seq>.jpg`: boxes projected
  into every 10th image of each camera (`--camera` to pick), with the ego pose
  at the image's own time. Each image shows the single nearest proposal frame
  among its paired sweeps' frames (earlier time wins a tie), including an empty
  frame. Combining several frame times would draw a moving object at several
  positions. Boxes retain their reference sweep time; object motion to the
  image time is not compensated, so a moving car can still appear offset.

`--show sources` colours lidar_mos orange and lifted cyan; `--show fused`
colours the fused boxes by class (grey: none).

## Inputs

Every cross-component artifact is read through `wato_common.artifacts`, which
owns its format:

| Artifact | Producer | Read through |
|---|---|---|
| `lidar_proc_index.parquet`, `lidar_proc/<sweep_id:06d>_world.npz` | lidar_preprocessing | `lidar_sweeps.ProcessedSweeps` (`.frames(canonical_lidar)` groups sweeps into frames) |
| `motion_clusters.parquet`, `lidar_proc/<sweep_id:06d>_motion_proposals.npz` | lidar_preprocessing Step F | `motion_proposals` |
| `ground.npz` | lidar_preprocessing Step C | `ground` (`GroundGrid.height_at`); a sentinel leaves boxes' bottoms as they are |
| `semantic_lifting/lifted_labels/`, `lifted_instances.parquet` | semantic_lifting | `lifted_labels` |
| `semantic_lifting/lifted_candidates/` (all independent point-instance claims) | semantic_lifting | `lifted_candidates` |
| `detections_3d.parquet` (`fusion_detector.enabled` only) | detection_3d | `detections_3d` |
| `calibration.json`, `frame_index.parquet`, `poses.parquet` (camera viewer only) | ingest | `calibration`, `frame_index.FrameIndex` |

No artifact records lidar_preprocessing's canonical LiDAR, so the config states
it (`frames.canonical_lidar`). A run fails with `CanonicalLidarError` when it
doesn't match: a frame without exactly one sweep from it, or a Step F cluster
not stamped with its frame's reference time.

Camera overlays use the shared full-intrinsic/lens
[camera model](../../docs/camera_geometry_contract.md) and check image
dimensions against calibration. Viewing leaves stored proposal geometry intact.

## Output

`proposals.parquet`, written through `wato_common.artifacts.proposals`, one
`ProposalRow` per box:

- `sweep_id` is the frame's reference sweep (the canonical LiDAR's),
  `sweep_ids` all its sweeps, `reference_timestamp_ns` its time. `frame_id` is
  None for a sweep frame_sync matched to no canonical sweep (a Step F
  cluster's `-1` maps to None).
- `provenance` `lidar_mos` / `lifted` / `fusion_detector` / `fused`; tracking
  reads the fused rows, the source rows stay to measure each source.
- `cls` is perception_2d's class (`car`, `truck`, `bus`, `motorcycle`,
  `bicycle`, `pedestrian` under the fixed taxonomy), else the detector's
  mapped class, or None.
- The evidence columns are null except on rows of the source that measures
  them.

`proposal_refinement.parquet` is a component-local diagnostic table, written
before the proposals completion marker. It includes EVERY instance/frame
match, including ones later rejected by box filters: method, status, best scale and
cluster ID, best IoU, seed/refined/addition/removal counts, whether a proposal
was emitted after coverage and box gates, and final exclusive geometric/retained-seed
counts when emitted. The
manifest records both outputs. No downstream component consumes this table.

Older proposal files without `lift_seed_point_count` remain readable. Their
original total count serves as labelled evidence in tracking. New lifted rows
always carry the separate count; fused rows keep evidence on their source
members.

## Configuration

`config/proposal_generation.yaml` (single-LiDAR bags, `canonical_lidar: null`)
and `config/proposal_generation.wato.yaml` (`canonical_lidar: lidar_cc`, as
`lidar_preprocessing.wato.yaml`'s frame_sync; a test fails if anything else
drifts). Every field is documented in
[`config.py`](src/wato_proposal_generation/config.py).
Set `lifted.support_method: scan_clusters` to opt in. Under that method,
`lifted.refinement.enabled: false` is an original-support ablation; it never
substitutes largest-group filtering for rejected matches. The legacy radius
fields are only used with `support_method: largest_group`. Ground masks are
required for either method: proposal points remain non-ground.
