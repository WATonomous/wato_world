# proposal_generation

3D box proposals from what upstream already produces: lidar_preprocessing's
Step F moving-object clusters and semantic_lifting's labelled instances, one
box per source and one fused box per object, per frame. CPU only. A LiDAR
detector is a later source; SLF (fitting shapes to camera masks) runs per
track in label_refinement.
Measurements behind the defaults: `docs/proposal_generation_measurements.md`.

## How a proposal is made

Per frame — lidar_preprocessing's frame_sync tick: a `lidar_cc` sweep and the
corner sweeps matched to it on WATO, one sweep on a single-LiDAR bag:

- **`lidar_mos`**: each Step F cluster passing the gates (`motion_score > 1`,
  `n_sources ≥ 2`, `frac_persistent < 0.5`; `docs/research/lidar_mos_guidance.md`)
  is one box: Step F's, with its bottom lowered to the ground grid (it stopped
  at Step F's 0.25 m height floor). Shape gates judge the cluster by its Step F
  track's median box (`track_hint_id`): a track longer than 4 × its width is a
  wall strip and is dropped; a track under 0.35 m² or 0.5 m³ keeps a box only
  on frames where a lifted label classes it (a pedestrian is that small). It
  takes a class when ≥ 30% of its points carry a lifted label, by a
  confidence-weighted vote; otherwise `cls` is None. Its Step F features are
  kept as evidence (`motion_score`, `n_sources`, `frac_persistent`,
  `track_life`).
- **`lifted`**: each semantic_lifting instance's labelled points in the frame.
  Masklets bleed onto background, so only the largest group of points chained
  within the class's radius (2 m; less for pedestrians and two-wheelers) is the
  object, minus points above the class's height cap (cars: 3 m over the
  ground); with ≥ 5 points it is boxed (`fit_bev_box`, bottom to the ground)
  and takes the instance's class and cameras. Two filters then drop a box that
  is another object's. A fragment: ≥ 80% of its points within 0.3 m of the
  points of a box with ≥ 5× as many, any class (a small mask beside a big one
  claims a few of the big object's edge returns, which box as a thin column
  beside it). A nested box: ≥ 30% of its footprint inside a same-class box
  with more points (a car hidden behind another, boxed on its occluder's edge
  returns). Evidence: `lift_confidence`, the points' mean lifting confidence.
- **`fused`**: a lidar_mos and a lifted box sharing ≥ 5 LiDAR points, ≥ 30%
  of the smaller one's, are one object (greedy, most shared first, one to one):
  a box fit to both point sets, with the lifted class. Every other box is
  fused on its own, class-less if it has none, so each object either source
  found has one fused row. `member_proposal_ids` lists the source rows.

Scores are placeholders until per-source reliability is measured: lidar_mos
`min(1, motion_score / 2)`, lifted `lift_confidence`, fused the members' mean.
Boxes are yaw-only and undirected (heading mod π). Neighbouring chunks overlap
by 6 s and both label the shared span; tracking links them.

## Run

`ingest`, `lidar_preprocessing` (with its `proposals` step) and
`semantic_lifting` must have run first.

```bash
./watod run proposal_generation --bag <bag_id> [--chunk <chunk_id>] [--force] \
    --config /ws/src/proposal_generation/config/proposal_generation.wato.yaml
```

Chunks whose `proposals.parquet` exists are skipped unless `--force`. Each
chunk gets `manifest_proposal_generation.json` (wato_common.manifest): config,
build provenance, and hashed inputs — the per-sweep files through their
producers' manifests and indexes. A WATO chunk takes 8–21 s.

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
- `provenance` `lidar_mos` / `lifted` / `fused`; tracking reads the fused rows,
  the source rows stay to measure each source.
- `cls` is perception_2d's class (`car`, `truck`, `bus`, `motorcycle`,
  `bicycle`, `pedestrian` under the fixed taxonomy), or None.
- The evidence columns are null except on rows of the source that measures
  them.

## Configuration

`config/proposal_generation.yaml` (single-LiDAR bags, `canonical_lidar: null`)
and `config/proposal_generation.wato.yaml` (`canonical_lidar: lidar_cc`, as
`lidar_preprocessing.wato.yaml`'s frame_sync; a test fails if anything else
drifts). Every field is documented in
[`config.py`](src/wato_proposal_generation/config.py).
