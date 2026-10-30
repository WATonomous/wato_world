# semantic_lifting

Occlusion-aware 2D → 3D label lifting: each LiDAR point takes the label of the
perception_2d masklet it projects into, if nothing in front of it hides it from
that camera (UniLiPs Eq. 1), with votes pooled across cameras.
Design and rationale: `docs/research/semantic_lifting_design.md`; measurements
behind the defaults: `docs/semantic_lifting_measurements.md`.

## How a point gets its label

For each sweep, every camera frame ingest paired with it (`frame_index`) is
used; the points are already world-registered, so they are projected with the
camera's pose at the frame's own time. A point inside a mask that passes the
visibility test (below) gets one vote from that camera: the innermost (smallest) mask
it lands in, weighted by the masklet's `det_score`. Votes are resolved per
point by **class** first — the class with the highest summed score across
cameras — then, within it, the highest-scoring instance. Masklets are
per-camera, so two cameras on one car agree on its class even though their
masklet ids differ.

**Instances.** Before votes are resolved, one object's masklets are merged
across cameras into one instance (`instances.py`): two same-class masklets
from different cameras merge when they claim at least 20 of the same visible
points over the chunk, ≥ 30% of the smaller one's claims where they overlap,
and stay within 5 m of each other (median, and for every pair in the group)
in the sweeps where both claim points. A merge is also refused when the group
would hold two masklets one camera saw in the same frame. On the WATO rig 69%
of labeled points end up on a multi-camera instance. Instance quality is
bounded by masklet quality: a masklet that drifts onto another object or a
mask on the road still labels what it covers. `confidence` = winning class score / (winning + runner-up
class score), 1.0 when every camera agrees; `n_supporting_cameras` /
`n_disagreeing_cameras` count the cameras voting the winning / another class.

**Visibility.** A point must have nothing nearer around its pixel, from two
sources: perception_2d's depth map, whose surfaces only count up to
`depth_map_max_range_m` (10 m) because its per-frame fit is off by metres
beyond ~15 m; and the sweep's own non-ground LiDAR returns, z-buffered in
8 px cells, which are exact at any range but sparse. On the WATO rig this
labels 77% of in-mask points instead of 57%, mostly cars at 20–50 m.

## Run

`ingest`, `lidar_preprocessing` and `perception_2d` must have run first.

```bash
./watod run semantic_lifting --bag <bag_id> [--chunk <chunk_id>] [--force]
```

Chunks whose `lifted_stats.parquet` exists are skipped unless `--force`. Each
chunk gets `manifest_semantic_lifting.json` (wato_common.manifest): config,
build provenance, and hashed inputs — the per-sweep world NPZs, masks and
depth maps through their producers' manifests and indexes. A full WATO chunk
(~1,900 sweeps × 8 cameras) takes ~3 min on CPU.

```bash
./watod run semantic_lifting viz --bag <bag_id> [--chunk <chunk_id>] [--color class]
```

writes `chunks/<chunk_id>/viz/lifted_labels_<instance|class>.html`: labeled
points coloured by lifted instance (or class) over grey unlabeled context, in
the world frame, with a slider over the chunk's sweeps in time order. The page
is `wato_common.viz.point_cloud_html`, the same viewer as lidar_preprocessing's
`viz`.

## Inputs

Every cross-component artifact is read through `wato_common.artifacts`, which
owns its format:

| Artifact | Producer | Read through |
|---|---|---|
| `calibration.json` (`K`, `ego_T_cam`) | ingest | `wato_common.artifacts.calibration` |
| `frame_index.parquet`, `poses.parquet` | ingest | `io.load_sweep_frames`: ingest's sweep ↔ camera pairing, camera pose at the camera's own timestamp (`wato_common.pose_lookup`) |
| `lidar_proc_index.parquet`, `lidar_proc/<sweep_id:06d>_world.npz` | lidar_preprocessing | `wato_common.artifacts.lidar_sweeps.ProcessedSweeps` |
| `tracklets_2d.parquet`, `masks_2d/` | perception_2d | `wato_common.artifacts.masklets.MaskletIndex` |
| `depth_2d/<cam>/<camera_seq:06d>.npz` | perception_2d | `wato_common.artifacts.depth` |

A camera frame with no depth artifact is skipped: perception_2d writes none
when its LiDAR-anchored depth fit failed. Depth `0` (fitted inverse depth ≤ 0)
is farther than the fit can represent and never occludes. Anything else
missing or malformed upstream (a world NPZ, a calibration entry, a mask PNG a
masklet lists) raises and stops the run.

## Outputs

Under `chunks/<chunk_id>/semantic_lifting/`, written through
`wato_common.artifacts.lifted_labels`:

| Artifact | Contents |
|---|---|
| `lifted_labels/<sweep_id:06d>.npz` | Per labeled point: `point_indices` (int32, into the sweep's world NPZ), `instance_ids` (the instance's smallest masklet_id) and `classes` (unicode string arrays — load without pickle), `confidences` (float32), `n_supporting_cameras`, `n_disagreeing_cameras` (int32). |
| `lifted_instances.parquet` | One `LiftedInstanceRow` per instance some point is labeled with: `cls`, its `masklet_ids` and `cam_ids` (to find its 2D masks), `n_points_labeled`, `n_sweeps`. Instance ids are chunk-local. |
| `lifted_stats.parquet` | One `LiftedStatsRow` per sweep (`sweep_id` int64). Written last: it is the chunk's completion marker. |

Labels written before 2026-10-11 (pickled object arrays, and empty) are rejected
by the reader; re-run semantic_lifting.

## Config

`config/semantic_lifting.yaml`: `visibility.*` (Eq. 1 τ, the depth-map patch
and its trusted range, the LiDAR occluder's cells and range tolerance — each
commented with its measurement), `instances.*` (the merge thresholds above).
The sweep ↔ camera window is ingest's
`max_cam_offset_ms`.

## Not done yet

- Dynamic points get no special handling: a moving object's points project
  where it was when the LiDAR hit it, up to ~1 m from where the camera saw it.
  Measured and deferred (`docs/semantic_lifting_measurements.md`).
- A sweep's world NPZ must carry `ground_mask` (ground never occludes); a sweep
  without one stops the run.
