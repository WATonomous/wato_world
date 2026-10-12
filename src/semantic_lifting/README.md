# semantic_lifting

Occlusion-aware 2D → 3D label lifting: each LiDAR point takes the label of the
perception_2d masklet it projects into, if the image's metric depth says the
point is visible there (UniLiPs Eq. 1), with votes pooled across cameras.
Design and rationale: `docs/research/semantic_lifting_design.md`.

## Run

`ingest`, `lidar_preprocessing` and `perception_2d` must have run first.

```bash
./watod run semantic_lifting run --bag <bag_id>
./watod run semantic_lifting run --bag <bag_id> --chunk <chunk_id>
```

Chunks whose `lifted_stats.parquet` exists are skipped (the CLI has no
`--force` yet; delete the chunk's `semantic_lifting/` directory to redo it).

## Inputs

Every cross-component artifact is read through `wato_common.artifacts`, which
owns its format:

| Artifact | Producer | Read through |
|---|---|---|
| `calibration.json` (`K`, `ego_T_cam`) | ingest | `wato_common.artifacts.calibration` |
| `frame_index.parquet`, `poses.parquet` | ingest | `io.load_frame_refs` + `wato_common.pose_lookup` (camera pose at the camera's own timestamp) |
| `lidar_proc_index.parquet`, `lidar_proc/<sweep_id:06d>_world.npz` | lidar_preprocessing | `io.load_sweeps`, `io.load_all_lidar_points` |
| `tracklets_2d.parquet`, `masks_2d/` | perception_2d | `wato_common.artifacts.masklets.MaskletIndex` |
| `depth_2d/<cam>/<camera_seq:06d>.npz` | perception_2d | `wato_common.artifacts.depth` |

A camera frame with no depth artifact is skipped: perception_2d writes none
when its LiDAR-anchored depth fit failed.

## Outputs

Under `chunks/<chunk_id>/semantic_lifting/`, written through
`wato_common.artifacts.lifted_labels`:

| Artifact | Contents |
|---|---|
| `lifted_labels/<sweep_id:06d>.npz` | Per labeled point: `point_indices` (int32, into the sweep's world NPZ), `instance_ids` and `classes` (unicode string arrays — load without pickle), `confidences` (float32), `n_supporting_cameras`, `n_disagreeing_cameras` (int32). |
| `lifted_stats.parquet` | One `LiftedStatsRow` per sweep (`sweep_id` int64). Written last: it is the chunk's completion marker. |

Labels written before 2026-10-11 used unpadded `<sweep_id>.npz` names, a string
`sweep_id`, and pickled object arrays. Those files were also empty:
`frames_present` was misread, so no masks were found. The reader rejects them,
so re-run semantic_lifting.

## Config

`config/semantic_lifting.yaml`: `temporal.max_offset_s` (camera ↔ sweep
matching window), `visibility.tolerance_m` / `neighborhood` (Eq. 1 τ and the
depth patch), `voting.*`.

## Not done yet

- No manifest and no lockfile (CLAUDE.md "Reproducibility"): the image is not
  pinned yet.
- Dynamic points get no special handling yet (the sweep's `dynamic_mask_path` is
  carried but unused).
