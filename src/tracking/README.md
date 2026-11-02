# tracking

4D tracking (3D Kalman filter, masklet association, DINOv2 ReID).

See `wato_world/README.md` and the architecture doc for context.

## Motion model and association

Build on `wato_common.tracking` rather than a new motion model: yaw-only box
fitting and 3D IoU (`boxes.py`), a constant-velocity Kalman filter
(`kalman.py`), and Chen et al.'s association cost with Hungarian assignment,
gating and a deactivated-track buffer (`association.py`). lidar_preprocessing's
Step F already scores cluster motion with these primitives, so the two stages
agree on what "moved" means. Add appearance (DINOv2 ReID) and class on top.

lidar_preprocessing's `motion_clusters.parquet` `track_hint_id` is chunk-local
and geometry-only — useful as a hint, never as an identity.

## Stitching across chunks

Every id upstream is chunk-local: `sweep_id`, `camera_seq`, masklet ids, lifted
instance ids and `track_hint_id` all restart in each chunk, so an object seen
across a chunk seam is a different object in each chunk until tracking links
them.

- **The shared span.** Neighbouring chunks k and k+1 both cover
  `[t_overlap_start_ns(k+1), t_overlap_end_ns(k)]` from `chunks/index.parquet`:
  3 × `chunk_overlap_seconds`, 6 s at ingest's defaults. Read the bounds from
  the index, not from constants. Each chunk's artifacts label the span
  independently.
- **Matching the same datum.** Its ids differ between the two chunks, so join
  on time: a sweep by `(lidar_id, reference_timestamp_ns)`, an image by
  `(cam_id, camera_timestamp_ns)`, a proposal by its `reference_timestamp_ns`.
  Link chunk k's tracks to chunk k+1's through what both saw in the span.
- **Ownership.** Each instant belongs to one chunk: chunk k before the midpoint
  `(t_end_ns(k) + t_start_ns(k+1)) / 2`, chunk k+1 from it. An owned instant
  is then at least half the span (3 s) from its chunk's nearest data edge (bag
  start and end aside), clear of the edge effects: SAM2 tracks starting at a
  detector keyframe, Step B's thinner evidence near a window edge, Step F's
  short `track_life` at a chunk start. tracking's output, and everything
  downstream of it (label_refinement, student_training), keeps only the
  owner's labels, so no instant is labelled twice.
- **Untested assumption.** Matching at the point level
  (`lifted_labels.point_indices`, motion proposals) needs a sweep decoded into
  two chunks to give the same world points in the same order in both. Each
  chunk deskews from its own `poses.parquet`, and nothing tests this yet.
