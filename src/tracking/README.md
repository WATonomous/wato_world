# tracking

Tracks over proposal_generation's fused boxes, one chunk at a time, and what
each track is: its class, and whether it is parked or moving. CPU only.
Stitching tracks across chunks and DINOv2 ReID come later. Measurements:
`docs/tracking_measurements.md`.

## How a track is made

Per chunk, every frame with fused rows (`proposals.fused`) is one tracker step:

- **Detections.** A fused row is a detection. It is *strong* when two sources
  agree on it (a lidar_mos + lifted pair) or its lifted box has confidence ≥
  0.5 and ≥ 20 points. Its lifted member's instance id (SAM2's identity across
  frames) is its key.
- **Association** is `wato_common.tracking`'s: the constant-velocity Kalman
  filter and Chen et al.'s cost with its gates, which Step F scores motion
  with, so the two stages agree on what "moved" means. Tracking adds three
  cues: strong boxes are matched before weak ones (SimpleTrack); a box with the
  track's last key costs less; and such a box is held only to the distance
  gate, since a one-sided box's IoU and volume swing as its visible face
  changes. A missed track coasts for 2 s; only matched frames are output.
- **Motion state**, over the whole track: moving iff the Kalman-filtered
  centre moved ≥ 1 m first → last (3DAL's and MS3D's threshold), or ≥ 0.5 m
  at ≥ 0.5 m/s on average (a fragment too short to cover 1 m); `unknown` under
  7 frames. Not MS3D's centre variance, which reads a one-sided parked car as
  moving.
- **Class**: the score-weighted vote over its frames' classes.
- **Filter**: a track is dropped only if it is both short (< 5 frames) and
  weak (no strong frame).

lidar_preprocessing's `motion_clusters.parquet` `track_hint_id` is chunk-local
and geometry-only; tracking doesn't use it. Step F's evidence reaches a track
through its fused rows (`frac_lidar_mos`).

## Run

```bash
./watod run tracking --bag <bag_id> [--chunk <chunk_id>] [--force] \
    [--config /ws/src/tracking/config/tracking.yaml]
python /ws/src/tracking/scripts/eval_tracks.py <bag_id> [--canonical-lidar lidar_cc]
```

proposal_generation must have run on every chunk. The bag's tracks are skipped
if they exist unless `--force`; `--chunk` re-tracks one chunk and keeps the
others'. Each chunk gets `manifest_tracking.json`. `eval_tracks.py` scores
against `gt_boxes.parquet`: frame recall / precision, true and false tracks,
fragmentation, purity (tracks switching objects), and the parked / moving
confusion.

## Output

Bag-level, through `wato_common.artifacts.tracks`:

- `tracks.parquet`: one `TrackRow` per (track, frame), the fused proposal it
  took, its box as proposed, the track's class on every row, and `owned`.
  `track_id` is `<chunk_id>-<n>` until tracks are stitched.
- `track_summaries.parquet`: one `TrackSummaryRow` per track: motion state,
  class, and the evidence behind them (frames, strong frames, two-source and
  lidar_mos shares, net displacement, centre variance, points per frame).

Configuration: `config/tracking.yaml`; every field is documented in
[`config.py`](src/wato_tracking/config.py).

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
