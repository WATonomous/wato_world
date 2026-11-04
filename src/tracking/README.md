# tracking

Tracks over proposal_generation's fused boxes, one chunk at a time, linked
across the spans neighbouring chunks share into the bag's tracks, and what
each track is: its class, and whether it is parked or moving. CPU only.
DINOv2 ReID comes later. Measurements: `docs/tracking_measurements.md`.

## How a track is made

Per chunk, every frame with fused rows (`proposals.fused`) is one tracker
step; then the chunks' tracks are linked and judged:

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
- **Linking across chunks** (below): a chunk's track and the next chunk's
  that agree in their shared span are one track of the bag.
- **Motion state**, over the whole linked track, one frame per instant:
  moving iff the Kalman-filtered centre moved ≥ 1 m first → last (3DAL's and
  MS3D's threshold), or ≥ 0.5 m at ≥ 0.5 m/s on average (a fragment too short
  to cover 1 m); `unknown` under 7 frames. Not MS3D's centre variance, which
  reads a one-sided parked car as moving.
- **Class**: the score-weighted vote over its frames' classes.
- **Filter**: a track is dropped only if it is both short (< 5 frames) and
  weak (no strong frame).

lidar_preprocessing's `motion_clusters.parquet` `track_hint_id` is chunk-local
and geometry-only; tracking doesn't use it. Step F's evidence reaches a track
through its fused rows (`frac_lidar_mos`).

## Run

```bash
./watod run tracking --bag <bag_id> [--force] \
    [--config /ws/src/tracking/config/tracking.yaml]
python /ws/src/tracking/scripts/eval_tracks.py <bag_id> [--canonical-lidar lidar_cc]
```

proposal_generation must have run on every chunk. The whole bag is tracked at
once (tracks are linked across chunks, so no chunk is tracked alone); it is
skipped if its tracks exist unless `--force`. Each chunk gets
`manifest_tracking.json`. `eval_tracks.py` scores against `gt_boxes.parquet`,
each labelled instant once, in the chunk that owns it: frame recall /
precision, true and false tracks, fragmentation, purity (tracks switching
objects), seams (objects crossing a chunk seam kept in one track), and the
parked / moving confusion.

## Output

Bag-level, through `wato_common.artifacts.tracks`:

- `tracks.parquet`: one `TrackRow` per (track, chunk, frame): the fused
  proposal it took in that chunk's frame, its box as proposed, the track's
  class on every row, and `owned`. `track_id` is `<chunk_id>-<n>`, named after
  the track's first chunk. A linked track has rows from each of its chunks, so
  an instant of a shared span can have two rows, one per chunk; `owned` marks
  the one its chunk labels (`tracks.per_instant` keeps one per instant).
- `track_summaries.parquet`: one `TrackSummaryRow` per track, judged over one
  frame per instant: motion state, class, and the evidence behind them
  (instants, strong frames, two-source and lidar_mos shares, net displacement,
  centre variance, points per frame).

Configuration: `config/tracking.yaml`; every field is documented in
[`config.py`](src/wato_tracking/config.py).

## Linking across chunks

Every id upstream is chunk-local: `sweep_id`, `camera_seq`, masklet ids, lifted
instance ids and `track_hint_id` all restart in each chunk, so an object
crossing a chunk seam is tracked once per chunk, with nothing in common but
time.

- **The shared span.** Neighbouring chunks k and k+1 both cover
  `[t_overlap_start_ns(k+1), t_overlap_end_ns(k)]` from `chunks/index.parquet`:
  3 × `chunk_overlap_seconds`, 6 s at ingest's defaults
  (`wato_common.chunk_spans`, which reads the bounds from the index). Both
  decode the same sweeps there: their world points are identical, and most
  boxes on them are too (`docs/tracking_measurements.md`).
- **Linking** (`stitch.py`, SUSHI's shared-frame link). A track of chunk k and
  one of chunk k+1 agree at an instant both saw in the span when their box
  centres are within 1 m. They are linked when they agree at ≥ 3 instants and
  at ≥ half the instants the shorter of the two saw in the span; one to one by
  the Hungarian method, maximising the instants agreed on, so a short
  fragment that agrees everywhere can't take the link from the object's long
  track. A chain of links is one track of the bag. Linking happens before the
  short-and-weak filter, which then judges the whole chain.
- **Ownership.** Each instant belongs to one chunk: chunk k before the midpoint
  `(t_end_ns(k) + t_start_ns(k+1)) / 2`, chunk k+1 from it. An owned instant
  is then at least half the span (3 s) from its chunk's nearest data edge (bag
  start and end aside), clear of the edge effects: SAM2 tracks starting at a
  detector keyframe, Step B's thinner evidence near a window edge, Step F's
  short `track_life` at a chunk start. Consumers read owned rows, so no
  instant is labelled twice; a track with no owned row (only the non-owning
  chunk saw it) is dropped.
- **Joining other data across chunks.** Join on time: a sweep by
  `(lidar_id, reference_timestamp_ns)`, an image by
  `(cam_id, camera_timestamp_ns)`, a proposal by its `reference_timestamp_ns`.
