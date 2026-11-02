# proposal_generation

Multimodal proposal generation (LiDAR detectors, Segment-Lift-Fit, fusion).

See `wato_world/README.md` and the architecture doc for context.

## Inputs from lidar_preprocessing

Besides the world-frame sweeps, lidar_preprocessing's Step C writes
`ground.npz`: read it through `wato_common.artifacts.ground` (`GroundGrid`'s
`height_at`, `height_above`, `normal_at`), which raises
`GroundUnavailableError` for a chunk Step C had no ground for.

Step F writes `motion_clusters.parquet`: class-less, recall-oriented
moving-object proposals with soft features (`motion_score`, `track_life`,
`n_sources`, `frac_seg_dynamic`, `frac_persistent`). Its box columns use
`ProposalRow`'s names, so a row maps onto a proposal with
`provenance="lidar_mos"`; false positives are expected, so gate on those
features (see `docs/research/lidar_mos_guidance.md`) and take the class from
semantic_lifting's labels. Read both proposal artifacts, and decode
`source_bits`, through `wato_common.artifacts.motion_proposals` —
lidar_preprocessing's package isn't installed in this image.

## Output

`proposals.parquet`, written through `wato_common.artifacts.proposals`, one
`ProposalRow` per object per frame_sync frame:

- `sweep_id` is the frame's reference sweep (the canonical LiDAR's), `sweep_ids`
  all its sweeps. `frame_id` is None for a sweep frame_sync matched to no
  canonical sweep; a Step F cluster's `-1` maps to None.
- Every source writes its own rows (`lidar_mos`, `slf`, `lidar_detector`), and
  fusion writes `fused` rows listing them in `member_proposal_ids`. tracking
  reads the fused rows; the source rows are kept to measure each source.
- `cls` is perception_2d's class (`car`, `truck`, `bus`, `motorcycle`,
  `bicycle`, `pedestrian` under the fixed taxonomy), or None when no lifted
  instance supports the box.
