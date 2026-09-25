# proposal_generation

Multimodal proposal generation (LiDAR detectors, Segment-Lift-Fit, fusion).

See `wato_world/README.md` and the architecture doc for context.

## Inputs from lidar_preprocessing

Besides the world-frame sweeps and `ground.npz`, lidar_preprocessing's Step F
writes `motion_clusters.parquet`: class-less, recall-oriented moving-object
proposals with soft features (`motion_score`, `track_life`, `n_sources`,
`frac_seg_dynamic`, `frac_persistent`). Its box columns use `ProposalRow`'s
names, so a row maps onto a proposal with `provenance="lidar_mos"`; false
positives are expected, so gate on those features (see
`docs/research/lidar_mos_guidance.md`) and take the class from
semantic_lifting's labels.
