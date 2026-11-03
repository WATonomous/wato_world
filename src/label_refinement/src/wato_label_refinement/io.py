"""Artifact I/O for label_refinement.

Reads tracks (wato_common.artifacts.tracks), world sweeps
(wato_common.artifacts.lidar_sweeps), the ground grid
(wato_common.artifacts.ground) and poses (wato_common.pose_lookup); writes
refined_labels.parquet through wato_common.artifacts.refined_labels. Read
another component's artifact only through its wato_common.artifacts module,
which owns the format.
"""
