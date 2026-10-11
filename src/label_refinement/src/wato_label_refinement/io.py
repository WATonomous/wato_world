"""Artifact I/O for label_refinement (stub).

Reads tracks, world sweeps (wato_common.artifacts.lidar_sweeps) and motion
proposals (wato_common.artifacts.motion_proposals).
Writes refined_labels.parquet (artifact_store.refined_labels_path).
Read another component's artifact only through its wato_common.artifacts
module, which owns the format.
"""
