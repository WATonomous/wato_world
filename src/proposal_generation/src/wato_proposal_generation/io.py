"""Artifact I/O for proposal_generation (stub).

Reads world sweeps (wato_common.artifacts.lidar_sweeps), lifted labels
(wato_common.artifacts.lifted_labels), motion clusters
(wato_common.artifacts.motion_proposals) and calibration
(wato_common.artifacts.calibration).
Writes proposals.parquet (artifact_store.proposals_path).
Read another component's artifact only through its wato_common.artifacts
module, which owns the format.
"""
