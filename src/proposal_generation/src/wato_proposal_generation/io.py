"""Artifact I/O for proposal_generation (stub).

Reads world sweeps (wato_common.artifacts.lidar_sweeps), lifted labels
(wato_common.artifacts.lifted_labels), motion clusters
(wato_common.artifacts.motion_proposals), the ground grid
(wato_common.artifacts.ground) and calibration
(wato_common.artifacts.calibration).
Writes proposals.parquet through wato_common.artifacts.proposals.
Read another component's artifact only through its wato_common.artifacts
module, which owns the format.
"""
