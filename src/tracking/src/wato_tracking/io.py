"""Artifact I/O for tracking (stub).

Reads proposals, masklets (wato_common.artifacts.masklets) and world sweeps
(wato_common.artifacts.lidar_sweeps).
Writes tracks.parquet (artifact_store.tracks_path).
Read another component's artifact only through its wato_common.artifacts
module, which owns the format.
"""
