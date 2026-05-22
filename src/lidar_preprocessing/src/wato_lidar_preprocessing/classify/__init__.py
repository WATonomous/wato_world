"""Step B — voxel static/dynamic classification.

Public surface mirrors the old monolithic classify.py: external callers
(component pipeline.py, __main__.py, tests) keep their imports unchanged.
"""

from wato_lidar_preprocessing.voxel import VoxelOverflowError

from .pipeline import ClassifyResult, process_chunk

__all__ = ["process_chunk", "ClassifyResult", "VoxelOverflowError"]
