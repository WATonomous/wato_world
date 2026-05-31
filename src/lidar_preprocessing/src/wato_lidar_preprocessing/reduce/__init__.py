"""Step D — Global static map reduce."""

from ._core import _voxel_snap_downsample, reduce_ground_map, reduce_static_map

__all__ = ["reduce_static_map", "reduce_ground_map", "_voxel_snap_downsample"]
