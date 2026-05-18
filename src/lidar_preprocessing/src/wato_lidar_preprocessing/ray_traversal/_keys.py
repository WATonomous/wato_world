"""Voxel-key packing constants shared by both DDA kernels.

Mirrors voxel.py's AXIS_BITS so the Numba kernel, the Python parity kernel,
and the rest of the pipeline all agree on the int64 key encoding.
"""

AXIS_BITS = 20
AXIS_RANGE = 1 << AXIS_BITS  # 1_048_576
SHIFT_X = 2 * AXIS_BITS       # 40
SHIFT_Y = AXIS_BITS            # 20
