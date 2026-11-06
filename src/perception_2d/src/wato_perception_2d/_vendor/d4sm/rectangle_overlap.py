"""VOT rectangle overlap restricted to positive integer mask bounding boxes.

Equivalent to votchallenge/toolkit@21a34c67's raster.calculate_overlap in this
domain, including its one-pixel-wide union special case. Avoids installing the
VOT benchmark runner just to compare two boxes. See README.md.
"""


def mask_box_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if min(ax, ay, bx, by) < 0 or min(aw, ah, bw, bh) < 1:
        raise ValueError("Expected nonnegative integer mask boxes with positive size")
    if (
        max(ax + aw, bx + bw) - min(ax, bx) <= 1
        or max(ay + ah, by + bh) - min(ay, by) <= 1
    ):
        return 1.0
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection / (aw * ah + bw * bh - intersection)
