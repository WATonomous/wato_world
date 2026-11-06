"""Integer mask-box overlap must retain the author's VOT threshold decisions."""

import pytest

from wato_perception_2d._vendor.d4sm.rectangle_overlap import mask_box_overlap


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([0, 0, 3, 3], [0, 0, 3, 3], 1),
        ([0, 0, 3, 3], [3, 0, 3, 3], 0),
        ([0, 0, 7, 10], [0, 0, 10, 10], 0.7),
        ([0, 0, 1, 1], [0, 5, 1, 1], 1),  # VOT's thin union branch
        ([0, 0, 1, 1], [5, 0, 1, 1], 1),
        ([0, 0, 1, 1], [5, 5, 1, 1], 0),
    ],
)
def test_overlap(a, b, expected):
    assert mask_box_overlap(a, b) == expected
