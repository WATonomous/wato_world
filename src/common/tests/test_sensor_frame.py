"""close_to_sensor: the nuScenes devkit's remove_close square."""

import numpy as np
import pytest

from wato_common.geometry import close_to_sensor


def test_selects_the_open_square_at_any_height():
    xyz = np.array(
        [
            [0.5, 0.2, -5.0],  # inside, far below
            [-0.99, 0.99, 3.0],  # inside corner
            [1.0, 0.0, 0.0],  # on the edge: kept (strict <)
            [0.0, -1.0, 0.0],  # on the edge
            [2.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
        ]
    )
    assert close_to_sensor(xyz, 1.0).tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize("r", [0.0, -1.0])
def test_non_positive_half_side_selects_nothing(r):
    assert not close_to_sensor(np.zeros((3, 3)), r).any()


def test_rejects_a_flat_array():
    with pytest.raises(ValueError):
        close_to_sensor(np.zeros(3), 1.0)
