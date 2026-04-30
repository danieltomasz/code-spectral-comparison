import numpy as np

from pesco.io import _remove_frauscher_zero_buffers


def test_remove_frauscher_zero_buffers_keeps_sixty_seconds_per_patient():
    fs = 2.0
    data = np.array(
        [
            [
                1, 2, 0, 4, 0, 0, 0, 0, 5, 6,
                7, 8, 9, 10, 11, 12, 0, 0, 0, 0,
            ],
            [
                2, 3, 0, 5, 0, 0, 0, 0, 6, 7,
                8, 9, 10, 11, 12, 13, 0, 0, 0, 0,
            ],
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        ],
        dtype=float,
    )
    patients = np.array([1, 1, 2])

    compact = _remove_frauscher_zero_buffers(
        data,
        patients,
        fs,
        duration=6.0,
        min_zero_run=2.0,
    )

    assert compact.shape == (3, 12)
    np.testing.assert_array_equal(
        compact[0],
        [1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    np.testing.assert_array_equal(
        compact[1],
        [2, 3, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    )
    np.testing.assert_array_equal(compact[2], np.arange(1, 13))
