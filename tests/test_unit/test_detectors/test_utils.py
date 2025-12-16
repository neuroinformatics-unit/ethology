from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from ethology.detectors.utils import (
    _get_padding_width,
    _pad_to_max_first_dimension,
)


@pytest.mark.parametrize(
    "array, final_first_dim, expected_exception",
    [
        (np.zeros(3), 5, does_not_raise()),
        (
            np.zeros(100),
            5,
            pytest.raises(
                ValueError, match="more rows than the requested padded size"
            ),
        ),
        (np.zeros((2, 2)), 5, does_not_raise()),
        (np.zeros((2, 2, 3)), 5, does_not_raise()),
    ],
)
def test_get_padding_width(array, final_first_dim, expected_exception):
    with expected_exception as excinfo:
        pad_width = _get_padding_width(array, final_first_dim)

        if not excinfo:
            expected_pad_width = array.ndim * [(0, 0)]
            expected_pad_width[0] = (0, final_first_dim - array.shape[0])
            assert pad_width == expected_pad_width


@pytest.mark.parametrize(
    "fill_value",
    [np.nan, 42, -1],
)
@pytest.mark.parametrize(
    "list_arrays", [[np.zeros((1, 1)), np.zeros((3, 1)), np.zeros((42, 1))]]
)
def test_pad_to_max_first_dimension(list_arrays, fill_value):
    list_arrays_padded = _pad_to_max_first_dimension(list_arrays, fill_value)

    # check padded arrays
    max_first_dimension = max([x.shape[0] for x in list_arrays])
    assert all([x.shape[0] == max_first_dimension for x in list_arrays_padded])

    # check fill value
    assert all(
        np.allclose(padded[orig.shape[0] :], fill_value, equal_nan=True)
        for orig, padded in zip(list_arrays, list_arrays_padded, strict=True)
        if padded[orig.shape[0] :].size > 0
    )


def test_centroid_shape_to_corners():
    pass


def test_corners_to_centroid_shape():
    pass
