import numpy as np
import pytest

from ethology.detectors.ensembles.utils import (
    _centroid_shape_to_corners,
    _corners_to_centroid_shape,
    _get_padding_width,
    _pad_to_max_first_dimension,
)


def test_get_padding_width():
    pass


@pytest.mark.parametrize(
    "fill_value",
    [
        np.nan,
        np.inf,
        42
    ],
)
def test_pad_to_max_first_dimension(fill_value):
    """Test padding all arrays in list along first dimension."""
    # Get max array length
    list_arrays = [np.zeros((1, 2, 3)), np.zeros((10, 2, 3))]
    max_array_length = max([arr.shape[0] for arr in list_arrays])

    # Pad
    list_arrays_padded = _pad_to_max_first_dimension(list_arrays, fill_value)

    # Assert all same length
    assert all(
        [arr.shape[0] == max_array_length for arr in list_arrays_padded]
    )
    # Assert other dimensions stay the same
    assert all(
        [
            arr.shape[1:] == arr_input.shape[1:]
            for arr, arr_input in zip(
                list_arrays_padded, list_arrays, strict=True
            )
        ]
    )
    # Assert padding value
    assert all(
        [
            np.allclose(
                arr[arr_input.shape[0]:],
                np.full_like(arr[arr_input.shape[0]:], fill_value),
                equal_nan=True,
            )
            for arr, arr_input in zip(
                list_arrays_padded, list_arrays, strict=True
            )
        ]
    )


def test_centroid_shape_to_corners():
    pass


def test_corners_to_centroid_shape():
    pass
