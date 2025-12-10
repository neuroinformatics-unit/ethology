import numpy as np
import pytest

from ethology.detectors.ensembles.utils import (
    _centroid_shape_to_corners,
    _corners_to_centroid_shape,
    _get_padding_width,
    _pad_to_max_first_dimension,
)


@pytest.mark.parametrize(
    "array, target_first_dim, expected_pad_width_first_dim",
    [
        (
            np.zeros((3,)),
            5,
            (0, 2),
        ),  # 1D array
        (
            np.zeros((1, 2, 3)),
            4,
            (0, 3),
        ),  # 3D array
        (
            np.zeros((10, 2, 3)),
            10,
            (0, 0),
        ),  # No padding needed
    ],
)
def test_get_padding_width(
    array, target_first_dim, expected_pad_width_first_dim
):
    """Test getting padding width for arrays of different dimensions."""
    pad_width = _get_padding_width(array, target_first_dim)

    assert len(pad_width) == array.ndim
    assert pad_width[0] == expected_pad_width_first_dim
    assert all(pw == (0, 0) for pw in pad_width[1:])


@pytest.mark.parametrize(
    "fill_value",
    [np.nan, np.inf, 42],
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
                arr[arr_input.shape[0] :],
                np.full_like(arr[arr_input.shape[0] :], fill_value),
                equal_nan=True,
            )
            for arr, arr_input in zip(
                list_arrays_padded, list_arrays, strict=True
            )
        ]
    )


@pytest.mark.parametrize(
    "position, shape, expected_x1y1, expected_x2y2",
    [
        (
            np.zeros((1, 2)),
            np.array([[4, 2]]),
            np.array([[-2, -1]]),
            np.array([[2, 1]]),
        )
    ],
)
def test_centroid_shape_to_corners(
    position, shape, expected_x1y1, expected_x2y2
):
    x1y1, x2y2 = _centroid_shape_to_corners(position, shape)
    np.testing.assert_array_equal(x1y1, expected_x1y1)
    np.testing.assert_array_equal(x2y2, expected_x2y2)


@pytest.mark.parametrize(
    "x1y1, x2y2, expected_position, expected_shape",
    [
        (
            np.zeros((1, 2)),
            np.ones((1, 2)),
            np.array([[0.5, 0.5]]),
            np.array([[1, 1]]),
        )
    ],
)
def test_corners_to_centroid_shape(
    x1y1, x2y2, expected_position, expected_shape
):
    position, shape = _corners_to_centroid_shape(x1y1, x2y2)
    np.testing.assert_array_equal(position, expected_position)
    np.testing.assert_array_equal(shape, expected_shape)
