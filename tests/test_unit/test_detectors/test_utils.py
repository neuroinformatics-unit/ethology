from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from ethology.detectors.utils import (
    _get_padding_width,
    _pad_to_max_first_dimension,
    centroid_shape_to_corners,
    corners_to_centroid_shape,
)


@pytest.mark.parametrize(
    "array, final_first_dim, expected_exception",
    [
        (np.zeros(3), 5, does_not_raise()),
        (
            np.zeros(100),
            5,
            pytest.raises(
                ValueError,
                match="more rows than the requested padded size",
            ),
        ),
        (np.zeros((2, 2)), 5, does_not_raise()),
        (np.zeros((2, 2, 3)), 5, does_not_raise()),
    ],
)
def test_get_padding_width(array, final_first_dim, expected_exception):
    """Test the computation of the width to pad along the first dimension."""
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
    "list_arrays",
    [
        [np.zeros((1, 1)), np.zeros((3, 1)), np.zeros((42, 1))],
    ],
)
def test_pad_to_max_first_dimension(list_arrays, fill_value):
    """Test the padding of a list of arrays to the max first dimension size."""
    # Pad input arrays with fill value
    list_arrays_padded = _pad_to_max_first_dimension(list_arrays, fill_value)

    # Check shapes
    max_first_dimension = max([x.shape[0] for x in list_arrays])
    assert all([x.shape[0] == max_first_dimension for x in list_arrays_padded])

    # Check fill value
    assert all(
        np.allclose(padded[orig.shape[0] :], fill_value, equal_nan=True)
        for orig, padded in zip(list_arrays, list_arrays_padded, strict=True)
        if padded[orig.shape[0] :].size > 0
    )


@pytest.mark.parametrize(
    "fill_value",
    [np.nan, 0.5],
)
def test_pad_to_max_first_dimension_dtype_mismatch(fill_value):
    """Test that TypeError is raised for incompatible fill_value dtype.

    We test a list of integer input arrays.
    """
    with pytest.raises(
        TypeError,
        match="Ensure fill_value is compatible with array dtype",
    ):
        list_int_arrays = [
            np.zeros((2, 2), dtype=int),
            np.zeros((3, 2), dtype=int),
        ]
        _pad_to_max_first_dimension(list_int_arrays, fill_value)


@pytest.mark.parametrize(
    "position, shape, expected_exception",
    [
        (
            np.zeros((2, 2)),
            np.ones((2, 2)),
            does_not_raise(),
        ),
        (
            np.zeros((2, 2)),
            np.ones((1, 2)),
            pytest.raises(
                ValueError, match="position and shape must have the same shape"
            ),
        ),
        (
            np.zeros((2, 3)),
            np.ones((2, 3)),
            pytest.raises(
                ValueError, match="position and shape last dimension must be 2"
            ),
        ),
    ],
)
def test_centroid_shape_to_corners(position, shape, expected_exception):
    """Test conversion of centroid and shape to x1y1, x2y2 corner arrays."""
    with expected_exception as excinfo:
        x1y1, x2y2 = centroid_shape_to_corners(position, shape)

        if not excinfo:
            # Check values
            assert np.allclose(x1y1, np.minimum(x1y1, x2y2))
            assert np.allclose(x2y2, np.maximum(x1y1, x2y2))
            assert np.allclose(x1y1, position - shape / 2)
            assert np.allclose(x2y2, position + shape / 2)


@pytest.mark.parametrize(
    "x1y1, x2y2, expected_exception",
    [
        (
            np.zeros((2, 2)),
            np.ones((2, 2)),
            does_not_raise(),
        ),
        (
            np.zeros((2, 2)),
            np.ones((1, 2)),
            pytest.raises(
                ValueError, match="x1y1 and x2y2 must have the same shape"
            ),
        ),
        (
            np.zeros((2, 3)),
            np.ones((2, 3)),
            pytest.raises(
                ValueError, match="x1y1 and x2y2 last dimension must be 2"
            ),
        ),
    ],
)
def test_corners_to_centroid_shape(x1y1, x2y2, expected_exception):
    """Test conversion of x1y1, x2y2 arrays to centroid and shape arrays."""
    with expected_exception as excinfo:
        centroid, shape = corners_to_centroid_shape(x1y1, x2y2)

        if not excinfo:
            # Check values
            assert np.allclose(centroid, 0.5 * (x1y1 + x2y2))
            assert np.allclose(shape, x2y2 - x1y1)
