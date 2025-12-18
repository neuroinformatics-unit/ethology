"""Utility functions for reshaping outputs of ensembles of detectors."""

import numpy as np


def _get_padding_width(array, final_first_dim):
    """Get pad_width to pad the end of an array along the first dimension."""
    # Throw an error if shape mismatch
    if array.shape[0] > final_first_dim:
        raise ValueError(
            "Array has more rows than the requested padded size: "
            f"{array.shape[0]} > {final_first_dim}"
        )
    pad_width = array.ndim * [(0, 0)]
    pad_width[0] = (0, final_first_dim - array.shape[0])
    return pad_width


def _pad_to_max_first_dimension(list_arrays, fill_value=np.nan):
    """Pad arrays in list to maximum size of their first dimension."""
    max_first_dimension = max(array.shape[0] for array in list_arrays)

    # Check for dtype compatibility between fill_value and arrays
    # (convert fill_value to numpy scalar/array to get its dtype first)
    for i, arr in enumerate(list_arrays):
        if not np.can_cast(np.asarray(fill_value).dtype, arr.dtype):
            raise TypeError(
                f"Cannot pad array (index {i}, dtype={arr.dtype}) "
                f"with fill_value={fill_value!r} "
                f"(type={type(fill_value).__name__}). "
                f"Ensure fill_value is compatible with array dtype."
            )

    list_arrays_padded = [
        np.pad(
            arr,
            _get_padding_width(arr, max_first_dimension),
            mode="constant",
            constant_values=fill_value,
        )
        for arr in list_arrays
    ]
    return list_arrays_padded


def centroid_shape_to_corners(position: np.ndarray, shape: np.ndarray):
    """Convert centroid and shape arrays to x1y1, x2y2 corner arrays.

    Parameters
    ----------
    position : numpy.ndarray
        Array of centroid coordinates with shape (..., 2, ...), where the
        second dimension contains (x, y) coordinates.
    shape : numpy.ndarray
        Array of bounding box dimensions with shape (..., 2, ...), where the
        second dimensioncontains (width, height).

    Returns
    -------
    x1y1 : numpy.ndarray
        Array of top-left corner coordinates with shape (..., 2,..), where the
        second dimension contains (x, y) coordinates. The top-left corner
        is the bounding box corner with minimum x and y coordinate values.
    x2y2 : numpy.ndarray
        Array of bottom-right corner coordinates with shape (..., 2,..), where
        the second dimension contains (x, y) coordinates. The bottom-right
        corner is the bounding box corner with maximum x and y coordinate
        values.


    Raises
    ------
    ValueError : If position and shape have different shapes or
    second dimension is not 2.

    See Also
    --------
    corners_to_centroid_shape : Inverse operation.

    """
    # Check position and shape have compatible shapes
    if position.shape != shape.shape:
        raise ValueError(
            f"position and shape must have the same shape, "
            f"got {position.shape} and {shape.shape}"
        )

    # Check dimension at index 1 is 2D
    if position.shape[1] != 2 or shape.shape[1] != 2:
        raise ValueError(
            "Dimension at index 1 must be 2"
            " for both position and shape, "
            f"but got position: {position.shape}, shape: {shape.shape}"
        )

    half_shape = shape / 2
    return (
        position - half_shape,  # x1y1
        position + half_shape,  # x2y2
    )


def corners_to_centroid_shape(x1y1: np.ndarray, x2y2: np.ndarray):
    """Convert x1y1, x2y2 corner arrays to centroid and shape arrays.

    Parameters
    ----------
    x1y1 : numpy.ndarray
        Array of top-left corner coordinates with shape (..., 2, ...), where
        the second dimension contains (x, y) coordinates. The top-left
        corner is the bounding box corner with minimum x and y coordinate
        values.
    x2y2 : numpy.ndarray
        Array of bottom-right corner coordinates with shape (..., 2, ...),
        where the second dimension contains (x, y) coordinates.
        The bottom-right corner is the bounding box corner with maximum x
        and y coordinate values.

    Returns
    -------
    centroid : numpy.ndarray
        Array of centroid coordinates with shape (..., 2, ...), where the
        second dimension contains (x, y) coordinates
    shape : numpy.ndarray
        Array of bounding box dimensions with shape (..., 2, ... ), where the
        second dimension contains (width, height).

    Raises
    ------
    ValueError : If x1y1 and x2y2 have different shapes or dimension
    at index 1 is not 2.

    See Also
    --------
    centroid_shape_to_corners : Inverse operation.

    """
    # Check x1y1 and x2y2 have compatible shapes
    if x1y1.shape != x2y2.shape:
        raise ValueError(
            f"x1y1 and x2y2 must have the same shape, "
            f"got x1y1: {x1y1.shape}, x2y2: {x2y2.shape}"
        )

    # Check dimension at index 1 is 2D
    if x1y1.shape[1] != 2 or x2y2.shape[1] != 2:
        raise ValueError(
            f"Dimension at index 1 must be 2 for both x1y1 and x2y2, "
            f"but got x1y1: {x1y1.shape}, x2y2: {x2y2.shape}"
        )

    return (
        0.5 * (x1y1 + x2y2),  # centroid
        x2y2 - x1y1,  # shape
    )
