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


def _centroid_shape_to_corners(position, shape):
    """Convert centroid and shape arrays to x1y1, x2y2 corner arrays.

    x1y1 is the top left corner (min x-coordinate, min y-coordinate),
    x2y2 is the bottom right corner (max x-coordinate, max y-coordinate)
    of the bounding box.

    Space dimension is assumed to be the second dimension.
    """
    half_shape = shape / 2
    return (
        position - half_shape,  # x1y1
        position + half_shape,  # x2y2
    )


def _corners_to_centroid_shape(x1y1, x2y2):
    """Convert x1y1, x2y2 corner arrays to centroid and shape arrays.

    Space dimension is assumed to be the second dimension.
    """
    return (
        0.5 * (x1y1 + x2y2),  # centroid
        x2y2 - x1y1,  # shape
    )
