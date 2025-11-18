"""Utility functions for reshaping outputs of ensembles of detectors."""

import numpy as np


def get_padding_width(array, max_n):
    """Get pad width for array to max_n detections in the first dimension."""
    pad_width = array.ndim * [(0, 0)]
    pad_width[0] = (0, max_n - array.shape[0])  # before, after
    return pad_width


def pad_to_max_first_dimension(list_arrays, fill_value=np.nan):
    """Pad arrays to maximum number across all arrays in the first dimension."""
    max_n_detections = max(array.shape[0] for array in list_arrays)
    list_arrays_padded = [
        np.pad(
            arr,
            get_padding_width(arr, max_n_detections),
            mode="constant",
            constant_values=fill_value,
        )
        for arr in list_arrays
    ]
    return list_arrays_padded
