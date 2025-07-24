"""Utility functions for transforming detection datasets."""


def add_bboxes_min_max_corners(ds):
    """Add xy_min and xy_max arrays to ds."""
    ds["xy_min"] = ds.position - 0.5 * ds.shape
    ds["xy_max"] = ds.position + 0.5 * ds.shape
    return ds
