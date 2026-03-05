"""Load tracked bounding box datasets into ``ethology`` format.

This module provides utilities to construct an ``ethology`` bounding box
tracks dataset from an existing ``movement`` bounding boxes dataset.
"""

from collections.abc import Iterable

import numpy as np
import xarray as xr

from ethology.validators.detections import ValidBboxTracksDataset
from ethology.validators.utils import _check_output


def _require_dims(dataset: xr.Dataset, required_dims: Iterable[str]) -> None:
    """Check required dimensions exist; raise ValueError if not."""
    missing = set(required_dims) - set(dataset.dims)
    if missing:
        raise ValueError(
            "Expected a movement-like dataset with dimensions "
            f"{sorted(required_dims)}, but missing {sorted(missing)}."
        )


def _require_vars(dataset: xr.Dataset, required_vars: Iterable[str]) -> None:
    """Check required data variables exist; raise ValueError if not."""
    missing = set(required_vars) - set(dataset.data_vars)
    if missing:
        raise ValueError(
            "Expected a movement-like dataset with data variables "
            f"{sorted(required_vars)}, but missing {sorted(missing)}."
        )


@_check_output(ValidBboxTracksDataset)
def from_movement_bboxes(movement_ds: xr.Dataset) -> xr.Dataset:
    """Create a bounding box tracks dataset from a ``movement`` dataset.

    Parameters
    ----------
    movement_ds : xarray.Dataset
        Input bounding boxes dataset in ``movement`` format. It is expected
        to have at least the following:

        - dimensions: ``time``, ``space``, ``individuals``,
        - data variables: ``position`` and ``shape``.

        Optionally, it may contain:

        - ``category``: (time, individuals),
        - ``confidence``: (time, individuals).

    Returns
    -------
    xarray.Dataset
        A valid ``ethology`` bounding box tracks dataset with dimensions
        ``image_id``, ``space`` and ``id`` and data variables
        ``position``, ``shape``, ``category`` and ``confidence``. The
        dataset is validated using
        :class:`ethology.validators.detections.ValidBboxTracksDataset`.

    Raises
    ------
    ValueError
        If the input dataset does not contain the expected dimensions
        or data variables.

    Notes
    -----
    The conversion is purely structural: the function renames the
    ``movement`` dimensions ``time`` → ``image_id`` and
    ``individuals`` → ``id`` and forwards the core variables and
    attributes. If ``category`` or ``confidence`` are missing, they
    are created with default values (``-1`` for category, indicating
    unknown category, and ``NaN`` for confidence).

    """
    _require_dims(movement_ds, {"time", "space", "individuals"})
    _require_vars(movement_ds, {"position", "shape"})

    ds = movement_ds.rename({"time": "image_id", "individuals": "id"})
    out = xr.Dataset(
        data_vars={
            "position": ds["position"],
            "shape": ds["shape"],
        },
        coords={
            "image_id": ds.coords.get(
                "image_id", ds["position"].coords["image_id"]
            ),
            "space": ds.coords.get("space", ds["position"].coords["space"]),
            "id": ds.coords.get("id", ds["position"].coords["id"]),
        },
        attrs=dict(ds.attrs),
    )

    n_images = out.sizes["image_id"]
    n_ids = out.sizes["id"]

    if "category" in ds.data_vars:
        out["category"] = ds["category"]
    else:
        out["category"] = xr.DataArray(
            np.full((n_images, n_ids), -1, dtype=int),
            dims=("image_id", "id"),
        )

    if "confidence" in ds.data_vars:
        out["confidence"] = ds["confidence"]
    else:
        out["confidence"] = xr.DataArray(
            np.full((n_images, n_ids), np.nan, dtype=float),
            dims=("image_id", "id"),
        )

    return out
