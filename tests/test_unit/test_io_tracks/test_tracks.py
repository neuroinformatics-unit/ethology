"""Tests for :mod:`ethology.io.tracks`."""

import numpy as np
import pytest
import xarray as xr

from ethology.io.tracks import from_movement_bboxes
from ethology.validators.detections import ValidBboxTracksDataset


def _make_movement_bbox_dataset(
    n_time=3,
    n_individuals=2,
    has_category=False,
    has_confidence=True,
) -> xr.Dataset:
    """Build a minimal movement-style bbox dataset for tests."""
    time = np.arange(n_time)
    individuals = np.arange(n_individuals)
    space = ["x", "y"]

    position = np.zeros((n_time, len(space), n_individuals))
    shape = np.ones_like(position)
    data_vars = {
        "position": (["time", "space", "individuals"], position),
        "shape": (["time", "space", "individuals"], shape),
    }
    if has_category:
        data_vars["category"] = (
            ["time", "individuals"],
            np.zeros((n_time, n_individuals), dtype=int),
        )
    if has_confidence:
        data_vars["confidence"] = (
            ["time", "individuals"],
            np.full((n_time, n_individuals), 0.5),
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={"time": time, "space": space, "individuals": individuals},
        attrs={"time_unit": "frames"},
    )


def test_from_movement_bboxes_converts_to_valid_tracks_dataset():
    """from_movement_bboxes output passes ValidBboxTracksDataset."""
    movement_ds = _make_movement_bbox_dataset()

    ds_tracks = from_movement_bboxes(movement_ds)

    ValidBboxTracksDataset(dataset=ds_tracks)
    assert set(ds_tracks.dims) >= {"image_id", "space", "id"}
    assert np.array_equal(
        ds_tracks.coords["image_id"].values, movement_ds.time
    )
    assert np.array_equal(
        ds_tracks.coords["id"].values, movement_ds.individuals
    )
    assert np.allclose(ds_tracks.position.values, movement_ds.position.values)
    assert np.allclose(ds_tracks.shape.values, movement_ds.shape.values)
    assert ds_tracks.confidence.shape == (3, 2)
    assert ds_tracks.category.shape == (3, 2)


def test_from_movement_bboxes_forwards_category_and_confidence_when_present():
    """When movement dataset has category, it is forwarded; confidence same."""
    movement_ds = _make_movement_bbox_dataset(has_category=True)

    ds_tracks = from_movement_bboxes(movement_ds)

    assert np.allclose(ds_tracks.category.values, movement_ds.category.values)
    assert np.allclose(
        ds_tracks.confidence.values, movement_ds.confidence.values
    )


def test_from_movement_bboxes_fills_missing_category_and_confidence():
    """Missing category/confidence are filled with -1 and NaN."""
    movement_ds = _make_movement_bbox_dataset(
        has_category=False, has_confidence=False
    )

    ds_tracks = from_movement_bboxes(movement_ds)

    assert (ds_tracks.category.values == -1).all()
    assert np.isnan(ds_tracks.confidence.values).all()


def test_from_movement_bboxes_raises_when_dim_missing():
    """from_movement_bboxes raises ValueError when required dims missing."""
    movement_ds = _make_movement_bbox_dataset()
    movement_ds = movement_ds.rename({"time": "frame"})

    with pytest.raises(ValueError) as excinfo:
        from_movement_bboxes(movement_ds)

    assert "dimensions" in str(excinfo.value)
    assert "time" in str(excinfo.value) or "individuals" in str(excinfo.value)


def test_from_movement_bboxes_raises_when_data_var_missing():
    """from_movement_bboxes raises ValueError when position/shape missing."""
    movement_ds = _make_movement_bbox_dataset()
    movement_ds = movement_ds.drop_vars("shape")

    with pytest.raises(ValueError) as excinfo:
        from_movement_bboxes(movement_ds)

    assert "data variables" in str(excinfo.value)
    assert "shape" in str(excinfo.value)
