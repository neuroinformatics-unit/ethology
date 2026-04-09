"""
Tests for save_bboxes.to_netcdf() and load_bboxes.from_netcdf().

These functions complete the netCDF4 I/O round trip for ethology
bbox annotation datasets.
"""
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from ethology.io.annotations.load_bboxes import from_files, from_netcdf
from ethology.io.annotations.save_bboxes import to_netcdf


@pytest.fixture(
    params=[
        "small_bboxes_COCO.json",
        "small_bboxes_VIA_subset.json",
    ]
)
def sample_dataset(request, annotations_test_data: dict):
    """
    Parametrised fixture that provides a loaded ethology Dataset
    from both COCO and VIA formats.
    """
    filename = request.param
    fmt: Literal["VIA", "COCO"] = (
        "VIA" if "VIA" in filename else "COCO"
    )
    file_path = annotations_test_data[filename]
    return from_files(file_path, format=fmt)


class TestToNetcdf:

    def test_creates_output_file(self, sample_dataset, tmp_path):
        """
        to_netcdf() must create a file at the specified path.
        """
        out = tmp_path / "output.nc"
        to_netcdf(sample_dataset, out)
        assert out.exists()

    def test_returns_path_object(self, sample_dataset, tmp_path):
        """
        to_netcdf() must return the output path as a Path object.
        """
        out = tmp_path / "output.nc"
        result = to_netcdf(sample_dataset, out)
        assert isinstance(result, Path)
        assert result == out

    def test_does_not_modify_caller_attrs(self, sample_dataset, tmp_path):
        """
        to_netcdf() must not convert dict attrs to strings in place.
        After the call the caller's Dataset must still have Python dicts.
        """
        original_cat_map = dict(sample_dataset.attrs["map_category_to_str"])
        original_img_map = dict(
            sample_dataset.attrs["map_image_id_to_filename"]
        )

        to_netcdf(sample_dataset, tmp_path / "output.nc")

        assert isinstance(
            sample_dataset.attrs["map_category_to_str"], dict
        ), "to_netcdf() mutated map_category_to_str in place"
        assert isinstance(
            sample_dataset.attrs["map_image_id_to_filename"], dict
        ), "to_netcdf() mutated map_image_id_to_filename in place"
        assert sample_dataset.attrs["map_category_to_str"] == original_cat_map
        assert (
            sample_dataset.attrs["map_image_id_to_filename"] == original_img_map
        )

    def test_dict_attrs_stored_as_json_strings_in_file(
        self, sample_dataset, tmp_path
    ):
        """
        In the raw netCDF4 file, dict attrs must be stored as JSON
        strings.

        This confirms the serialisation step happened.
        """
        out = tmp_path / "output.nc"
        to_netcdf(sample_dataset, out)

        # Read raw without from_netcdf() deserialisation
        raw_ds = xr.open_dataset(out)
        assert isinstance(
            raw_ds.attrs.get("map_category_to_str"), str
        ), (
            "map_category_to_str should be a JSON string in the raw "
            "netCDF4 file"
        )


class TestFromNetcdf:

    @pytest.fixture
    def saved_nc(self, sample_dataset, tmp_path):
        """
        Save sample_dataset to netCDF4 and return (path, original_ds).
        """
        out = tmp_path / "saved.nc"
        to_netcdf(sample_dataset, out)
        return out, sample_dataset

    def test_returns_xarray_dataset(self, saved_nc):
        nc_path, _ = saved_nc
        result = from_netcdf(nc_path)
        assert isinstance(result, xr.Dataset)

    def test_dimensions_match_original(self, saved_nc):
        nc_path, original = saved_nc
        loaded = from_netcdf(nc_path)
        assert dict(loaded.dims) == dict(original.dims)

    def test_position_values_match_original(self, saved_nc):
        """
        position array must be numerically identical after round-trip.

        equal_nan=True so that NaN padding slots compare as equal.
        """
        nc_path, original = saved_nc
        loaded = from_netcdf(nc_path)
        np.testing.assert_array_equal(
            loaded["position"].values,
            original["position"].values,
        )

    def test_category_values_match_original(self, saved_nc):
        nc_path, original = saved_nc
        loaded = from_netcdf(nc_path)
        if "category" in original.data_vars:
            np.testing.assert_array_equal(
                loaded["category"].values,
                original["category"].values,
            )

    def test_map_category_to_str_is_dict_after_load(self, saved_nc):
        """
        After from_netcdf(), map_category_to_str must be a dict.
        """
        nc_path, _ = saved_nc
        loaded = from_netcdf(nc_path)
        assert isinstance(loaded.attrs["map_category_to_str"], dict)

    def test_map_category_to_str_has_integer_keys(self, saved_nc):
        """
        json.loads() produces string keys ('1', '3').
        from_netcdf() must convert them back to integers (1, 3).
        """
        nc_path, _ = saved_nc
        loaded = from_netcdf(nc_path)
        for key in loaded.attrs["map_category_to_str"].keys():
            assert isinstance(key, int), (
                f"Expected int key in map_category_to_str, "
                f"got {type(key).__name__}: {key!r}"
            )

    def test_map_category_to_str_values_match_original(self, saved_nc):
        nc_path, original = saved_nc
        loaded = from_netcdf(nc_path)
        assert (
            loaded.attrs["map_category_to_str"]
            == original.attrs["map_category_to_str"]
        )

    def test_map_image_id_to_filename_has_integer_keys(self, saved_nc):
        nc_path, _ = saved_nc
        loaded = from_netcdf(nc_path)
        for key in loaded.attrs["map_image_id_to_filename"].keys():
            assert isinstance(key, int)

    def test_map_image_id_to_filename_matches_original(self, saved_nc):
        nc_path, original = saved_nc
        loaded = from_netcdf(nc_path)
        assert (
            loaded.attrs["map_image_id_to_filename"]
            == original.attrs["map_image_id_to_filename"]
        )

    def test_annotation_format_attr_preserved(self, saved_nc):
        nc_path, original = saved_nc
        loaded = from_netcdf(nc_path)
        assert (
            loaded.attrs.get("annotation_format")
            == original.attrs.get("annotation_format")
        )

    def test_space_coordinate_is_x_y(self, saved_nc):
        nc_path, _ = saved_nc
        loaded = from_netcdf(nc_path)
        assert list(loaded.coords["space"].values) == ["x", "y"]

    def test_file_not_found_raises_with_clear_message(self, tmp_path):
        missing = tmp_path / "does_not_exist.nc"
        with pytest.raises(FileNotFoundError, match="not found"):
            from_netcdf(missing)

    def test_accepts_string_path(self, saved_nc):
        """from_netcdf() must accept a plain string, not only Path."""
        nc_path, _ = saved_nc
        loaded = from_netcdf(str(nc_path))
        assert isinstance(loaded, xr.Dataset)


class TestFullRoundTrip:

    def test_all_data_variables_survive_round_trip(
        self, sample_dataset, tmp_path
    ):
        """
        Complete chain:
          from_files() → to_netcdf() → from_netcdf()
        All data variables must be numerically identical.
        """
        nc_path = tmp_path / "round_trip.nc"
        to_netcdf(sample_dataset, nc_path)
        reloaded = from_netcdf(nc_path)

        for var_name in sample_dataset.data_vars:
            np.testing.assert_array_equal(
                sample_dataset[var_name].values,
                reloaded[var_name].values,
                err_msg=(
                    f"Round-trip mismatch in data variable '{var_name}'"
                ),
            )

    def test_all_key_attrs_survive_round_trip(
        self, sample_dataset, tmp_path
    ):
        """
        All dict attributes must be identical after round-trip.
        """
        nc_path = tmp_path / "round_trip.nc"
        to_netcdf(sample_dataset, nc_path)
        reloaded = from_netcdf(nc_path)

        for key in [
            "map_category_to_str",
            "map_image_id_to_filename",
            "annotation_format",
        ]:
            assert (
                reloaded.attrs.get(key) == sample_dataset.attrs.get(key)
            ), f"Round-trip mismatch in attr '{key}'"

    def test_reloaded_dataset_equals_original(
        self, sample_dataset, tmp_path
    ):
        """
        xr.Dataset.equals() checks that all variables and coordinates
        are identical. This is the definitive round-trip assertion.
        """
        nc_path = tmp_path / "round_trip.nc"
        to_netcdf(sample_dataset, nc_path)
        reloaded = from_netcdf(nc_path)
        assert sample_dataset.equals(reloaded)