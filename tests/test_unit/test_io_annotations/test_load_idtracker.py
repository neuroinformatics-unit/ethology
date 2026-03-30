"""Tests for ethology.io.annotations.load_idtracker."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ethology.io.annotations.load_idtracker import (
    _arrays_from_blobs,
    _arrays_from_trajectories,
    from_idtracker,
)

# Constants used across all tests
_N_FRAMES = 10
_N_ANIMALS = 3
_BBOX_SIZE = (40.0, 30.0)


# Fixtures
@pytest.fixture
def sample_trajectories() -> np.ndarray:
    """Return a synthetic trajectories array of shape (10, 3, 2).

    Animal 0: detected in every frame.
    Animal 1: detected in even frames only (NaN on odd frames).
    Animal 2: never detected (all NaN).
    """
    rng = np.random.default_rng(42)
    traj = rng.uniform(100.0, 900.0, size=(_N_FRAMES, _N_ANIMALS, 2))
    traj[1::2, 1, :] = np.nan  # animal 1 absent on odd frames
    traj[:, 2, :] = np.nan  # animal 2 always absent
    return traj


@pytest.fixture
def trajectories_file(tmp_path: Path, sample_trajectories: np.ndarray) -> Path:
    """Save sample_trajectories to a .npy file and return its path."""
    path = tmp_path / "trajectories.npy"
    np.save(path, sample_trajectories)
    return path


class _Blob:
    """Minimal picklable blob object matching the idtracker.ai interface."""

    def __init__(self, identity: int, bounding_box: list[float]):
        self.identity = identity
        self.bounding_box = bounding_box


class _BlobsCollection:
    """Minimal picklable blobs collection object."""

    def __init__(self, blobs_in_video: list[list[_Blob]]):
        self.blobs_in_video = blobs_in_video


@pytest.fixture
def sample_blobs_collection(
    sample_trajectories: np.ndarray,
) -> _BlobsCollection:
    """Return a picklable blobs collection whose bboxes match trajectories.

    For every detected animal in each frame the blob bounding box is
    centred at the trajectory centroid with a fixed size of (40, 30)
    pixels, i.e. [cx-20, cy-15, cx+20, cy+15].
    """
    blobs_in_video = []
    for frame_idx in range(_N_FRAMES):
        frame_blobs = []
        for animal_idx in range(_N_ANIMALS):
            centroid = sample_trajectories[frame_idx, animal_idx, :]
            if np.any(np.isnan(centroid)):
                continue
            cx, cy = float(centroid[0]), float(centroid[1])
            frame_blobs.append(
                _Blob(
                    identity=animal_idx + 1,
                    bounding_box=[cx - 20.0, cy - 15.0, cx + 20.0, cy + 15.0],
                )
            )
        blobs_in_video.append(frame_blobs)
    return _BlobsCollection(blobs_in_video)


@pytest.fixture
def blobs_collection_file(
    tmp_path: Path, sample_blobs_collection: _BlobsCollection
) -> Path:
    """Pickle sample_blobs_collection to disk and return its path."""
    path = tmp_path / "blobs_collection.pkl"
    with open(path, "wb") as fh:
        pickle.dump(sample_blobs_collection, fh)
    return path


# Tests: from_idtracker – trajectories + fixed bbox_size
class TestFromIdtrackerTrajectories:
    """Tests for from_idtracker when using trajectories + bbox_size."""

    def test_returns_xarray_dataset(self, trajectories_file: Path):
        """Output must be an xr.Dataset."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0, 2, 4],
            bbox_size=_BBOX_SIZE,
        )
        assert isinstance(ds, xr.Dataset)

    def test_required_data_vars_present(self, trajectories_file: Path):
        """position, shape and category must all be present."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        for var in ("position", "shape", "category"):
            assert var in ds.data_vars

    def test_required_dims_present(self, trajectories_file: Path):
        """image_id, space and id must all be dimensions."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        for dim in ("image_id", "space", "id"):
            assert dim in ds.dims

    def test_position_array_shape(self, trajectories_file: Path):
        """Position shape: (n_selected_frames, 2, n_animals)."""
        frame_indices = [0, 2, 4]
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=frame_indices,
            bbox_size=_BBOX_SIZE,
        )
        assert ds.position.shape == (len(frame_indices), 2, _N_ANIMALS)

    def test_shape_array_shape(self, trajectories_file: Path):
        """Shape array shape: (n_selected_frames, 2, n_animals)."""
        frame_indices = [0, 2, 4]
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=frame_indices,
            bbox_size=_BBOX_SIZE,
        )
        assert ds.shape.shape == (len(frame_indices), 2, _N_ANIMALS)

    def test_category_array_shape(self, trajectories_file: Path):
        """Category shape: (n_selected_frames, n_animals)."""
        frame_indices = [0, 2, 4]
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=frame_indices,
            bbox_size=_BBOX_SIZE,
        )
        assert ds.category.shape == (len(frame_indices), _N_ANIMALS)

    def test_space_coordinate_values(self, trajectories_file: Path):
        """Space coordinate must be ['x', 'y']."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        assert list(ds.coords["space"].values) == ["x", "y"]

    def test_image_id_coordinate_length(self, trajectories_file: Path):
        """image_id length must equal the number of unique selected frames."""
        frame_indices = [0, 2, 4]
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=frame_indices,
            bbox_size=_BBOX_SIZE,
        )
        assert len(ds.coords["image_id"]) == len(frame_indices)

    def test_id_coordinate_length(self, trajectories_file: Path):
        """Id coordinate length must equal n_animals."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        assert len(ds.coords["id"]) == _N_ANIMALS

    def test_nan_for_undetected_animal_in_frame(
        self,
        trajectories_file: Path,
        sample_trajectories: np.ndarray,
    ):
        """Animal not detected in a frame must produce NaN position/shape."""
        # Frame 1 is odd -> animal 1 (idx 1) is absent
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[1],
            bbox_size=_BBOX_SIZE,
        )
        assert np.isnan(ds.position.values[0, :, 1]).all()
        assert np.isnan(ds.shape.values[0, :, 1]).all()

    def test_minus_one_category_for_undetected_animal(
        self, trajectories_file: Path
    ):
        """Undetected animal must have category == -1."""
        # Animal 2 is never detected
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        assert ds.category.values[0, 2] == -1

    def test_detected_animal_position_equals_centroid(
        self,
        trajectories_file: Path,
        sample_trajectories: np.ndarray,
    ):
        """Detected animal position must equal the trajectory centroid."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        # Animal 0 is always detected
        assert ds.position.values[0, 0, 0] == pytest.approx(
            sample_trajectories[0, 0, 0]
        )
        assert ds.position.values[0, 1, 0] == pytest.approx(
            sample_trajectories[0, 0, 1]
        )

    def test_fixed_bbox_size_in_shape_array(self, trajectories_file: Path):
        """Shape array values must match the supplied bbox_size."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        # Animal 0 detected in frame 0
        assert ds.shape.values[0, 0, 0] == pytest.approx(_BBOX_SIZE[0])
        assert ds.shape.values[0, 1, 0] == pytest.approx(_BBOX_SIZE[1])

    def test_category_values_are_one_based(self, trajectories_file: Path):
        """Detected animal categories must be 1-based integers."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        assert ds.category.values[0, 0] == 1  # animal_idx=0 -> category=1

    def test_duplicate_frame_indices_removed(self, trajectories_file: Path):
        """Duplicate frame indices must be silently deduplicated."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0, 0, 2, 2, 4],
            bbox_size=_BBOX_SIZE,
        )
        assert len(ds.coords["image_id"]) == 3  # 0, 2, 4

    def test_image_id_to_filename_map(self, trajectories_file: Path):
        """map_image_id_to_filename must use frame_<index:06d>.png format."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[3, 7],
            bbox_size=_BBOX_SIZE,
        )
        assert ds.attrs["map_image_id_to_filename"] == {
            0: "frame_000003.png",
            1: "frame_000007.png",
        }

    def test_image_id_assigned_in_sorted_order(self, trajectories_file: Path):
        """Frames must be sorted before assigning image_ids."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[7, 3],  # deliberately unsorted
            bbox_size=_BBOX_SIZE,
        )
        # image_id 0 -> frame 3, image_id 1 -> frame 7
        assert ds.attrs["map_image_id_to_filename"][0] == "frame_000003.png"
        assert ds.attrs["map_image_id_to_filename"][1] == "frame_000007.png"

    def test_category_to_str_map(self, trajectories_file: Path):
        """map_category_to_str must cover all animals with 'animal_<n>'."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        expected = {i + 1: f"animal_{i + 1}" for i in range(_N_ANIMALS)}
        assert ds.attrs["map_category_to_str"] == expected

    def test_trajectories_file_attribute(self, trajectories_file: Path):
        """trajectories_file attr must equal the input path as string."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        assert ds.attrs["trajectories_file"] == str(trajectories_file)

    def test_blobs_collection_file_none_when_not_provided(
        self, trajectories_file: Path
    ):
        """blobs_collection_file attr must be None when blobs not given."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        assert ds.attrs["blobs_collection_file"] is None

    def test_images_dir_stored_in_attributes(
        self, trajectories_file: Path, tmp_path: Path
    ):
        """images_directory attr must match the supplied images_dir."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
            images_dir=tmp_path,
        )
        assert ds.attrs["images_directory"] == str(tmp_path)

    def test_images_dir_none_when_not_provided(self, trajectories_file: Path):
        """images_directory attr must be None when images_dir not given."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=_BBOX_SIZE,
        )
        assert ds.attrs["images_directory"] is None


# Tests: from_idtracker – blobs collection
class TestFromIdtrackerBlobsCollection:
    """Tests for from_idtracker when using a blobs collection."""

    def test_returns_xarray_dataset(
        self,
        trajectories_file: Path,
        blobs_collection_file: Path,
    ):
        """Output must be an xr.Dataset when blobs are used."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0, 2],
            blobs_collection_path=blobs_collection_file,
        )
        assert isinstance(ds, xr.Dataset)

    def test_position_from_blob_centroid(
        self,
        trajectories_file: Path,
        blobs_collection_file: Path,
        sample_trajectories: np.ndarray,
    ):
        """Position must equal the centroid implied by the blob bbox."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            blobs_collection_path=blobs_collection_file,
        )
        # Blob bbox centred at trajectory centroid -> position must match
        assert ds.position.values[0, 0, 0] == pytest.approx(
            sample_trajectories[0, 0, 0]
        )
        assert ds.position.values[0, 1, 0] == pytest.approx(
            sample_trajectories[0, 0, 1]
        )

    def test_shape_from_blob_bounding_box(
        self,
        trajectories_file: Path,
        blobs_collection_file: Path,
    ):
        """Shape values must reflect the actual blob bbox dimensions."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            blobs_collection_path=blobs_collection_file,
        )
        # Blob: x_min=cx-20, x_max=cx+20 -> width=40
        #       y_min=cy-15, y_max=cy+15 -> height=30
        assert ds.shape.values[0, 0, 0] == pytest.approx(40.0)
        assert ds.shape.values[0, 1, 0] == pytest.approx(30.0)

    def test_blobs_collection_file_attribute(
        self,
        trajectories_file: Path,
        blobs_collection_file: Path,
    ):
        """blobs_collection_file attr must equal the input path as string."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            blobs_collection_path=blobs_collection_file,
        )
        assert ds.attrs["blobs_collection_file"] == str(blobs_collection_file)

    def test_bbox_size_ignored_when_blobs_provided(
        self,
        trajectories_file: Path,
        blobs_collection_file: Path,
    ):
        """bbox_size must be ignored when a blobs collection is supplied."""
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            bbox_size=(999.0, 999.0),  # should be overridden by blobs
            blobs_collection_path=blobs_collection_file,
        )
        # Width must come from the blob (40), not from bbox_size (999)
        assert ds.shape.values[0, 0, 0] == pytest.approx(40.0)

    def test_nan_for_undetected_animal_via_blobs(
        self,
        trajectories_file: Path,
        blobs_collection_file: Path,
    ):
        """Animal with no blob in a frame must produce NaN position/shape."""
        # Animal 2 is never detected -> no blob exists for it
        ds = from_idtracker(
            trajectories_path=trajectories_file,
            frame_indices=[0],
            blobs_collection_path=blobs_collection_file,
        )
        assert np.isnan(ds.position.values[0, :, 2]).all()
        assert np.isnan(ds.shape.values[0, :, 2]).all()
        assert ds.category.values[0, 2] == -1


# Tests: from_idtracker – error cases
class TestFromIdtrackerErrors:
    """Tests that from_idtracker raises the correct errors."""

    def test_trajectories_file_not_found(self, tmp_path: Path):
        """FileNotFoundError when trajectories file is missing."""
        with pytest.raises(
            FileNotFoundError, match="Trajectories file not found"
        ):
            from_idtracker(
                trajectories_path=tmp_path / "missing.npy",
                frame_indices=[0],
                bbox_size=_BBOX_SIZE,
            )

    def test_blobs_file_not_found(
        self, trajectories_file: Path, tmp_path: Path
    ):
        """FileNotFoundError when blobs collection file is missing."""
        with pytest.raises(
            FileNotFoundError, match="Blobs collection file not found"
        ):
            from_idtracker(
                trajectories_path=trajectories_file,
                frame_indices=[0],
                blobs_collection_path=tmp_path / "missing.pkl",
            )

    def test_empty_frame_indices(self, trajectories_file: Path):
        """ValueError when frame_indices is an empty list."""
        with pytest.raises(
            ValueError, match="frame_indices must not be empty"
        ):
            from_idtracker(
                trajectories_path=trajectories_file,
                frame_indices=[],
                bbox_size=_BBOX_SIZE,
            )

    def test_negative_frame_index(self, trajectories_file: Path):
        """ValueError when a frame index is negative."""
        with pytest.raises(ValueError, match="non-negative"):
            from_idtracker(
                trajectories_path=trajectories_file,
                frame_indices=[-1, 0],
                bbox_size=_BBOX_SIZE,
            )

    def test_out_of_range_frame_index(self, trajectories_file: Path):
        """ValueError when a frame index exceeds trajectories length."""
        with pytest.raises(ValueError, match="out of range"):
            from_idtracker(
                trajectories_path=trajectories_file,
                frame_indices=[_N_FRAMES + 5],
                bbox_size=_BBOX_SIZE,
            )

    def test_no_bbox_size_and_no_blobs(self, trajectories_file: Path):
        """ValueError when neither bbox_size nor blobs path is given."""
        with pytest.raises(
            ValueError,
            match="Either bbox_size or blobs_collection_path",
        ):
            from_idtracker(
                trajectories_path=trajectories_file,
                frame_indices=[0],
            )

    def test_bbox_size_wrong_number_of_elements(self, trajectories_file: Path):
        """ValueError when bbox_size has the wrong number of elements."""
        with pytest.raises(ValueError, match="two elements"):
            from_idtracker(
                trajectories_path=trajectories_file,
                frame_indices=[0],
                bbox_size=(50.0,),  # type: ignore[arg-type]
            )

    def test_bbox_size_non_positive_width(self, trajectories_file: Path):
        """ValueError when bbox_size width is non-positive."""
        with pytest.raises(ValueError, match="positive"):
            from_idtracker(
                trajectories_path=trajectories_file,
                frame_indices=[0],
                bbox_size=(0.0, 20.0),
            )

    def test_bbox_size_non_positive_height(self, trajectories_file: Path):
        """ValueError when bbox_size height is non-positive."""
        with pytest.raises(ValueError, match="positive"):
            from_idtracker(
                trajectories_path=trajectories_file,
                frame_indices=[0],
                bbox_size=(20.0, -5.0),
            )

    def test_invalid_trajectories_ndim(self, tmp_path: Path):
        """ValueError when trajectories array is 2-D."""
        path = tmp_path / "bad.npy"
        np.save(path, np.zeros((_N_FRAMES, _N_ANIMALS)))
        with pytest.raises(ValueError, match="Expected trajectories array"):
            from_idtracker(
                trajectories_path=path,
                frame_indices=[0],
                bbox_size=_BBOX_SIZE,
            )

    def test_invalid_trajectories_last_dim(self, tmp_path: Path):
        """ValueError when last dim of trajectories array is not 2."""
        path = tmp_path / "bad.npy"
        np.save(path, np.zeros((_N_FRAMES, _N_ANIMALS, 3)))
        with pytest.raises(ValueError, match="Expected trajectories array"):
            from_idtracker(
                trajectories_path=path,
                frame_indices=[0],
                bbox_size=_BBOX_SIZE,
            )

    def test_blobs_missing_attribute(
        self, trajectories_file: Path, tmp_path: Path
    ):
        """ValueError when blobs object has no blobs_in_video attribute."""
        bad_path = tmp_path / "bad.pkl"
        with open(bad_path, "wb") as fh:
            pickle.dump({"wrong_key": []}, fh)
        with pytest.raises(ValueError, match="blobs_in_video"):
            from_idtracker(
                trajectories_path=trajectories_file,
                frame_indices=[0],
                blobs_collection_path=bad_path,
            )


# Tests: _arrays_from_trajectories
class TestArraysFromTrajectories:
    """Unit tests for the _arrays_from_trajectories private helper."""

    def test_output_shapes(self, sample_trajectories: np.ndarray):
        """All three output arrays must have the correct shapes."""
        frame_indices = [0, 2, 4]
        pos, shp, cat = _arrays_from_trajectories(
            sample_trajectories, frame_indices, _N_ANIMALS, _BBOX_SIZE
        )
        assert pos.shape == (3, 2, _N_ANIMALS)
        assert shp.shape == (3, 2, _N_ANIMALS)
        assert cat.shape == (3, _N_ANIMALS)

    def test_nan_for_always_undetected_animal(
        self, sample_trajectories: np.ndarray
    ):
        """Animal 2 (always NaN) must produce NaN position and shape."""
        pos, shp, cat = _arrays_from_trajectories(
            sample_trajectories, [0], _N_ANIMALS, _BBOX_SIZE
        )
        assert np.isnan(pos[0, :, 2]).all()
        assert np.isnan(shp[0, :, 2]).all()
        assert cat[0, 2] == -1

    def test_detected_animal_position_correct(
        self, sample_trajectories: np.ndarray
    ):
        """Detected animal position must equal the trajectory centroid."""
        pos, _, _ = _arrays_from_trajectories(
            sample_trajectories, [0], _N_ANIMALS, _BBOX_SIZE
        )
        assert pos[0, 0, 0] == pytest.approx(sample_trajectories[0, 0, 0])
        assert pos[0, 1, 0] == pytest.approx(sample_trajectories[0, 0, 1])

    def test_detected_animal_shape_equals_bbox_size(
        self, sample_trajectories: np.ndarray
    ):
        """Detected animal shape must equal the supplied bbox_size."""
        _, shp, _ = _arrays_from_trajectories(
            sample_trajectories, [0], _N_ANIMALS, _BBOX_SIZE
        )
        assert shp[0, 0, 0] == pytest.approx(_BBOX_SIZE[0])
        assert shp[0, 1, 0] == pytest.approx(_BBOX_SIZE[1])

    def test_detected_animal_category_is_one_based(
        self, sample_trajectories: np.ndarray
    ):
        """Detected animal category must equal animal_idx + 1."""
        _, _, cat = _arrays_from_trajectories(
            sample_trajectories, [0], _N_ANIMALS, _BBOX_SIZE
        )
        assert cat[0, 0] == 1

    def test_partially_detected_animal(self, sample_trajectories: np.ndarray):
        """Animal 1 must be detected on even frames and absent on odd ones."""
        # Even frame: detected
        _, _, cat_even = _arrays_from_trajectories(
            sample_trajectories, [0], _N_ANIMALS, _BBOX_SIZE
        )
        assert cat_even[0, 1] == 2  # 1-based

        # Odd frame: not detected
        pos_odd, _, cat_odd = _arrays_from_trajectories(
            sample_trajectories, [1], _N_ANIMALS, _BBOX_SIZE
        )
        assert np.isnan(pos_odd[0, :, 1]).all()
        assert cat_odd[0, 1] == -1


# Tests: _arrays_from_blobs
class TestArraysFromBlobs:
    """Unit tests for the _arrays_from_blobs private helper."""

    def test_output_shapes(self, blobs_collection_file: Path):
        """All three output arrays must have the correct shapes."""
        frame_indices = [0, 2]
        pos, shp, cat = _arrays_from_blobs(
            blobs_collection_file, frame_indices, _N_ANIMALS
        )
        assert pos.shape == (2, 2, _N_ANIMALS)
        assert shp.shape == (2, 2, _N_ANIMALS)
        assert cat.shape == (2, _N_ANIMALS)

    def test_position_equals_blob_centroid(
        self,
        blobs_collection_file: Path,
        sample_trajectories: np.ndarray,
    ):
        """Position must be the centre of the blob bounding box."""
        pos, _, _ = _arrays_from_blobs(blobs_collection_file, [0], _N_ANIMALS)
        assert pos[0, 0, 0] == pytest.approx(sample_trajectories[0, 0, 0])
        assert pos[0, 1, 0] == pytest.approx(sample_trajectories[0, 0, 1])

    def test_shape_equals_blob_bbox_dimensions(
        self, blobs_collection_file: Path
    ):
        """Shape values must equal the actual blob bbox width and height."""
        _, shp, _ = _arrays_from_blobs(blobs_collection_file, [0], _N_ANIMALS)
        assert shp[0, 0, 0] == pytest.approx(40.0)  # width
        assert shp[0, 1, 0] == pytest.approx(30.0)  # height

    def test_missing_attribute_raises_value_error(self, tmp_path: Path):
        """ValueError when the loaded object has no blobs_in_video attr."""
        bad_path = tmp_path / "bad.pkl"
        with open(bad_path, "wb") as fh:
            pickle.dump(object(), fh)
        with pytest.raises(ValueError, match="blobs_in_video"):
            _arrays_from_blobs(bad_path, [0], _N_ANIMALS)
