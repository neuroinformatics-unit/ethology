"""Load bounding box annotations from idtracker.ai output files."""

import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from ethology.validators.annotations import ValidBboxAnnotationsDataset
from ethology.validators.utils import _check_output


@_check_output(ValidBboxAnnotationsDataset)
def from_idtracker(
    trajectories_path: Path | str,
    frame_indices: list[int],
    bbox_size: tuple[float, float] | None = None,
    blobs_collection_path: Path | str | None = None,
    images_dir: Path | str | None = None,
) -> xr.Dataset:
    """Generate a bounding box annotations dataset from idtracker.ai output.

    Creates an ``ethology`` bounding box annotations dataset from
    idtracker.ai trajectory data and, optionally, from blob detection
    data.  Each selected frame becomes one ``image_id``; each tracked
    animal becomes one ``id`` entry per frame.

    Parameters
    ----------
    trajectories_path : Path or str
        Path to the idtracker.ai trajectories file (``.npy``).
        The array must have shape ``(n_frames, n_animals, 2)`` where the
        last dimension holds ``(x, y)`` centroid coordinates in pixels.
        ``NaN`` values indicate that an animal was not detected in a
        given frame.
    frame_indices : list[int]
        Zero-based frame indices for which to generate bounding box
        annotations.  Duplicate indices are silently removed.  Every
        index must be non-negative and within the number of frames
        recorded in the trajectories file.
    bbox_size : tuple[float, float], optional
        Fixed bounding box size ``(width, height)`` in pixels applied to
        every detected animal.  Required when ``blobs_collection_path``
        is ``None``.  Ignored when ``blobs_collection_path`` is provided.
        Both values must be strictly positive.
    blobs_collection_path : Path or str, optional
        Path to the idtracker.ai blobs collection file (``.pkl``).  When
        provided, bounding boxes are extracted directly from the blob
        objects rather than computed from centroids and a fixed size.
        The pickled object must expose a ``.blobs_in_video`` attribute:
        a list (indexed by frame) of lists of blob objects, where each
        blob exposes:

        - ``.bounding_box``: sequence ``[x_min, y_min, x_max, y_max]``
          in pixel coordinates;
        - ``.identity``: 1-based integer animal identity (``0`` or
          ``None`` means the blob was not identified).

    images_dir : Path or str, optional
        Directory that contains the extracted video frames.  Stored in
        dataset attributes when provided but otherwise not used.

    Returns
    -------
    xarray.Dataset
        A valid ``ethology`` bounding box annotations dataset with
        dimensions ``image_id``, ``space``, ``id`` and data variables:

        - ``position`` (``image_id``, ``space``, ``id``): bbox centroid
          ``(x, y)`` in pixels.  ``NaN`` for undetected animals.
        - ``shape`` (``image_id``, ``space``, ``id``): bbox
          ``(width, height)`` in pixels.  ``NaN`` for undetected animals.
        - ``category`` (``image_id``, ``id``): 1-based integer animal
          identity.  ``-1`` for undetected animals.

        Dataset attributes:

        - ``trajectories_file``: path to the trajectories file.
        - ``blobs_collection_file``: path to the blobs file, or ``None``.
        - ``images_directory``: path to the images directory, or ``None``.
        - ``map_category_to_str``: mapping from 1-based animal ID to the
          string label ``"animal_<id>"``.
        - ``map_image_id_to_filename``: mapping from ``image_id`` to the
          canonical frame filename ``"frame_<frame_index:06d>.png"``.

    Raises
    ------
    FileNotFoundError
        If ``trajectories_path`` or ``blobs_collection_path`` does not
        exist on disk.
    ValueError
        If ``frame_indices`` is empty or contains negative values.
        If any frame index is out of range for the given trajectories.
        If neither ``bbox_size`` nor ``blobs_collection_path`` is
        provided.
        If ``bbox_size`` is provided but does not have exactly two
        elements or contains non-positive values.
        If the loaded trajectories array does not have shape
        ``(n_frames, n_animals, 2)``.
        If the blobs collection object does not have a
        ``.blobs_in_video`` attribute.

    Notes
    -----
    The ``image_id`` coordinate is assigned as the 0-based position of
    each frame in the **sorted, deduplicated** ``frame_indices`` list.
    The ``map_image_id_to_filename`` attribute maps each ``image_id`` to
    a canonical frame filename ``"frame_{frame_index:06d}.png"``.

    The ``id`` coordinate ranges from ``0`` to ``n_animals - 1`` and
    corresponds directly to the column index in the trajectories array.
    For the blobs case the animal index is derived from the 1-based blob
    identity as ``identity - 1``.

    Examples
    --------
    Generate annotations with a fixed bounding box size:

    >>> import numpy as np
    >>> from ethology.io.annotations.load_idtracker import from_idtracker
    >>> ds = from_idtracker(
    ...     trajectories_path="path/to/trajectories.npy",
    ...     frame_indices=[0, 10, 20],
    ...     bbox_size=(50.0, 50.0),
    ... )
    >>> print(ds.position.shape)  # (3, 2, n_animals)

    Generate annotations from a blobs collection:

    >>> ds = from_idtracker(
    ...     trajectories_path="path/to/trajectories.npy",
    ...     frame_indices=[0, 10, 20],
    ...     blobs_collection_path="path/to/blobs_collection.pkl",
    ... )

    """
    # Input validation + load trajectories
    trajectories_path = Path(trajectories_path)
    if blobs_collection_path is not None:
        blobs_collection_path = Path(blobs_collection_path)

    trajectories = _validate_inputs(
        trajectories_path=trajectories_path,
        frame_indices=frame_indices,
        bbox_size=bbox_size,
        blobs_collection_path=blobs_collection_path,
    )
    n_total_frames, n_animals, _ = trajectories.shape

 
    # Sort and deduplicate frame indices
 
    frame_indices_sorted = sorted(set(frame_indices))
    n_selected_frames = len(frame_indices_sorted)

 
    # Build position / shape / category arrays
 
    if blobs_collection_path is not None:
        logger.info(
            "Loading bounding boxes from blobs collection: "
            f"{blobs_collection_path}"
        )
        position_arr, shape_arr, category_arr = _arrays_from_blobs(
            blobs_collection_path,
            frame_indices_sorted,
            n_animals,
        )
    else:
        logger.info(
            "Computing bounding boxes from trajectories with fixed "
            f"bbox_size={bbox_size}."
        )
        position_arr, shape_arr, category_arr = _arrays_from_trajectories(
            trajectories,
            frame_indices_sorted,
            n_animals,
            bbox_size,  # type: ignore[arg-type]  # cannot be None here
        )

 
    # Build metadata maps
 
    map_image_id_to_filename = {
        img_id: f"frame_{frame_idx:06d}.png"
        for img_id, frame_idx in enumerate(frame_indices_sorted)
    }
    map_category_to_str = {
        animal_id + 1: f"animal_{animal_id + 1}"
        for animal_id in range(n_animals)
    }

 
    # Assemble xarray dataset
 
    return xr.Dataset(
        data_vars={
            "position": (["image_id", "space", "id"], position_arr),
            "shape": (["image_id", "space", "id"], shape_arr),
            "category": (["image_id", "id"], category_arr),
        },
        coords={
            "image_id": list(range(n_selected_frames)),
            "space": ["x", "y"],
            "id": list(range(n_animals)),
        },
        attrs={
            "trajectories_file": str(trajectories_path),
            "blobs_collection_file": (
                str(blobs_collection_path)
                if blobs_collection_path is not None
                else None
            ),
            "images_directory": (
                str(images_dir) if images_dir is not None else None
            ),
            "map_category_to_str": map_category_to_str,
            "map_image_id_to_filename": map_image_id_to_filename,
        },
    )

def _validate_paths_and_bbox(
    trajectories_path: Path,
    frame_indices: list[int],
    bbox_size: tuple[float, float] | None,
    blobs_collection_path: Path | None,
) -> None:
    """Validate file paths, frame indices and bbox_size.

    Parameters
    ----------
    trajectories_path
        Path to the trajectories ``.npy`` file.
    frame_indices
        List of 0-based frame indices to process.
    bbox_size
        Fixed bounding box ``(width, height)``, or ``None``.
    blobs_collection_path
        Path to the blobs collection ``.pkl``, or ``None``.

    Raises
    ------
    FileNotFoundError
        If either file path does not exist.
    ValueError
        If any input constraint is violated.

    """
    if not trajectories_path.exists():
        raise FileNotFoundError(
            f"Trajectories file not found: {trajectories_path}"
        )
    if not frame_indices:
        raise ValueError("frame_indices must not be empty.")
    negative = [i for i in frame_indices if i < 0]
    if negative:
        raise ValueError(
            "All frame indices must be non-negative integers, "
            f"got {negative}."
        )
    if blobs_collection_path is None and bbox_size is None:
        raise ValueError(
            "Either bbox_size or blobs_collection_path must be provided."
        )
    if bbox_size is not None:
        if len(bbox_size) != 2:
            raise ValueError(
                "bbox_size must be a tuple of exactly two elements "
                "(width, height)."
            )
        if any(v <= 0 for v in bbox_size):
            raise ValueError(
                "Both elements of bbox_size must be strictly positive, "
                f"got {bbox_size}."
            )
    blobs_missing = (
        blobs_collection_path is not None
        and not blobs_collection_path.exists()
    )
    if blobs_missing:
        raise FileNotFoundError(
            f"Blobs collection file not found: {blobs_collection_path}"
        )


def _validate_inputs(
    trajectories_path: Path,
    frame_indices: list[int],
    bbox_size: tuple[float, float] | None,
    blobs_collection_path: Path | None,
) -> np.ndarray:
    """Validate all inputs and return the loaded trajectories array.

    Parameters
    ----------
    trajectories_path
        Path to the trajectories ``.npy`` file.
    frame_indices
        List of 0-based frame indices to process.
    bbox_size
        Fixed bounding box ``(width, height)``, or ``None``.
    blobs_collection_path
        Path to the blobs collection ``.pkl``, or ``None``.

    Returns
    -------
    np.ndarray
        Loaded trajectories array of shape ``(n_frames, n_animals, 2)``.

    Raises
    ------
    FileNotFoundError
        If either file path does not exist.
    ValueError
        If any input constraint is violated.

    """
    _validate_paths_and_bbox(
        trajectories_path, frame_indices, bbox_size, blobs_collection_path
    )
    trajectories = np.load(trajectories_path, allow_pickle=False)
    if trajectories.ndim != 3 or trajectories.shape[2] != 2:
        raise ValueError(
            "Expected trajectories array of shape "
            f"(n_frames, n_animals, 2), got {trajectories.shape}."
        )
    out_of_range = [
        i for i in frame_indices if i >= trajectories.shape[0]
    ]
    if out_of_range:
        raise ValueError(
            f"Frame indices {out_of_range} are out of range for the "
            f"trajectories array with {trajectories.shape[0]} frames."
        )
    return trajectories
def _arrays_from_trajectories(
    trajectories: np.ndarray,
    frame_indices: list[int],
    n_animals: int,
    bbox_size: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build position, shape and category arrays from a trajectories array.

    Parameters
    ----------
    trajectories
        Array of shape ``(n_frames, n_animals, 2)`` holding centroid
        coordinates ``(x, y)``.  ``NaN`` signals an undetected animal.
    frame_indices
        Sorted, deduplicated list of 0-based frame indices to process.
    n_animals
        Number of tracked animals (second axis of ``trajectories``).
    bbox_size
        Fixed bounding box ``(width, height)`` in pixels applied to
        every detected animal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - ``position_arr`` shape ``(n_frames, 2, n_animals)``.
        - ``shape_arr``    shape ``(n_frames, 2, n_animals)``.
        - ``category_arr`` shape ``(n_frames, n_animals)``, dtype int.
        ``NaN`` / ``-1`` are used for undetected animals.

    """
    n_frames = len(frame_indices)
    bbox_width = float(bbox_size[0])
    bbox_height = float(bbox_size[1])

    position_arr = np.full((n_frames, 2, n_animals), np.nan)
    shape_arr = np.full((n_frames, 2, n_animals), np.nan)
    category_arr = np.full((n_frames, n_animals), -1, dtype=int)

    for img_id, frame_idx in enumerate(frame_indices):
        for animal_idx in range(n_animals):
            centroid = trajectories[frame_idx, animal_idx, :]
            if np.any(np.isnan(centroid)):
                continue  # not detected: keep NaN / -1 defaults

            position_arr[img_id, 0, animal_idx] = centroid[0]  # x
            position_arr[img_id, 1, animal_idx] = centroid[1]  # y
            shape_arr[img_id, 0, animal_idx] = bbox_width
            shape_arr[img_id, 1, animal_idx] = bbox_height
            category_arr[img_id, animal_idx] = animal_idx + 1  # 1-based

    return position_arr, shape_arr, category_arr


def _arrays_from_blobs(
    blobs_collection_path: Path,
    frame_indices: list[int],
    n_animals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build position, shape and category arrays from a blobs collection.

    Parameters
    ----------
    blobs_collection_path
        Path to a pickled blobs collection.  The object must expose
        ``.blobs_in_video``: a list (indexed by frame) of lists of blob
        objects, each with:

        - ``.bounding_box``: ``[x_min, y_min, x_max, y_max]``;
        - ``.identity``: 1-based int animal identity (0/None = unidentified).

    frame_indices
        Sorted, deduplicated list of 0-based frame indices to process.
    n_animals
        Number of tracked animals derived from the trajectories array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Same shape convention as :func:`_arrays_from_trajectories`.

    Raises
    ------
    ValueError
        If the loaded object does not have a ``.blobs_in_video`` attribute.

    """
    n_frames = len(frame_indices)

    with open(blobs_collection_path, "rb") as fh:
        blobs_collection = pickle.load(fh)

    if not hasattr(blobs_collection, "blobs_in_video"):
        raise ValueError(
            "The blobs collection object loaded from "
            f"'{blobs_collection_path}' does not have a "
            "'blobs_in_video' attribute."
        )

    n_video_frames = len(blobs_collection.blobs_in_video)

    position_arr = np.full((n_frames, 2, n_animals), np.nan)
    shape_arr = np.full((n_frames, 2, n_animals), np.nan)
    category_arr = np.full((n_frames, n_animals), -1, dtype=int)

    for img_id, frame_idx in enumerate(frame_indices):
        if frame_idx >= n_video_frames:
            logger.warning(
                f"Frame index {frame_idx} exceeds the number of frames in "
                f"the blobs collection ({n_video_frames}).  Skipping."
            )
            continue

        for blob in blobs_collection.blobs_in_video[frame_idx]:
            if not hasattr(blob, "bounding_box") or not hasattr(
                blob, "identity"
            ):
                logger.warning(
                    f"Blob in frame {frame_idx} is missing 'bounding_box' "
                    "or 'identity' attributes.  Skipping."
                )
                continue

            identity = blob.identity
            if not identity:  # 0 or None: unidentified blob
                continue

            animal_idx = identity - 1  # convert 1-based to 0-based
            if animal_idx >= n_animals:
                logger.warning(
                    f"Blob identity {identity} exceeds n_animals={n_animals}."
                    "  Skipping."
                )
                continue

            x_min, y_min, x_max, y_max = blob.bounding_box
            width = float(x_max - x_min)
            height = float(y_max - y_min)

            position_arr[img_id, 0, animal_idx] = x_min + width / 2.0
            position_arr[img_id, 1, animal_idx] = y_min + height / 2.0
            shape_arr[img_id, 0, animal_idx] = width
            shape_arr[img_id, 1, animal_idx] = height
            category_arr[img_id, animal_idx] = identity

    return position_arr, shape_arr, category_arr