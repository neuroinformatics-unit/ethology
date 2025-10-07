"""Utilities for creating and manipulating datasets for detection."""

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from loguru import logger
from torch.utils.data import random_split
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

from ethology import ETHOLOGY_CACHE_DIR
from ethology.io.annotations import save_bboxes
from ethology.io.annotations.validate import (
    ValidCOCO,
    _check_input,
)


def split_annotations_dataset_group_by(
    dataset: xr.Dataset,
    group_by_var: str,  # should be 1-dimensional along the samples_coordinate
    list_fractions: list[float],
    seed: int = 42,
    tolerance: int = 0,
    samples_coordinate: str = "image_id",
) -> tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    """Split an annotations dataset using an approximate subset-sum algorithm.

    The dataset is split in two. It returns the smallest split first.
    We assume the smallest split is the test set.
    """
    # Checks
    if sum(list_fractions) != 1:
        raise ValueError("The split fractions must sum to 1.")

    if len(list_fractions) != 2:
        raise ValueError("The list of fractions must have only two elements.")

    if any(fraction < 0 or fraction > 1 for fraction in list_fractions):
        raise ValueError("The split fractions must be between 0 and 1.")

    if len(dataset[group_by_var].shape) != 1:
        raise ValueError(
            f"The grouping variable {group_by_var} must be 1-dimensional along"
            f" {samples_coordinate}."
        )

    # Count number of samples per group and sort by count
    count_per_group_id = dict(
        Counter(dataset[group_by_var].values).most_common()[::-1]
    )

    # assert sum(count_per_group_id.values()) ==
    # len(dataset.get(samples_coordinate))

    # Compute number of samples in target subset
    # We define the target subset as the smallest subset.
    target_subset_count = int(
        min(list_fractions) * len(dataset.get(samples_coordinate))
    )

    # Get indices for target subset
    # idcs are from enumerating the keys of target_subset_count
    subset_idcs, _subset_n_samples = approximate_subset_sum(
        count_per_group_id,
        target_subset_count,
        seed=seed,
        tolerance=tolerance,
    )

    # Get group ids for the subset
    list_all_group_id = list(count_per_group_id.keys())
    subset_group_id = [list_all_group_id[x] for x in subset_idcs]

    # Create datasets for subset and not subset
    ds_subset = dataset.isel(
        {samples_coordinate: dataset[group_by_var].isin(subset_group_id)}
    )
    ds_not_subset = dataset.isel(
        {samples_coordinate: ~dataset[group_by_var].isin(subset_group_id)}
    )

    # # assert
    # assert np.unique(
    #     ds_subset.isel(
    #       {samples_coordinate: ds_subset[group_by_var].isin(subset_idcs)}
    #     )[group_by_var].values
    # ) == sorted(subset_idcs)

    return ds_subset, ds_not_subset


def split_annotations_dataset_random(
    ds: xr.Dataset,
    list_fractions: list[float],
    seed: int = 42,
    samples_coordinate: str = "image_id",
) -> list[xr.Dataset]:
    """Split a bbox annotations dataset into subsets using random sampling.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to split.
    list_fractions : list[float]
        Fractions of the total number of image_id samples to allocate to
        each subset.
    seed : int, optional
        Seed to use for the random number generator. Default is 42.
    samples_coordinate : str, optional
        The coordinate along which to split the dataset. Default is "image_id".

    Returns
    -------
    list[xr.Dataset]
        Subsets of the input dataset.

    """
    rng = np.random.default_rng(seed)

    # Compute number of samples for each split
    list_n_samples: list[int] = []
    for fraction in list_fractions[:-1]:
        list_n_samples.append(int(fraction * len(ds.get(samples_coordinate))))
    # append the remaining samples
    list_n_samples.append(
        len(ds.get(samples_coordinate)) - sum(list_n_samples)
    )

    # Sample indices for each split
    list_idcs_per_split: list[list[int]] = []
    for n_samples in list_n_samples:
        list_available_choices = list(
            set(range(len(ds.get(samples_coordinate))))
            - set(*list_idcs_per_split)
        )
        list_idcs_per_split.append(
            rng.choice(
                list_available_choices, n_samples, replace=False
            ).tolist()
        )

    # Indices per split should be exclusive
    assert (
        set.intersection(
            *[set(list_idcs) for list_idcs in list_idcs_per_split]
        )
        == set()
    )

    # Create datasets for each split
    list_ds = []
    for idcs in list_idcs_per_split:
        list_ds.append(ds.isel({samples_coordinate: idcs}))

    return list_ds


def approximate_subset_sum(
    map_ids_to_counts: dict[int, int],
    target: int,
    tolerance: int,
    seed: int | None,
) -> tuple[list[int], int]:
    """Solve subset sum problem by approximate greedy algorithm.

    We iterate through the elements in the list and add the element
    to the subset if the sum of the subset is less than the target value.
    We stop adding new elements when the sum of the subset is
    within the tolerance of the target value.

    Parameters
    ----------
    map_ids_to_counts : dict[int, int]
        Mapping from ids to counts. Usually the counts are the frame counts
        for the corresponding video ids.
    target : int
        The number of samples to allocate to the target subset.
    tolerance : int
        How many samples are allowed to be deviated from the count in
        the target subset. Should be positive. Same unit as the target value.
    seed : int | None
        Used to shuffle the order in which the elements are visited. If None,
        the order is not shuffled.

    Returns
    -------
    subset_idcs : list[int]
        Indices of the elements in the subset.
    subset_sum : int
        Sum of the elements in the subset.

    """
    # Check if keys are castable to int
    # if not, use keys from enumerate
    castable_as_int = True
    try:
        map_ids_to_counts = {int(k): v for k, v in map_ids_to_counts.items()}
    except ValueError:
        castable_as_int = False

    if not castable_as_int:
        list_id_count_tuples = [
            (k, v) for k, v in enumerate(map_ids_to_counts.values())
        ]
    else:
        list_id_count_tuples = list(map_ids_to_counts.items())

    # If seed is provided, shuffle the order in which the elems are visited
    # If not, the list_id_value_tuples are visited in order.
    if seed:
        rng = np.random.default_rng(seed)
        rng.shuffle(list_id_count_tuples)

    # loop thru elements in dict and add to list as long as
    # sum of elements in list is below target
    current_sum = 0
    current_subset_idcs = []
    for id, values_one_id in list_id_count_tuples:
        if current_sum + values_one_id <= target + tolerance:
            current_subset_idcs.append(id)
            current_sum += values_one_id

        if abs(current_sum - target) <= tolerance:
            break

    return current_subset_idcs, current_sum


def annotations_dataset_to_torch_dataset(
    ds: xr.Dataset,
    out_filepath: Path | str | None = None,
    images_directory: Path | str | None = None,
    transforms: transforms.Compose | None = None,
    kwargs: dict[str, Any] | None = None,
) -> CocoDetection:
    """Convert an bounding boxes annotations dataset to a torch dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to convert.
    out_filepath : Path | str | None, optional
        The path to the output COCO file.
    images_directory : Path | str | None, optional
        The path to the images directory.
    transforms : torchvision.transforms.v2.Compose | None, optional
        The transforms to apply to the dataset.
    kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the torch dataset constructor.

    Returns
    -------
    CocoDetection
        The converted torch dataset.

    """
    # Export xarray dataset to COCO file
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    if out_filepath is None:
        out_filepath = ETHOLOGY_CACHE_DIR / f"tmp_out_{timestamp}.json"
    else:
        suffix = Path(out_filepath).suffix
        path_without_suffix = Path(out_filepath).with_suffix("")
        out_filepath = Path(f"{path_without_suffix}_{timestamp}.{suffix}")

    out_file = save_bboxes.to_COCO_file(ds, out_filepath)
    logger.info(f"Exported temporary COCO file to {out_file}")

    # Get images directory
    # if not provided, check the dataset attributes
    if images_directory is None:
        try:
            images_directory = ds.attrs["images_directories"]
            if (
                isinstance(images_directory, list)
                and len(images_directory) > 0
            ):
                images_directory = images_directory[0]
                logger.warning(
                    f"Using first images directory only: {images_directory}"
                )  # TODO: loop thru them
        except KeyError as e:
            raise KeyError(
                "`images_directories` is not a dataset attribute. "
                "Please provide `images_directory` as an input."
            ) from e

    # Create torch dataset
    return CocoDetection(
        root=images_directory,
        annFile=out_file,
        transforms=transforms,
        **kwargs if kwargs is not None else {},
    )


# def torch_dataset_to_annotations_dataset(
#     torch_dataset: torch.utils.data.Dataset,
# ) -> xr.Dataset:
#     """Convert a torch dataset to an annotations dataset."""
#     # Read list of rows
#     list_rows = [annot for _img, annot in torch_dataset]

#     # ---------
#     # Read list of rows as a dataframe
#     df = pd.DataFrame(list_rows)

#     # Sort annotations by image_filename
#     df = df.sort_values(by=["image_filename"])

#     # Drop duplicates and reindex
#     # The resulting axis is labeled 0,1,…,n-1.
#     df = df.drop_duplicates(
#         subset=[col for col in df.columns if col != "annotation_id"],
#         ignore_index=True,
#         inplace=False,
#     )

#     # Cast bbox coordinates and shape as floats
#     for col in ["x_min", "y_min", "width", "height"]:
#         df[col] = df[col].astype(np.float64)

#     # Set the index name to "annotation_id"
#     df = df.set_index("annotation_id")
#     # ---------

#     # Get maps to set as dataset attributes
#     map_image_id_to_filename, map_category_to_str = (
#        load_bboxes._get_map_attributes_from_df(df)
#     )

#     # Convert dataframe to xarray dataset
#     ds = load_bboxes._df_to_xarray_ds(df)

#     # Add attributes to the xarray dataset
#     ds.attrs = {
#         # "annotation_files": file_paths,
#         "annotation_format": 'torch-dataset',
#         "map_category_to_str": map_category_to_str,
#         "map_image_id_to_filename": map_image_id_to_filename,
#     }
#     # -----------

#     # Add image dir as metadata
#     root = _find_nested_root(torch_dataset)
#     if root:
#         ds.attrs["images_directories"] = root


#     return ds


# def _find_nested_root(
#     dataset: torch.utils.data.Dataset
# ) -> str | Path | None:
#     """Find root of a possibly nested dataset.

#     Parameters
#     ----------
#     dataset : torch.utils.data.Dataset
#         The dataset to check. It may be the result of multiple
#         splits, and therefore be nested.

#     Returns
#     -------
#     str or Path or None
#         The nested root value for the dataset, or None if not found

#     """
#     current = dataset

#     # Check current level
#     if hasattr(current, "root"):
#         return current

#     # Check through dataset levels
#     while hasattr(current, "dataset"):
#         current = current.dataset
#         if hasattr(current, "root"):
#             return current.root

#     return None


# -----------------------------------------------------------------------------


@_check_input(validator=ValidCOCO)
def torch_dataset_from_COCO_file(
    annotations_file: str | Path,
    images_directory: str | Path,
    kwargs: dict[str, Any] | None = None,
) -> CocoDetection:
    """Create a COCO dataset for object detection.

    Note: transforms are applied to the full dataset. If the dataset
    is later split, all splits will have the same transforms.

    Parameters
    ----------
    annotations_file : str | Path
        The path to the input COCO file.
    images_directory : str | Path
        The path to the images directory.
    kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the torch dataset constructor.

    Returns
    -------
    torch.utils.data.Dataset
        The converted torch dataset.

    """
    dataset_coco = CocoDetection(
        root=str(images_directory),
        annFile=str(annotations_file),
        **kwargs if kwargs is not None else {},
    )

    # wrap dataset for transforms v2
    dataset_transformed = wrap_dataset_for_transforms_v2(dataset_coco)

    return dataset_transformed


def split_torch_dataset(
    dataset: torch.utils.data.Dataset,
    train_val_test_fractions: list[float],
    seed: int = 42,
) -> tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    """Split a torchdataset into train, validation, and test sets.

    Note that transforms are already applied to the input dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The torch dataset to split.
    train_val_test_fractions : list[float]
        The fractions of the dataset to allocate to the train, validation,
        and test sets.
    seed : int, optional
        The seed to use for the random number generator. Default is 42.

    Returns
    -------
    tuple[torch.utils.data.Dataset]
        The train, validation, and test sets.

    """
    # Check that the fractions sum to 1
    if sum(train_val_test_fractions) != 1:
        raise ValueError("The split fractions must sum to 1.")

    # Log transforms applied to the dataset
    logger.info(
        f"Dataset transforms (propagated to all splits): {dataset.transforms}"
    )

    # Create random number generator for reproducibility if seed is provided
    rng_split = None
    if seed is not None:
        rng_split = torch.Generator().manual_seed(seed)

    # Split dataset
    train_dataset, test_dataset, val_dataset = random_split(
        dataset,
        train_val_test_fractions,
        generator=rng_split,
    )

    # Print number of samples in each split
    logger.info(f"Seed: {seed}")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")
    logger.info(f"Number of test samples: {len(test_dataset)}")

    return train_dataset, test_dataset, val_dataset
