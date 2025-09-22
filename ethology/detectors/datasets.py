"""Utilities for creating and manipulating datasets for detection."""

import itertools
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from loguru import logger
from torch.utils.data import random_split
from torchvision import datasets
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
    list_fractions: list[float, float],
    seed: int | None = None,
    tolerance: float = 0.01,
    samples_coordinate: str = "image_id",
) -> tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    """Split an annotations dataset into two using an approximate subset sum algorithm.

    Returns the smallest split first. We assume the smallest split is the test
    set. Only two splits.
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

    # Count number of samples per group
    group_count_per_group_id = {
        int(k): len(list(g))
        for k, g in itertools.groupby(dataset[group_by_var].values)
    }

    # Compute number of samples in smallest subset
    target_n_samples = min(list_fractions) * len(
        dataset.get(samples_coordinate)
    )

    # Get indices for smallest subset
    subset_idcs, _subset_n_samples = approximate_subset_sum(
        group_count_per_group_id,
        target_n_samples,
        seed=seed,
        tolerance=tolerance,
    )

    # Create datasets for subset and not subset
    ds_subset = dataset.isel(
        {samples_coordinate: dataset[group_by_var].isin(subset_idcs)}
    )
    ds_not_subset = dataset.isel(
        {samples_coordinate: ~dataset[group_by_var].isin(subset_idcs)}
    )

    # # assert
    # assert np.unique(
    #     ds_subset.isel({samples_coordinate: ds_subset[group_by_var].isin(subset_idcs)})[group_by_var].values
    # ) == sorted(subset_idcs)

    return ds_subset, ds_not_subset


def split_annotations_dataset_random(
    ds: xr.Dataset,
    list_fractions: list[float, ...],
    seed: int = 42,
    samples_coordinate: str = "image_id",
) -> tuple[xr.Dataset, ...]:
    """Split a bbox annotations dataset into subsets using random sampling.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to split.
    list_fractions : list[float, ...]
        Fractions of the total number of image_id samples to allocate to
        each subset.
    seed : int, optional
        Seed to use for the random number generator. Default is 42.
    samples_coordinate : str, optional
        The coordinate along which to split the dataset. Default is "image_id".

    Returns
    -------
    tuple[xr.Dataset, ...]
        Subsets of the input dataset.
    """
    rng = np.random.default_rng(seed)

    # Compute number of samples for each split
    list_n_samples = []
    for fraction in list_fractions[:-1]:
        list_n_samples.append(int(fraction * len(ds.get(samples_coordinate))))
    # append the remaining samples
    list_n_samples.append(
        len(ds.get(samples_coordinate)) - sum(list_n_samples)
    )

    # Sample indices for each split
    list_idcs = []
    for n_samples in list_n_samples:
        list_choices = set(np.arange(len(ds.get(samples_coordinate)))) - set(
            list_idcs
        )
        list_idcs.append(rng.choice(list_choices, n_samples, replace=False))

    # Indices per split should be exclusive
    assert set.intersection(*list_idcs) == set()

    # Create datasets for each split
    list_ds = []
    for idcs in list_idcs:
        list_ds.append(ds.isel({samples_coordinate: idcs}))

    return list_ds


def approximate_subset_sum(
    map_ids_to_values: dict[int, int] | list[int],
    target: int,
    tolerance: float = 0.05,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[list[int], int]:
    """Solve subset sum problem by approximate greedy algorithm.

    We iterate through the elements in the list and add the element
    to the subset if the sum of the subset is less than the target value.
    We stop adding new elements when the sum of the subset is
    within the tolerance of the target value.

    Parameters
    ----------
    map_ids_to_values : dict[int, int] | list[int]
        Mapping from ids to values. Usually the values are the frame counts
        for the corresponding video ids.
    target : int
        Target value. Usually the number of frames to allocate to the test set.
    tolerance : float, optional
        Tolerance. The percentage of the target value that is allowed to be
        deviated from. Default is 0.05.
    seed : int, optional
        Used to shuffle the order in which the elements are visited.
        Default is 42.
    shuffle : bool, optional
        Whether to shuffle the order in which the elements are visited.
        Default is True.

    Returns
    -------
    subset_idcs : list[int]
        Indices of the elements in the subset.
    subset_sum : int
        Sum of the elements in the subset.

    """
    if isinstance(map_ids_to_values, list):
        list_id_value_tuples = [
            (k, v) for k, v in enumerate(map_ids_to_values)
        ]
    else:
        list_id_value_tuples = [(k, v) for k, v in map_ids_to_values.items()]

    # shuffle the order in which the elems are visted
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(list_id_value_tuples)

    # loop thru elements in dict and add to list as long as
    # sum of elements in list is below target
    current_sum = 0
    current_subset_idcs = []
    for id, values_one_id in list_id_value_tuples:
        if current_sum + values_one_id <= target * (1 + tolerance):
            current_subset_idcs.append(id)
            current_sum += values_one_id

        if abs(current_sum - target) <= target * tolerance:
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
    return datasets.CocoDetection(
        root=images_directory,
        annFile=out_file,
        transforms=transforms,
        **kwargs if kwargs is not None else {},
    )


# -----------------------------------------------------------------------------


def split_torch_dataset(
    dataset: torch.utils.data.Dataset,
    train_val_test_fractions: list[float, float, float],
    seed: int | None = None,
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
        The dataset to split.
    train_val_test_fractions : list[float]
        The fractions of the dataset to allocate to the train, validation,
        and test sets.
    seed : int | None, optional
        The seed to use for the random number generator.

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
