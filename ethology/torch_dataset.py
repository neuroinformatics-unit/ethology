"""Utilities for creating and manipulating torchvision COCO datasets."""

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
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


@_check_input(validator=ValidCOCO)
def from_COCO_file(
    annotations_file: str | Path,
    images_directory: str | Path,
    kwargs: dict[str, Any] | None = None,
) -> CocoDetection:
    """Create a COCO dataset for object detection.

    Note: transforms are applied to the full dataset. If the dataset
    is later split, all splits will have the same transforms.
    """
    dataset_coco = CocoDetection(
        root=str(images_directory),
        annFile=str(annotations_file),
        **kwargs if kwargs is not None else {},
    )

    # wrap dataset for transforms v2
    dataset_transformed = wrap_dataset_for_transforms_v2(dataset_coco)

    return dataset_transformed


def from_annotations_dataset(
    ds: xr.Dataset,
    out_filepath: Path | str | None = None,
    images_directory: Path | str | None = None,
    kwargs: dict[str, Any] | None = None,
) -> torch.utils.data.Dataset:
    """Convert an bounding boxes annotations dataset to a torch dataset."""
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
                )
        except KeyError as e:
            raise KeyError(
                "`images_directories` is not a dataset attribute. "
                "Please provide `images_directory` as an input."
            ) from e

    # Create torch dataset
    return datasets.CocoDetection(
        root=images_directory,
        annFile=out_file,
        **kwargs if kwargs is not None else {},
    )


def split_torch_dataset(
    dataset: torch.utils.data.Dataset,
    train_val_test_fractions: list[float],
    seed: int | None = None,
) -> tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    """Split a dataset into train, validation, and test sets.

    Note: transforms are already applied to the input dataset.
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
