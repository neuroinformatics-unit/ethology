"""Utilities for creating datasets."""

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from loguru import logger
from torch.utils.data import random_split
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2


def create_coco_dataset(
    images_dir: str | Path,
    annotations_file: str | Path,
    composed_transform: transforms.Compose,
) -> CocoDetection:
    """Create a COCO dataset for object detection.

    Note: transforms are applied to the full dataset. If the dataset
    is later split, all splits will have the same transforms.
    """
    dataset_coco = CocoDetection(
        root=images_dir,
        annFile=annotations_file,
        transforms=composed_transform,
    )

    # wrap dataset for transforms v2
    dataset_transformed = wrap_dataset_for_transforms_v2(dataset_coco)

    return dataset_transformed


def split_dataset(
    dataset: torch.utils.data.Dataset,
    train_val_test_fractions: list[float],
    seed: int,
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
