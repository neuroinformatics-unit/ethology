"""Utilities for creating and manipulating datasets for detection."""

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from loguru import logger
from torch.utils.data import random_split
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

from ethology import ETHOLOGY_CACHE_DIR
from ethology.io.annotations import save_bboxes
from ethology.io.annotations.validate import ValidCOCO, _check_input


def annotations_dataset_to_torch_dataset(
    ds: xr.Dataset,
    images_directory: Path | str | None = None,
    transforms: transforms.Compose | None = None,
    out_filepath: Path | str | None = None,
    kwargs: dict[str, Any] | None = None,
) -> CocoDetection:
    """Convert an bounding boxes annotations dataset to a torch dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to convert.
    images_directory : Path | str | None, optional
        The path to the images directory.
    transforms : torchvision.transforms.v2.Compose | None, optional
        The transforms to apply to the dataset.
    out_filepath : Path | str | None, optional
        The path to the output COCO file.
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
        images_directory = ds.attrs.get("images_directories", None)
        if isinstance(images_directory, list) and len(images_directory) > 0:
            images_directory = images_directory[0]
            logger.warning(
                f"Using first images directory only: {images_directory}"
            )  # TODO: loop thru them?
        elif images_directory is None:
            raise KeyError(
                "`images_directories` is not set. "
                "Please provide `images_directory` as an input or "
                "add it to the dataset attributes."
            )

    # Create torch dataset
    return CocoDetection(
        root=images_directory,
        annFile=out_file,
        transforms=transforms,
        **kwargs if kwargs is not None else {},
    )


def torch_dataset_to_annotations_dataset(
    torch_dataset: torch.utils.data.Dataset,
) -> xr.Dataset:
    """Convert a torch dataset to an annotations dataset."""
    pass


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
