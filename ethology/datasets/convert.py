"""Convert betweendataset formats."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr


def torch_dataset_to_xr_dataset(
    torch_dataset: torch.utils.data.Dataset,
) -> xr.Dataset:
    """Convert a torch dataset to an xarray detections dataset."""
    # Read list of annotations as a dataframe
    list_annot = [annot for _img, annot in torch_dataset]
    df_annot = pd.DataFrame(list_annot)

    # Compute centroid, shape and labels
    df_annot["centroid"] = df_annot["boxes"].apply(
        lambda x: (0.5 * (x[:, 0:2] + x[:, 2:4])).numpy().astype(float)
    )
    df_annot["shape"] = df_annot["boxes"].apply(
        lambda x: (x[:, 2:4] - x[:, 0:2]).numpy().astype(float)
    )
    df_annot["labels"] = df_annot["labels"].apply(
        lambda x: x.numpy().reshape(-1, 1).astype(int)
    )

    # Compute maximum number of annotations per image
    df_annot["n_annotations"] = df_annot["boxes"].apply(lambda x: x.shape[0])
    n_max_annotations = df_annot["n_annotations"].max()

    # Pad arrays to n_max_annotations
    array_dict = {}
    map_name_to_padding = {
        "centroid": np.nan,
        "shape": np.nan,
        "labels": -1,
    }
    for array_name in map_name_to_padding:
        array_dict[array_name] = np.stack(
            [
                np.pad(
                    arr,
                    ((0, n_max_annotations - arr.shape[0]), (0, 0)),
                    mode="constant",
                    constant_values=map_name_to_padding[array_name],
                ).T
                for arr in df_annot[array_name].to_list()
            ]
        )

    # Return xarray dataset
    xr_dataset = xr.Dataset(
        data_vars={
            "position": (["image_id", "space", "id"], array_dict["centroid"]),
            "shape": (["image_id", "space", "id"], array_dict["shape"]),
            "category": (["image_id", "id"], array_dict["labels"].squeeze()),
        },
        coords={
            "image_id": df_annot["image_id"].values,
            "space": ["x", "y"],
            "id": range(n_max_annotations),
        },
    )

    # Add metadata
    root = find_nested_root(torch_dataset)
    if root:
        xr_dataset.attrs["images_directories"] = root

    return xr_dataset


def find_nested_root(dataset: torch.utils.data.Dataset) -> str | Path | None:
    """Find root of a possibly nested dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to check. It may be the result of multiple
        splits, and therefore be nested.

    Returns
    -------
    str or Path or None
        The nested root value for the dataset, or None if not found

    """
    current = dataset

    # Check current level
    if hasattr(current, "root"):
        return current

    # Check through dataset levels
    while hasattr(current, "dataset"):
        current = current.dataset
        if hasattr(current, "root"):
            return current.root

    return None
