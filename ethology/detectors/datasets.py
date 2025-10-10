"""Utilities for creating and manipulating datasets for detection."""

from collections import Counter
from typing import TypedDict

import numpy as np
import xarray as xr
from loguru import logger


def split_annotations_dataset_group_by(
    dataset: xr.Dataset,
    group_by_var: str,  # should be 1-dimensional along the samples_coordinate
    list_fractions: list[float],
    epsilon: float = 0.01,
    samples_coordinate: str = "image_id",
) -> tuple[
    xr.Dataset,
    xr.Dataset,
]:
    """Split an annotations dataset using an approximate subset-sum approach.

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

    # Compute number of samples in target subset
    # We define the target subset as the smallest subset.
    target_subset_count = int(
        min(list_fractions) * len(dataset.get(samples_coordinate))
    )

    # Get list of (id, count) tuples
    # Count number of samples per group and sort by count
    count_per_group_id = Counter(dataset[group_by_var].values).most_common()[
        ::-1
    ]

    # Cast ids to integer
    try:
        list_id_count_tuples = [(int(id), c) for id, c in count_per_group_id]
    except ValueError:
        list_id_count_tuples = list(
            enumerate(c for _id, c in count_per_group_id)
        )

    # # # If seed is provided, shuffle? -- make a tuple?
    # if seed:
    #     rng = np.random.default_rng(seed)
    #     rng.shuffle(list_id_count_tuples)

    # Get indices for target subset
    # idcs are from enumerating the keys of target_subset_count
    subset_dict = _approximate_subset_sum(
        list_id_count_tuples,
        target_subset_count,
        epsilon=epsilon,
    )

    # Create datasets for subset and not subset
    subset_group_ids = [count_per_group_id[x][0] for x in subset_dict["ids"]]
    ds_subset = dataset.isel(
        {samples_coordinate: dataset[group_by_var].isin(subset_group_ids)}
    )
    ds_not_subset = dataset.isel(
        {samples_coordinate: ~dataset[group_by_var].isin(subset_group_ids)}
    )

    # throw warning if a subset is empty
    if any(len(ds.image_id) == 0 for ds in [ds_subset, ds_not_subset]):
        logger.warning("One of the subset datasets is empty.")

    # # assert
    # assert np.unique(
    #     ds_subset.isel(
    #       {samples_coordinate: ds_subset[group_by_var].isin(subset_idcs)}
    #     )[group_by_var].values
    # ) == sorted(subset_idcs)

    # Return result in the same order as the input list of fractions
    # argsort twice gives the inverse permutation
    idcs_sorted = np.argsort(list_fractions)  # idcs to map input -> sorted
    idcs_original = np.argsort(idcs_sorted)  # idcs to map sorted -> input

    list_ds_sorted = [ds_subset, ds_not_subset]
    return tuple(list_ds_sorted[i] for i in idcs_original)


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


class SubsetDict(TypedDict):
    """Type definition for subset dictionary.

    Used in approximate subset sum algorithm.
    """

    sum: int
    ids: list[int]


def _approximate_subset_sum(list_id_counts, target, epsilon) -> SubsetDict:
    """Approximate subset sum problem.

    At each iteration of the loop, trimming introduces a small error.
    The algorithm runs n iterations (one per item), so errors can compound.
    The formula delta = epsilon / (2n) ensures that when errors compound over
    n iterations, the total error stays within epsilon.

    epsilon is a percentage of the optimal solution, not the target.
    If OPT is the best possible subset sum ≤ target, the algorithm guarantees:
    result ≥ (1 - epsilon) * OPT
    So with epsilon = 0.2:
    You're guaranteed to get a result within 20% of the optimal
    NOT necessarily within 20% of the target

    Example:
    target = 3450
    epsilon = 0.2

    Suppose the optimal subset sum is OPT = 3400 (best you can do without
    exceeding 3450).
    The algorithm guarantees your result will be somewhere between
    (3400 * 0.8) = 2720 and 3450.

    Ref:
    - https://en.wikipedia.org/wiki/Subset_sum_problem#Fully-polynomial_time_approximation_scheme
    - https://nerderati.com/bartering-for-beers-with-approximate-subset-sums/

    """
    # Checks
    if np.min([x[1] for x in list_id_counts]) > target:
        logger.warning("All counts are greater than the target.")
        return {"sum": 0, "ids": []}

    # Early exit if the minimum count is equal to the target
    elif np.min([x[1] for x in list_id_counts]) == target:
        idx_min = np.argmin([x[1] for x in list_id_counts])
        id, count = list_id_counts[idx_min]
        return {"sum": count, "ids": [id]}

    # initialize list of all subsets whose sum is below the target
    list_subsets: list[SubsetDict] = [{"sum": 0, "ids": []}]

    # loop thru list of (id, count) pairs
    for id, count in list_id_counts:
        # Add current element to each existing subset in list
        # and extend list if the resulting subset sum is below the target.
        list_subsets.extend(
            [
                {
                    "sum": subset["sum"] + count,
                    "ids": subset["ids"] + [id],
                }
                for subset in list_subsets
                if subset["sum"] + count <= target
            ]
        )

        # Remove near-duplicate subsets in terms of total sum
        list_subsets = _remove_near_duplicate_subsets(
            list_subsets, delta=float(epsilon) / (2 * len(list_id_counts))
        )

    if len(list_subsets) == 0:
        logger.warning("No subset found with sum below the target.")
        return {"sum": 0, "ids": []}

    # Return the subset with highest sum but below the target
    return list_subsets[-1]


def _remove_near_duplicate_subsets(list_subsets, delta):
    """Remove near-duplicate subsets in terms of their total sum.

    It only keeps subsets whose sum is sufficiently larger than the previous
    subset sum, in ascending order.

    This means that given two subsets whose sums are within delta% of each
    other, it will keep the smaller one, since the values are visited in order.
    """
    # ensure list of subsets is sorted by total sum, in ascending order
    list_subsets = sorted(list_subsets, key=lambda x: x["sum"])

    # loop thru list of subsets
    # keep only those whose sum is delta% larger than the previous subset sum
    list_subsets_trimmed = [
        list_subsets[0]
    ]  # never trim zero subset; [{"sum": 0, "ids": []}]
    previous_subset_sum = 0
    for subset in list_subsets[1:]:  # do not trim zero subset?
        if subset["sum"] > previous_subset_sum * (1 + delta):
            list_subsets_trimmed.append(subset)
            previous_subset_sum = subset["sum"]

    return list_subsets_trimmed
