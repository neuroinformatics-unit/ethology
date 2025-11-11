"""Utilities for splitting annotations datasets."""

from collections import Counter
from typing import TypedDict

import numpy as np
import xarray as xr
from loguru import logger
from sklearn.model_selection import GroupKFold


def split_dataset_group_by(
    dataset: xr.Dataset,
    group_by_var: str,
    list_fractions: list[float],
    samples_coordinate: str = "image_id",
    method: str = "auto",
    seed: int = 42,
    epsilon: float = 0,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Split an annotations dataset by grouping variable.

    Split an ``ethology`` annotations dataset into two subsets ensuring that
    the subsets are disjoint in the grouping variable (i.e., no group appears
    in both subsets). The function automatically chooses between a
    "group k-fold" approach and an "approximate subset-sum" approach based on
    the number of unique groups and requested split fractions.

    Parameters
    ----------
    dataset : xarray.Dataset
        The annotations dataset to split.
    group_by_var : str
        The grouping variable to use for splitting the dataset. Must be
        1-dimensional along the ``samples_coordinate``.
    list_fractions : list[float, float]
        The fractions of the input annotations dataset to allocate to
        each subset. Must contain only two elements and sum to 1.
    samples_coordinate : str, optional
        The coordinate along which to split the dataset. Default is
        ``image_id``.
    method : str, optional
        Method to use: ``auto``, ``kfold``, or ``apss``. When ``auto``,
        it automatically selects between ``kfold`` or ``apss`` based on
        the number of unique groups. See Notes for further details.
        Default is ``auto``.
    seed : int, optional
        Random seed for reproducibility used in the "group k-fold" approach.
        Controls both the shuffling of the sample indices and the random
        selection of the output split from all possible ones. Only used when
        ``method`` is ``kfold`` or when ``auto`` selects ``kfold``.
        Default is 42.
    epsilon : float, optional
        The approximation tolerance for the "approximate subset-sum" approach,
        expressed as a fraction of 1. The sum of samples in the smallest subset
        is guaranteed to be at least ``(1 - epsilon)`` times the optimal sum
        for the requested fractions and grouping variable. When ``epsilon``
        is 0, the algorithm finds the exact optimal sum. Larger values result
        in faster computation but may yield subsets with a total number of
        samples further from the optimal. Only used when ``method`` is
        ``apss`` or when ``auto`` selects ``apss``. Default is 0.

    Returns
    -------
    tuple[xarray.Dataset, xarray.Dataset]
        The two subsets of the input dataset. The subsets are returned in the
        same order as the input list of fractions ``list_fractions``.

    Raises
    ------
    ValueError
        If the elements of ``list_fractions`` are not exactly two, are not
        between 0 and 1, or do not sum to 1. If ``group_by_var`` is not
        1-dimensional along the ``samples_coordinate``. If ``method`` is
        ``kfold`` but there are insufficient groups for the requested split
        fractions.

    Notes
    -----
    When ``method`` is ``auto``, the function automatically selects between two
    approaches:

    - **Group k-fold method** (default when sufficient groups exist): used
      when the number of unique groups is greater than or equal to the number
      of required folds (calculated as ``1 / min(list_fractions)``). This
      method computes all possible partitions of groups into folds and randomly
      selects one of them as the output split. The selection is controlled by
      the ``seed`` parameter for reproducibility. We use
      :class:`sklearn.model_selection.GroupKFold` cross-validator to compute
      all possible partitions of groups into folds.

    - **Approximate subset-sum method** (fallback): used when there are too few
      unique groups for group k-fold splitting. This method deterministically
      finds a subset of groups whose combined sample count best matches the
      requested fractions. The ``epsilon`` parameter controls the
      speed-accuracy tradeoff. When ``epsilon`` is 0, the algorithm finds the
      exact optimal sum. Larger values of ``epsilon`` result in faster
      computation but may yield subsets with a total number of samples further
      from the optimal. In cases where no valid split exists (e.g., all groups
      have more samples than the target), one subset may be empty and a
      warning is logged.

    See Also
    --------
    :class:`sklearn.model_selection.GroupKFold` : Group k-fold cross-validator.

    Examples
    --------
    Split a dataset of 100 images extracted from 10 different videos.
    The xarray dataset has a single data variable ``video_id`` defined along
    the ``image_id`` dimension. We would like to compute an 80/20 split,
    ensuring the subsets of the dataset are disjoint in the grouping
    variable ``video_id``. Since there are many unique groups
    (i.e., unique video IDs), the function automatically selects
    the "group k-fold" method.

    >>> import xarray as xr
    >>> from ethology.datasets.split import split_dataset_group_by
    >>> ds_large = xr.Dataset(
    ...     data_vars=dict(
    ...         video_id=("image_id", np.tile(np.arange(10), 10)),
    ...     ),  # 10 different video IDs across 100 images
    ...     coords=dict(image_id=range(100)),
    ... )
    >>> ds_subset_1, ds_subset_2 = split_dataset_group_by(
    ...     ds_large, "video_id", [0.8, 0.2], seed=42
    ... )
    >>> print(len(ds_subset_1.image_id) / len(ds_large.image_id))  # 0.8
    >>> print(len(ds_subset_2.image_id) / len(ds_large.image_id))  # 0.2


    Using different seeds produces different splits when the "group k-fold"
    method is used:

    >>> ds_subset_1b, ds_subset_2b = split_dataset_group_by(
    ...     ds_large, "video_id", [0.8, 0.2], seed=123
    ... )
    >>> assert not ds_subset_1.equals(ds_subset_1b)
    >>> assert not ds_subset_2.equals(ds_subset_2b)
    >>> print(len(ds_subset_1b.image_id) / len(ds_large.image_id))  # 0.8
    >>> print(len(ds_subset_2b.image_id) / len(ds_large.image_id))  # 0.2

    The function automatically selects the appropriate method. In the example
    below, a smaller dataset with 3 unique video IDs is used. With a 0.2
    minimum fraction (requiring 5 folds), the "group k-fold" method cannot be
    used, since there would be more folds than groups. Therefore, the
    "approximate subset-sum" method is selected automatically and an
    approximate split is returned. Note that when using the ``apss`` method,
    the seed value is ignored.

    >>> ds_small = xr.Dataset(
    ...     data_vars=dict(
    ...         video_id=("image_id", [1, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 1]),
    ...     ),
    ...     coords=dict(image_id=range(12)),
    ... )
    >>> ds_subset_1, ds_subset_2 = split_dataset_group_by(
    ...     ds_small,
    ...     group_by_var="video_id",
    ...     list_fractions=[0.8, 0.2],
    ... )
    >>> print(len(ds_subset_1.image_id) / len(ds_small.image_id))  # 0.833
    >>> print(len(ds_subset_2.image_id) / len(ds_small.image_id))  # 0.166

    The ``epsilon`` parameter controls the approximation for the subset-sum
    method when auto-selected or explicitly specified:

    >>> ds_subset_1, ds_subset_2 = split_dataset_group_by(
    ...     ds_small,
    ...     group_by_var="video_id",
    ...     list_fractions=[0.8, 0.2],
    ...     epsilon=0.1,  # accept a solution >= 90% of the optimal
    ...     method="apss",
    ... )
    >>> print(len(ds_subset_1.image_id) / len(ds_small.image_id))  # 0.833
    >>> print(len(ds_subset_2.image_id) / len(ds_small.image_id))  # 0.166

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

    # Count unique groups
    n_unique_groups = len(np.unique(dataset[group_by_var].values))
    n_required_folds = int(np.rint(1 / min(list_fractions)))

    # Auto-select method
    if method == "auto":
        if n_unique_groups >= n_required_folds:
            method = "kfold"
        else:
            method = "apss"
            logger.info(
                f"Only {n_unique_groups} unique groups exist but "
                f"{n_required_folds} are required for k-fold method. "
                "Auto-selected approximate subset-sum method "
                f"with epsilon={epsilon}. Seed setting is ignored."
            )

    # Dispatch to appropriate method
    if method == "kfold":
        logger.info(
            f"Using group k-fold method with {n_required_folds} folds "
            f"and seed={seed}."
        )
        return _split_dataset_group_by_kfold(
            dataset, group_by_var, list_fractions, samples_coordinate, seed
        )
    elif method == "apss":
        logger.info(
            f"Using approximate subset-sum method with epsilon={epsilon}."
        )
        return _split_dataset_group_by_apss(
            dataset, group_by_var, list_fractions, epsilon, samples_coordinate
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _split_dataset_group_by_kfold(
    dataset: xr.Dataset,
    group_by_var: str,
    list_fractions: list[float],
    samples_coordinate: str = "image_id",
    seed: int = 42,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Split an annotations dataset using scikit-learn's GroupKFold.

    Split an ``ethology`` annotations dataset into two subsets
    ensuring that the subsets are disjoint in the grouping variable.
    This method uses scikit-learn's GroupKFold cross-validator to
    randomly partition groups into folds and then selects one fold
    as the smaller subset and the remaining folds as the larger subset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The annotations dataset to split.
    group_by_var : str
        The grouping variable to use for splitting the dataset. Must be
        1-dimensional along the ``samples_coordinate``.
    list_fractions : list[float, float]
        The fractions of the input annotations dataset to allocate to
        each subset. Must contain only two elements and sum to 1.
    samples_coordinate : str, optional
        The coordinate along which to split the dataset. Default is
        ``image_id``.
    seed : int, optional
        Random seed for reproducibility. Controls both the GroupKFold
        indices shuffling and the random selection of the output split from
        all the possible ones. Default is 42.

    Returns
    -------
    tuple[xarray.Dataset, xarray.Dataset]
        The two subsets of the input dataset. The subsets are returned in the
        same order as the input list of fractions.

    Raises
    ------
    ValueError

        If the elements of ``list_fractions`` are not exactly two, are not
        between 0 and 1, or do not sum to 1. If ``group_by_var`` is not
        1-dimensional along the ``samples_coordinate``.

    Examples
    --------
    Split a dataset with a single data variable ``foo`` defined along the
    ``image_id`` dimension into an 80/20 split, ensuring that the
    subsets are disjoint in the grouping variable ``foo``.

    >>> from ethology.datasets.split import _split_dataset_group_by_kfold
    >>> import xarray as xr
    >>> ds = xr.Dataset(
    >>>     data_vars=dict(
    >>>         foo=("image_id", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    >>>     ),  # 0: 10 counts, 1: 2 counts
    >>>     coords=dict(
    >>>         image_id=range(12),
    >>>     ),
    >>> )
    >>> ds_subset_1, ds_subset_2 = _split_dataset_group_by_kfold(
    >>>     ds,
    >>>     group_by_var="foo",
    >>>     list_fractions=[0.2, 0.8],
    >>>     seed=42,
    >>> )

    The ``seed`` parameter ensures reproducibility. Using the same seed
    on the same dataset will always produce the same split. Using a different
    seed will produce a different split.

    >>> ds_subset_1b, ds_subset_2b = _split_dataset_group_by_kfold(
    >>>     ds,
    >>>     group_by_var="foo",
    >>>     list_fractions=[0.2, 0.8],
    >>>     seed=123,
    >>> )
    >>> assert not ds_subset_1.equals(ds_subset_1b)
    >>> assert not ds_subset_2.equals(ds_subset_2b)

    """
    # Initialise k-fold iterator
    n_folds_per_shuffle = int(np.rint(1 / min(list_fractions)))
    gkf = GroupKFold(
        n_splits=n_folds_per_shuffle, shuffle=True, random_state=seed
    )

    # Compute all possible shuffles
    # In each shuffle, one fold is the test set,
    # the rest of folds make up the train set
    train_test_idcs_per_shuffle = list(
        gkf.split(
            dataset[samples_coordinate].values,
            groups=dataset[group_by_var].values,
        )
    )

    # Randomly pick one of the shuffles
    rng = np.random.default_rng(seed)
    shuffle_idx = rng.choice(len(train_test_idcs_per_shuffle))
    train_idcs, test_idcs = train_test_idcs_per_shuffle[shuffle_idx]

    # Split the datasets
    ds_train = dataset.isel({samples_coordinate: train_idcs})
    ds_test = dataset.isel({samples_coordinate: test_idcs})
    list_ds_sorted = [ds_test, ds_train]  # sorted in increasing size

    # Return datasets in the same order as the input list of fractions
    idcs_sorted = np.argsort(list_fractions)  # idcs to map input -> sorted
    idcs_original = np.argsort(idcs_sorted)  # idcs to map sorted -> input

    return tuple(list_ds_sorted[i] for i in idcs_original)


def _split_dataset_group_by_apss(
    dataset: xr.Dataset,
    group_by_var: str,
    list_fractions: list[float],
    epsilon: float = 0,
    samples_coordinate: str = "image_id",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Split an annotations dataset using an approximate subset-sum approach.

    Split an ``ethology`` annotations dataset into two subsets
    ensuring that the subsets are disjoint in the grouping variable.

    Parameters
    ----------
    dataset : xarray.Dataset
        The annotations dataset to split.
    group_by_var : str
        The grouping variable to use for splitting the dataset. Must be
        1-dimensional along the ``samples_coordinate``.
    list_fractions : list[float, float]
        The fractions of the input annotations dataset to allocate to
        each subset. Must contain only two elements and sum to 1.
    epsilon : float, optional
        The approximation tolerance for the subset sum as a fraction of 1.
        The sum of samples in the smallest subset is guaranteed to be at
        least ``(1 - epsilon)`` times the optimal sum for the requested
        fractions and grouping variable. When ``epsilon`` is 0, the algorithm
        finds the exact optimal sum. Larger values result in faster
        computation but may yield subsets with a total number of samples
        further from optimal. Default is 0.
    samples_coordinate : str, optional
        The coordinate along which to split the dataset. Default is
        ``image_id``.

    Returns
    -------
    tuple[xarray.Dataset, xarray.Dataset]
        The two subsets of the input dataset. The subsets are returned in the
        same order as the input list of fractions.

    Raises
    ------
    ValueError

        If the elements of ``list_fractions`` are not exactly two, are not
        between 0 and 1, or do not sum to 1. If ``group_by_var`` is not
        1-dimensional along the ``samples_coordinate``.

    Examples
    --------
    Split a dataset with a single data variable ``foo`` defined along the
    ``image_id`` dimension into an approximate 80/20 split, ensuring that the
    subsets are disjoint in the grouping variable ``foo``.

    >>> from ethology.datasets.split import _split_dataset_group_by_apss
    >>> import xarray as xr
    >>> ds = xr.Dataset(
    >>>     data_vars=dict(
    >>>         foo=("image_id", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    >>>     ),  # 0: 10 counts, 1: 2 counts
    >>>     coords=dict(
    >>>         image_id=range(12),
    >>>     ),
    >>> )
    >>> ds_subset_1, ds_subset_2 = _split_dataset_group_by_apss(
    >>>     ds,
    >>>     group_by_var="foo",
    >>>     list_fractions=[0.2, 0.8],
    >>>     epsilon=0,
    >>> )
    >>> print(len(ds_subset_1.image_id) / len(ds.image_id))  # 0.166
    >>> print(len(ds_subset_2.image_id) / len(ds.image_id))  # 0.833

    """
    # Compute number of samples in target subset
    # the target subset is the subset with the smallest fraction.
    target_subset_count = int(
        min(list_fractions) * len(dataset.get(samples_coordinate))
    )

    # Get list of (id, count) tuples
    # Count number of samples per group and sort by count in ascending order
    count_per_group_id = Counter(dataset[group_by_var].values).most_common()[
        ::-1
    ]

    # Cast group ids to integers and create mapping
    map_group_id_int_to_original = {}
    count_per_group_id_as_int = []
    for idx, (group_id, count) in enumerate(count_per_group_id):
        # try casting group ID as integer
        try:
            int_id = int(group_id)
        # if not castable: use ID from enumerate
        except (ValueError, TypeError):
            int_id = idx
        count_per_group_id_as_int.append((int_id, count))
        map_group_id_int_to_original[int_id] = group_id

    # Get group ids (as integers) for target subset
    subset_dict = _approximate_subset_sum(
        count_per_group_id_as_int,
        target_subset_count,
        epsilon=epsilon,
    )

    # Get original group IDs (they are not necessarily integers)
    subset_group_ids = [
        map_group_id_int_to_original[x] for x in subset_dict["ids"]
    ]

    # Extract datasets for target subset and not target subset
    ds_subset = dataset.isel(
        {samples_coordinate: dataset[group_by_var].isin(subset_group_ids)}
    )
    ds_not_subset = dataset.isel(
        {samples_coordinate: ~dataset[group_by_var].isin(subset_group_ids)}
    )

    # Throw warning if a subset is empty
    if any(len(ds.image_id) == 0 for ds in [ds_subset, ds_not_subset]):
        logger.warning("At least one of the subset datasets is empty.")

    # Return datasets in the same order as the input list of fractions
    # (argsort twice gives the inverse permutation)
    idcs_sorted = np.argsort(list_fractions)  # idcs to map input -> sorted
    idcs_original = np.argsort(idcs_sorted)  # idcs to map sorted -> input

    list_ds_sorted = [ds_subset, ds_not_subset]
    return tuple(list_ds_sorted[i] for i in idcs_original)


def split_dataset_random(
    dataset: xr.Dataset,
    list_fractions: list[float],
    seed: int = 42,
    samples_coordinate: str = "image_id",
) -> tuple[xr.Dataset, ...]:
    """Split an annotations dataset using random sampling.

    Split an ``ethology`` annotations dataset into multiple subsets by randomly
    shuffling all samples and then partitioning them sequentially according to
    the specified fractions.


    Parameters
    ----------
    dataset : xarray.Dataset
        The annotations dataset to split.
    list_fractions : list[float, ...]
        The fractions of the input annotations dataset to allocate to
        each subset. The list must contain at least two elements, all elements
        must be between 0 and 1, and add up to 1.
    seed : int, optional
        Seed to use for the random number generator. Default is 42.
    samples_coordinate : str, optional
        The coordinate along which to split the dataset. Default is
        ``image_id``.

    Returns
    -------
    tuple[xarray.Dataset, ...]
        The subsets of the input dataset. The subsets are returned in the
        same order as the input list of fractions.

    Raises
    ------
    ValueError
        If the elements of ``list_fractions`` are less than two, are not
        between 0 and 1, or do not sum to 1.

    Examples
    --------
    Split a dataset with a single data variable ``foo``, with 100 values
    defined along the ``image_id`` dimension into 70/20/10 splits.

    >>> from ethology.datasets.split import split_dataset_random
    >>> import numpy as np
    >>> import xarray as xr
    >>> ds = xr.Dataset(
    ...     data_vars=dict(
    ...         foo=("image_id", np.random.randint(0, 100, size=100)),
    ...     ),
    ...     coords=dict(
    ...         image_id=range(100),
    ...     ),
    ... )
    >>> ds_train, ds_val, ds_test = split_dataset_random(
    ...     ds,
    ...     list_fractions=[0.7, 0.2, 0.1],
    ...     seed=42,
    ... )
    >>> print(len(ds_train.image_id))  # 70
    >>> print(len(ds_val.image_id))  # 20
    >>> print(len(ds_test.image_id))  # 10

    Notes
    -----
    The function operates in two steps: first, it shuffles all sample indices
    along the ``samples_coordinate`` dimension using the provided random seed;
    then, it partitions the shuffled indices into contiguous blocks, one for
    each subset.

    The size of each block is determined by rounding down (floor) the product
    of the subset's fraction and the total number of samples. To ensure all
    samples are included, the last subset receives any remaining samples after
    the earlier subsets have been allocated. Due to this rounding behavior,
    the actual fraction for the last subset may differ slightly from the
    requested fraction.

    """
    # Checks
    if len(list_fractions) < 2:
        raise ValueError(
            "The list of fractions must have at least two elements."
        )

    if any(fraction < 0 or fraction > 1 for fraction in list_fractions):
        raise ValueError("The split fractions must be between 0 and 1.")

    if sum(list_fractions) != 1:
        raise ValueError("The split fractions must sum to 1.")

    # Compute number of samples for each split
    list_n_samples: list[int] = []
    n_total_samples = len(dataset.get(samples_coordinate))
    for fraction in list_fractions[:-1]:
        list_n_samples.append(int(fraction * n_total_samples))
    # append the remaining samples to the last split
    list_n_samples.append(n_total_samples - sum(list_n_samples))

    # Shuffle all indices
    rng = np.random.default_rng(seed)
    shuffled_idcs = rng.permutation(n_total_samples)

    # Extract datasets for each split
    list_ds = []
    start_idx = 0
    for n_samples in list_n_samples:
        end_idx = start_idx + n_samples
        list_ds.append(
            dataset.isel(
                {samples_coordinate: shuffled_idcs[start_idx:end_idx]}
            )
        )
        start_idx = end_idx

    # Throw warning if a subset is empty
    if any(len(ds.image_id) == 0 for ds in list_ds):
        logger.warning("At least one of the subset datasets is empty.")

    # Return subsets in the same order as the input list of fractions
    # (argsort twice gives the inverse permutation)
    idcs_sorted = np.argsort(list_fractions)  # idcs to map input -> sorted
    idcs_original = np.argsort(idcs_sorted)  # idcs to map sorted -> input
    return tuple(list_ds[i] for i in idcs_original)


class _SubsetDict(TypedDict):
    """Subset dictionary.

    Each subset dictionary is made up of a list of ``group IDs`` ("ids") and
    their total ``group count`` ("sum"). Used as a type definition for the
    approximate subset sum algorithm.

    Attributes
    ----------
    sum : int
        The total ``group count`` of the subset.
    ids : list[int]
        The list of ``group IDs`` in the subset.

    """

    sum: int
    ids: list[int]


def _approximate_subset_sum(
    list_id_counts: list[tuple[int, int]], target: int, epsilon: float
) -> _SubsetDict:
    """Find a subset of the input list whose sum is maximum but below target.

    The input is a list of pairs (``group IDs``, ``group count``). We want
    to extract a subset of elements from this list such that their total
    ``group count`` is as close as possible but does not exceed the
    ``target`` value.

    Parameters
    ----------
    list_id_counts : list[tuple[int, int]]
        The list of pairs (``group IDs``, ``group count``).
    target : int
        The target value for the total ``group count`` of the subset.
    epsilon : float
        The approximation tolerance for the subset sum as a fraction of 1.
        When ``epsilon`` is 0, the algorithm finds the optimal subset for the
        requested target value and input list. Larger values of ``epsilon``
        result in faster computation but may yield subsets with a total
        ``group count`` further from the optimal.

    Returns
    -------
    _SubsetDict
        The subset dictionary.

    Raises
    ------
    Warning
        If all groups in the input list have more samples than the target
        value. In this case, the function returns an empty subset.

    Notes
    -----
    The function uses a fully polynomial-time approximation scheme (FPTAS) to
    approximately solve this subset sum problem. When ``epsilon`` is 0, it
    finds the exact optimal subset. When ``epsilon`` > 0, it returns an
    approximate solution guaranteed to be within ``epsilon`` times the optimal
    sum from below. Using an ``epsilon`` value larger than 0 may be convenient
    in cases with a large number of subsets for faster runtime.

    The algorithm iteratively processes each element in the input list,
    maintaining a list of candidate subsets whose total ``group counts`` falls
    below the target. At each iteration, it removes near duplicate subsets
    from the list of candidates ("trimming") to prevent exponential growth,
    while ensuring the total approximation error stays within ``epsilon``.
    Two subsets are near duplicates if their total ``group count`` is
    sufficiently close.

    Note that ``epsilon`` bounds the error from below relative to the optimal
    subset sum, not the ``target``. If ``OPT`` is the best possible subset sum
    below or equal to ``target``, the result is guaranteed to be
    at least ``(1 - epsilon)*OPT``. E.g. for ``epsilon = 0.2``,
    the result is guaranteed to be at least ``0.8*OPT``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Subset_sum_problem#Fully-polynomial_time_approximation_scheme
    .. [2] https://nerderati.com/bartering-for-beers-with-approximate-subset-sums/

    """
    # Checks
    if np.min([x[1] for x in list_id_counts]) > target:
        logger.warning(
            "All groups have more samples than the target value. "
            "Returning empty subset."
        )
        return {"sum": 0, "ids": []}

    # Initialize list of candidate subsets
    list_subsets: list[_SubsetDict] = [{"sum": 0, "ids": []}]

    # Loop thru list of (id, count) pairs
    for id, count in list_id_counts:
        # Add current (id, count) pair to each candidate subset in the list
        # if the resulting subset sum is below the target.
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

        # Remove near-duplicate subsets in terms of total group count ("sum")
        # At each iteration of the loop, trimming introduces a small error.
        # The algorithm runs n iterations (one per item), so errors can
        # compound. Using ``delta = epsilon / (2n)`` ensures that when errors
        # compound over iterations, the total error stays within ``epsilon``.
        list_subsets = _remove_near_duplicate_subsets(
            list_subsets,
            delta=float(epsilon) / (2 * len(list_id_counts)),
        )

    # Return the subset with highest sum but below the target
    return list_subsets[-1]


def _remove_near_duplicate_subsets(
    list_subsets: list[_SubsetDict], delta: float
) -> list[_SubsetDict]:
    """Remove near-duplicate subsets from the list in terms of their total sum.

    Keeps only subsets whose sum is sufficiently larger than the previous
    subset sum in ascending order. When two subsets have sums within
    ``delta``% of each other, retains the smaller one (which is visited first
    after sorting).

    Parameters
    ----------
    list_subsets : list[_SubsetDict]
        The list of candidate subsets. Each subset is a dictionary with a list
        of ``group IDs`` ("ids") and their total ``group count`` ("sum").
    delta : float
        If two subsets are within ``delta``% of each other, they are considered
        near duplicates and the smaller one is removed.

    Returns
    -------
    list[_SubsetDict]
        The list of subsets after trimming.

    """
    # Ensure list of subsets is sorted by total sum, in ascending order
    list_subsets = sorted(list_subsets, key=lambda x: x["sum"])

    # Keep only subsets whose sum is delta% larger than the previous one
    list_subsets_trimmed = [
        list_subsets[0]
    ]  # always retain the zero subset; [{"sum": 0, "ids": []}]
    previous_subset_sum = 0
    for subset in list_subsets[1:]:  # do not trim zero subset
        if subset["sum"] > previous_subset_sum * (1 + delta):
            list_subsets_trimmed.append(subset)
            previous_subset_sum = subset["sum"]

    return list_subsets_trimmed
