import logging
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from ethology.datasets.split import (
    _approximate_subset_sum,
    _split_dataset_group_by_apss,
    _split_dataset_group_by_kfold,
    split_dataset_group_by,
    split_dataset_random,
)


def split_at_any_delimiter(text: str, delimiters: list[str]) -> list[str]:
    """Split a string at any of the specified delimiters if present."""
    for delimiter in delimiters:
        if delimiter in text:
            return text.split(delimiter)
    return [text]


@pytest.fixture
def valid_bbox_annotations_ds_to_split_1(valid_bbox_annotations_dataset):
    # We add a `foo` variable to the dataset that is
    # one-dimensional along the `image_id` dimension to
    # use for grouping by.
    # Note: len(valid_bboxes_dataset.image_id) = 3
    ds = valid_bbox_annotations_dataset.copy(deep=True)
    ds["foo"] = (
        ["image_id"],
        np.array([0, 1, 1]),
    )
    return ds


@pytest.fixture
def valid_bbox_annotations_ds_to_split_2(valid_bbox_annotations_dataset):
    # We add a `foo` variable to the dataset that is
    # one-dimensional along the `image_id` dimension to
    # use for grouping by. In this case we ensure we have
    # 3 groups to be able to split using 3 folds (with
    # GroupKFold we cannot have more folds than groups).
    # Note: len(valid_bboxes_dataset.image_id) = 3
    ds = valid_bbox_annotations_dataset.copy(deep=True)
    ds["foo"] = (
        ["image_id"],
        np.array([0, 1, 2]),
    )
    return ds


@pytest.fixture
def ACTD_dataset_to_split(annotations_test_data):
    from ethology.io.annotations import load_bboxes

    # load dataset
    input_file_path = annotations_test_data[
        "ACTD_1_Terrestrial_group_data_CCT.json"
    ]
    ds = load_bboxes.from_files(input_file_path, format="COCO")

    # Get species per image
    species_per_image_id = np.array(
        [
            ds.map_image_id_to_filename[i].split("\\")[-2]
            for i in ds.image_id.values
        ]
    )
    assert species_per_image_id.shape[0] == len(ds.image_id)

    # Add to dataset as "foo" variable
    ds["foo"] = xr.DataArray(species_per_image_id, dims="image_id")
    return ds


@pytest.mark.parametrize(
    "inputs, expected_subset_dict",
    [
        (
            {
                "list_id_counts": [(0, 10), (1, 3), (2, 5)],
                "target": 1,
                "epsilon": 0,
            },
            {
                "ids": [],
                "sum": 0,
            },
        ),  # all counts above target, returns empty subset
        (
            {
                "list_id_counts": [(0, 10), (1, 3), (2, 5)],
                "target": 3,
                "epsilon": 0,
            },
            {
                "ids": [1],
                "sum": 3,
            },
        ),  # one count equal to target, returns that subset
        (
            {
                "list_id_counts": [(0, 10), (1, 3), (2, 5)],
                "target": 8,
                "epsilon": 0,
            },
            {
                "ids": [1, 2],
                "sum": 8,
            },
        ),  # epsilon 0, integer group IDs
        (
            {
                "list_id_counts": [("a", 10), ("b", 3), ("c", 5)],
                "target": 8,
                "epsilon": 0,
            },
            {
                "ids": ["b", "c"],
                "sum": 8,
            },
        ),  # epsilon 0, non-integer group IDs
        (
            {
                "list_id_counts": [(0, 10), (1, 3), (2, 5)],
                "target": 14,
                "epsilon": 0.1,
            },
            {
                "ids": [
                    0,
                    1,
                ],
                "sum": 13,
            },
        ),  # epsilon 0.1
    ],
)
def test_approximate_subset_sum(inputs, expected_subset_dict):
    subset_dict = _approximate_subset_sum(**inputs)

    assert subset_dict["ids"] == expected_subset_dict["ids"]
    assert subset_dict["sum"] == expected_subset_dict["sum"]


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "dataset": "valid_bbox_annotations_ds_to_split_1",
            "list_fractions": [0.334, 0.666],
            "samples_coordinate": "image_id",
        },  # fractions in increasing order
        {
            "dataset": "valid_bbox_annotations_ds_to_split_1",
            "list_fractions": [0.666, 0.334],
            "samples_coordinate": "image_id",
        },  # fractions in decreasing order
        {
            "dataset": "ACTD_dataset_to_split",
            "list_fractions": [0.13, 0.87],
            "samples_coordinate": "image_id",
        },  # realistic dataset
    ],
)
def test_split_dataset_group_by_apss(inputs, request):
    # prepare inputs
    dataset = request.getfixturevalue(inputs["dataset"])
    inputs["dataset"] = dataset

    # we split datasets grouping by the `foo` variable
    group_by_var = "foo"

    # split dataset
    ds_subset_1, ds_subset_2 = _split_dataset_group_by_apss(
        **inputs,
        group_by_var=group_by_var,
        epsilon=0,
    )

    # assert dataset sizes
    list_input_fractions = inputs["list_fractions"]
    total_n_images = len(inputs["dataset"].image_id)
    fraction_subset_1 = len(ds_subset_1.image_id) / total_n_images
    fraction_subset_2 = len(ds_subset_2.image_id) / total_n_images
    assert fraction_subset_1 == pytest.approx(
        list_input_fractions[0], abs=0.005
    )
    assert fraction_subset_2 == pytest.approx(
        list_input_fractions[1], abs=0.005
    )

    # assert that the subsets are disjoint in the grouping variable
    assert (
        set.intersection(
            set(ds_subset_1[group_by_var].values),
            set(ds_subset_2[group_by_var].values),
        )
        == set()
    )


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "dataset": "valid_bbox_annotations_ds_to_split_2",
            "list_fractions": [0.334, 0.666],
            "samples_coordinate": "image_id",
        },  # fractions in increasing order
        {
            "dataset": "valid_bbox_annotations_ds_to_split_2",
            "list_fractions": [0.666, 0.334],
            "samples_coordinate": "image_id",
        },  # fractions in decreasing order
        {
            "dataset": "ACTD_dataset_to_split",
            "list_fractions": [0.13, 0.87],
            "samples_coordinate": "image_id",
        },  # realistic dataset
    ],
)
def test_split_dataset_group_by_kfold(inputs, request):
    # prepare inputs
    dataset = request.getfixturevalue(inputs["dataset"])
    inputs["dataset"] = dataset

    # we split datasets grouping by the `foo` variable
    group_by_var = "foo"

    # split dataset
    ds_subset_1, ds_subset_2 = _split_dataset_group_by_kfold(
        **inputs, group_by_var=group_by_var
    )

    # assert dataset sizes
    list_input_fractions = inputs["list_fractions"]
    total_n_images = len(inputs["dataset"].image_id)
    fraction_subset_1 = len(ds_subset_1.image_id) / total_n_images
    fraction_subset_2 = len(ds_subset_2.image_id) / total_n_images
    assert fraction_subset_1 == pytest.approx(
        list_input_fractions[0], abs=0.01
    )
    assert fraction_subset_2 == pytest.approx(
        list_input_fractions[1], abs=0.01
    )

    # assert that the subsets are disjoint in the grouping variable
    assert (
        set.intersection(
            set(ds_subset_1[group_by_var].values),
            set(ds_subset_2[group_by_var].values),
        )
        == set()
    )


def test_split_dataset_group_by_kfold_seed(valid_bbox_annotations_ds_to_split_2):
    """Test the behaviour of the seed when using the `kfold` method."""
    # prepare inputs
    dataset = valid_bbox_annotations_ds_to_split_2
    list_fractions = [0.334, 0.666]
    samples_coordinate = "image_id"
    group_by_var = "foo"
    seed = 42

    # split dataset using seed 42
    ds_subset_1, ds_subset_2 = _split_dataset_group_by_kfold(
        dataset=dataset,
        list_fractions=list_fractions,
        samples_coordinate=samples_coordinate,
        seed=seed,
        group_by_var=group_by_var,
    )

    # assert same seed gives the same subsets
    ds_subset_1_2, ds_subset_2_2 = _split_dataset_group_by_kfold(
        dataset=dataset,
        list_fractions=list_fractions,
        samples_coordinate=samples_coordinate,
        seed=seed,
        group_by_var=group_by_var,
    )
    assert ds_subset_1.equals(ds_subset_1_2)
    assert ds_subset_2.equals(ds_subset_2_2)

    # assert different seeds gives different subsets
    ds_subset_1_3, ds_subset_2_3 = _split_dataset_group_by_kfold(
        dataset=dataset,
        list_fractions=list_fractions,
        samples_coordinate=samples_coordinate,
        seed=seed + 1,
        group_by_var=group_by_var,
    )
    assert not ds_subset_1.equals(ds_subset_1_3)
    assert not ds_subset_2.equals(ds_subset_2_3)


@pytest.mark.parametrize(
    "method, function_to_mock",
    [
        (
            "kfold",
            "ethology.datasets.split._split_dataset_group_by_kfold",
        ),
        (
            "apss",
            "ethology.datasets.split._split_dataset_group_by_apss",
        ),
    ],
)
def test_split_dataset_group_by(
    method, function_to_mock, valid_bbox_annotations_ds_to_split_1
):
    """Test the wrapper function dispatches to the appropriate method."""
    # Create mock return datasets
    mock_return_value = (xr.Dataset(), xr.Dataset())

    # Patch the internal function and call the wrapper
    with patch(function_to_mock, return_value=mock_return_value) as mock:
        _ds_subset_1, _ds_subset_2 = split_dataset_group_by(
            dataset=valid_bbox_annotations_ds_to_split_1,
            group_by_var="foo",
            list_fractions=[0.334, 0.666],
            samples_coordinate="image_id",
            method=method,
        )

        # Verify the correct internal function was called once
        mock.assert_called_once()


@pytest.mark.parametrize(
    "dataset, expected_method",
    [
        ("valid_bbox_annotations_ds_to_split_1", "apss"),
        ("valid_bbox_annotations_ds_to_split_2", "kfold"),
    ],
)
def test_split_dataset_group_by_auto(dataset, expected_method, request):
    """Test the automatic selection of the method."""
    dataset = request.getfixturevalue(dataset)

    # mock the internal function for the expected method
    function_to_mock = (
        f"ethology.datasets.split._split_dataset_group_by_{expected_method}"
    )
    mock_return_value = (xr.Dataset(), xr.Dataset())

    # split dataset
    with patch(function_to_mock, return_value=mock_return_value) as mock:
        _ds_subset_1, _ds_subset_2 = split_dataset_group_by(
            dataset=dataset,
            list_fractions=[0.334, 0.666],
            samples_coordinate="image_id",
            group_by_var="foo",
            method="auto",
        )

        # Verify the correct internal function was called once
        mock.assert_called_once()


def test_split_dataset_group_by_unknown_method(
    valid_bbox_annotations_ds_to_split_1,
):
    """Test that an unknown method raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown method"):
        split_dataset_group_by(
            dataset=valid_bbox_annotations_ds_to_split_1,
            group_by_var="foo",
            list_fractions=[0.5, 0.5],
            method="unknown_method",
        )


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "dataset": "valid_bbox_annotations_ds_to_split_1",
            "list_fractions": [0.334, 0.666],
            "samples_coordinate": "image_id",
        },  # fractions in increasing order
        {
            "dataset": "valid_bbox_annotations_ds_to_split_1",
            "list_fractions": [0.666, 0.334],
            "samples_coordinate": "image_id",
        },  # fractions in decreasing order
        {
            "dataset": "valid_bbox_annotations_ds_to_split_1",
            "list_fractions": [1 / 3, 1 / 3, 1 / 3],
            "samples_coordinate": "image_id",
        },  # more than two fractions
        {
            "dataset": "ACTD_dataset_to_split",
            "list_fractions": [0.13, 0.87],
            "samples_coordinate": "image_id",
        },  # realistic dataset
    ],
)
def test_split_dataset_random(inputs, request):
    # prepare inputs
    # override dataset in inputs
    dataset = request.getfixturevalue(inputs["dataset"])
    inputs["dataset"] = dataset

    # split dataset
    ds_subsets = split_dataset_random(
        **inputs,
    )

    # assert dataset sizes
    list_expected_fractions = inputs["list_fractions"]
    total_n_images = len(inputs["dataset"].image_id)
    list_output_fractions = [
        len(ds.image_id) / total_n_images for ds in ds_subsets
    ]
    assert all(
        fraction == pytest.approx(list_expected_fractions[i], abs=0.005)
        for i, fraction in enumerate(list_output_fractions)
    )

    # Indices per split should be exclusive
    list_output_idcs_per_split = [
        ds.image_id.values.tolist() for ds in ds_subsets
    ]
    assert (
        set.intersection(
            *[
                set(idcs_in_split)
                for idcs_in_split in list_output_idcs_per_split
            ]
        )
        == set()
    )


@pytest.mark.parametrize(
    "method, dataset,expected_log_fragment",
    [
        (
            "auto",
            "valid_bbox_annotations_ds_to_split_1",
            # dataset that will trigger auto-selection of apss
            # with the requested fractions 0.334 and 0.666
            "Auto-selected approximate subset-sum method",
        ),
        (
            "auto",
            "valid_bbox_annotations_ds_to_split_2",
            # dataset with 3 groups so kfold method can be used
            "Using group k-fold method with",
        ),
        (
            "kfold",
            "valid_bbox_annotations_ds_to_split_2",
            # dataset with 3 groups so kfold method can be used
            "Using group k-fold method with",
        ),
        (
            "apss",
            "valid_bbox_annotations_ds_to_split_2",
            # dataset with 3 groups so apss method can be used
            "Using approximate subset-sum method with",
        ),
    ],
    ids=["auto-apss", "auto-kfold", "kfold", "apss"],
)
def test_split_dataset_group_by_logger_info(
    caplog, method, dataset, expected_log_fragment, request
):
    """Test that info messages are emitted for the used methods."""
    # Use dataset with 3 groups so kfold method can be used
    ds = request.getfixturevalue(dataset)
    seed_value = 42
    epsilon_value = 0.1

    with caplog.at_level(logging.INFO):
        split_dataset_group_by(
            dataset=ds,
            group_by_var="foo",
            list_fractions=[0.334, 0.666],
            samples_coordinate="image_id",
            method=method,
            seed=seed_value,
            epsilon=epsilon_value,
        )

    # Check that the expected log message was logged
    assert expected_log_fragment in caplog.text

    # Check that the seed value appears in the log if required
    test_id = request.node.callspec.id
    if "kfold" in test_id:
        assert f"seed={seed_value}" in caplog.text

    # Check that the epsilon value appears in the log if required
    if "apss" in test_id:
        assert f"epsilon={epsilon_value}" in caplog.text


@pytest.mark.parametrize(
    "inputs, expected_error_message",
    [
        (
            {
                "dataset": xr.Dataset(),
                "list_fractions": [0.2, 0.2],
                "samples_coordinate": "image_id",
            },
            "The split fractions must sum to 1.",
        ),
        (
            {
                "dataset": xr.Dataset(),
                "list_fractions": [0.2, 0.4, 0.4],
                "samples_coordinate": "image_id",
            },
            "The list of fractions must have only two elements.",
        ),  # more than two fractions
        (
            {
                "dataset": xr.Dataset(),
                "list_fractions": [1],
                "samples_coordinate": "image_id",
            },
            "The list of fractions must have only two elements.",
        ),  # less than two fractions
        (
            {
                "dataset": xr.Dataset(),
                "list_fractions": [1.2, -0.2],
                "samples_coordinate": "image_id",
            },
            "The split fractions must be between 0 and 1.",
        ),
        (
            {
                "dataset": xr.Dataset(
                    data_vars=dict(
                        foo=(
                            ["image_id", "space"],
                            np.zeros((100, 2)),
                        )
                    ),
                    coords=dict(
                        image_id=range(100),
                        space=["x", "y"],
                    ),
                ),
                "list_fractions": [0.2, 0.8],
                "samples_coordinate": "image_id",
            },
            "The grouping variable foo must be 1-dimensional along image_id.",
        ),
    ],
)
@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {"epsilon": 0},
        {},
    ],
)
def test_split_dataset_group_by_error(
    inputs, extra_kwargs, expected_error_message
):
    with pytest.raises(ValueError) as e:
        _ds_subset_1, _ds_subset_2 = split_dataset_group_by(
            **inputs, group_by_var="foo", **extra_kwargs
        )
    assert str(e.value) in expected_error_message


@pytest.mark.parametrize(
    "inputs, expected_error_message",
    [
        (
            {
                "dataset": xr.Dataset(),
                "list_fractions": [0.2, 0.2],
                "samples_coordinate": "image_id",
            },
            "The split fractions must sum to 1.",
        ),
        (
            {
                "dataset": xr.Dataset(),
                "list_fractions": [1.2, -0.2],
                "samples_coordinate": "image_id",
            },
            "The split fractions must be between 0 and 1.",
        ),
        (
            {
                "dataset": xr.Dataset(),
                "list_fractions": [0.1],
                "samples_coordinate": "image_id",
            },
            "The list of fractions must have at least two elements.",
        ),
    ],
)
def test_split_dataset_random_error(inputs, expected_error_message):
    with pytest.raises(ValueError) as e:
        _ds_subsets = split_dataset_random(**inputs)

    assert str(e.value) in expected_error_message


@pytest.mark.parametrize(
    "split_function, inputs",
    [
        (
            split_dataset_random,
            {
                "list_fractions": [0.9, 0.05, 0.05],
                "samples_coordinate": "image_id",
            },
        ),
        (
            split_dataset_group_by,
            {
                "list_fractions": [0.9, 0.1],
                "samples_coordinate": "image_id",
                "group_by_var": "foo",
                "epsilon": 0,
            },
            # should defer to the `apss` method
        ),
    ],
)
def test_split_dataset_warning_empty_subset(
    caplog, request, split_function, inputs
):
    """Test that a warning is thrown when at least one subset is empty."""
    # Get dataset to split
    ds = request.getfixturevalue("valid_bbox_annotations_ds_to_split_1")
    inputs["dataset"] = ds

    # We use fractions that will cause an empty subset
    with caplog.at_level(logging.WARNING):
        ds_subsets = split_function(**inputs)

    # Verify at least one subset is empty
    assert any(len(subset.image_id) == 0 for subset in ds_subsets)

    # Check that the warning was logged
    assert "At least one of the subset datasets is empty." in caplog.text
