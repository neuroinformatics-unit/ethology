import numpy as np
import pytest
import xarray as xr

from ethology.detectors.datasets import (
    _approximate_subset_sum,
    split_annotations_dataset_group_by,
)


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
        ),  # epsilon 0
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


# @pytest.mark.parametrize(
#     "inputs",
#     [
#         {
#             "dataset": dataset,
#             "group_by_var": group_by_var,
#             "list_fractions": [0.2, 0.8],
#             "samples_coordinate": "image_id",
#         },  # fractions in increasing order
#         {
#             "dataset": dataset,
#             "group_by_var": group_by_var,
#             "list_fractions": [0.8, 0.2],
#             "samples_coordinate": "image_id",
#         },  # fractions in decreasing order
#     ],
# )
# def test_split_annotations_dataset_group_by(inputs):
#     # mock xarray dataset
#     ds_subset_1, ds_subset_2 = split_annotations_dataset_group_by(
#         dataset=xr.Dataset(
#             vars={"foo": xr.DataArray(range(100))},
#             coords={"image_id": range(100)},
#         ),
#         group_by_var="foo",
#         **inputs,
#         epsilon=0,
#     )

#     # assert dataset sizes
#     list_fractions = inputs["list_fractions"]
#     assert len(ds_subset_1) == list_fractions[0] * len(inputs["dataset"])
#     assert len(ds_subset_2) == list_fractions[1] * len(inputs["dataset"])

#     # assert that the subsets are disjoint in the grouping variable
#     assert (
#         set.intersection(
#             set(ds_subset_1[inputs["group_by_var"]].values),
#             set(ds_subset_2[inputs["group_by_var"]].values),
#         )
#         == set()
#     )


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
def test_split_annotations_dataset_group_by_error(
    inputs, expected_error_message
):
    with pytest.raises(ValueError) as e:
        _ds_subset_1, _ds_subset_2 = split_annotations_dataset_group_by(
            **inputs,
            group_by_var="foo",
            epsilon=0,
        )
    assert str(e.value) in expected_error_message


def test_split_annotations_dataset_random():
    pass
