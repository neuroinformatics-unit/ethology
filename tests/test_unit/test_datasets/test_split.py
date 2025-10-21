import numpy as np
import pytest
import xarray as xr

from ethology.datasets.split import (
    _approximate_subset_sum,
    split_dataset_group_by,
)


# Helper function
def split_at_any_delimiter(text: str, delimiters: list[str]) -> list[str]:
    """Split a string at any of the specified delimiters if present."""
    for delimiter in delimiters:
        if delimiter in text:
            return text.split(delimiter)
    return [text]


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


@pytest.fixture
def valid_bboxes_dataset_to_split(valid_bboxes_dataset):
    # Note: len(valid_bboxes_dataset.image_id) = 3
    ds = valid_bboxes_dataset.copy(deep=True)
    ds["foo"] = (
        ["image_id"],
        np.array([0, 1, 1]),
    )
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


@pytest.mark.parametrize(
    "inputs",
    [
        # {
        #     "dataset": "valid_bboxes_dataset_to_split",
        #     "list_fractions": [1/3, 2/3],
        #     "samples_coordinate": "image_id",
        # },  # fractions in increasing order
        # {
        #     "dataset": "valid_bboxes_dataset_to_split",
        #     "list_fractions": [0.8, 0.2],
        #     "samples_coordinate": "image_id",
        # },  # fractions in decreasing order
        {
            "dataset": "ACTD_dataset_to_split",
            "list_fractions": [0.13, 0.87],
            "samples_coordinate": "image_id",
        },  # realistic dataset
    ],
)
def test_split_annotations_dataset_group_by(inputs, request):
    # prepare inputs
    # override dataset in inputs
    dataset = request.getfixturevalue(inputs["dataset"])
    inputs["dataset"] = dataset
    group_by_var = "foo"

    # split dataset
    ds_subset_1, ds_subset_2 = split_dataset_group_by(
        **inputs,
        group_by_var=group_by_var,
        epsilon=0,
    )

    # assert dataset sizes
    list_fractions = inputs["list_fractions"]
    total_n_images = len(inputs["dataset"].image_id)
    fraction_subset_1 = len(ds_subset_1.image_id) / total_n_images
    fraction_subset_2 = len(ds_subset_2.image_id) / total_n_images
    assert fraction_subset_1 == pytest.approx(list_fractions[0], abs=0.005)
    assert fraction_subset_2 == pytest.approx(list_fractions[1], abs=0.005)

    # assert that the subsets are disjoint in the grouping variable
    assert (
        set.intersection(
            set(ds_subset_1[group_by_var].values),
            set(ds_subset_2[group_by_var].values),
        )
        == set()
    )


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
        _ds_subset_1, _ds_subset_2 = split_dataset_group_by(
            **inputs,
            group_by_var="foo",
            epsilon=0,
        )
    assert str(e.value) in expected_error_message


def test_split_annotations_dataset_random():
    pass
    # # Indices per split should be exclusive
    # assert (
    #     set.intersection(
    #         *[set(list_idcs) for list_idcs in list_idcs_per_split]
    #     )
    #     == set()
    # )
