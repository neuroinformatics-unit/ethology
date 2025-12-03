import pytest
from attrs import define

from ethology.validators.utils import ValidDataset


@pytest.mark.parametrize(
    "missing_attr, expected_error_match",
    [
        (
            "required_dims",
            ".*must define 'required_dims' class variable",
        ),
        (
            "required_data_vars",
            ".*must define 'required_data_vars' class variable",
        ),
        (
            "both",
            ".*must define 'required_dims' class variable",
        ),
    ],
    ids=[
        "missing_required_dims",
        "missing_required_data_vars",
        "missing_both_class_vars",
    ],
)
def test_subclass_missing_class_vars_raises_type_error(
    missing_attr, expected_error_match
):
    """Test that subclasses without required class vars raise TypeError."""
    with pytest.raises(TypeError, match=expected_error_match):
        if missing_attr == "required_dims":

            @define
            class InvalidDataset(ValidDataset):
                required_data_vars = {"position": {"x", "y"}}

        elif missing_attr == "required_data_vars":

            @define
            class InvalidDataset(ValidDataset):
                required_dims = {"x", "y"}

        else:

            @define
            class InvalidDataset(ValidDataset):
                pass


def test_subclass_with_both_class_vars_does_not_raise():
    """Test that a valid subclass with both class vars works correctly."""
    required_dims_in = {"x", "y"}
    required_data_vars_in = {"position": {"x", "y"}}

    @define
    class ValidCustomDataset(ValidDataset):
        required_dims = required_dims_in
        required_data_vars = required_data_vars_in

    # Verify the class attributes
    assert ValidCustomDataset.required_dims == required_dims_in
    assert ValidCustomDataset.required_data_vars == required_data_vars_in
