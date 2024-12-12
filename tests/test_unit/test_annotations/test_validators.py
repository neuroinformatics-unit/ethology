# import json
from contextlib import nullcontext as does_not_raise

# import pooch
import pytest

from ethology.annotations.validators import (
    ValidJSON,
)


@pytest.mark.parametrize(
    "valid_json_file",
    [
        "VIA_JSON_sample_1.json",
        "VIA_JSON_sample_2.json",
    ],
)
def test_valid_json(valid_json_file, annotations_test_data):
    """Test the ValidJSON validator on valid data."""
    input_json_file = annotations_test_data[valid_json_file]
    with does_not_raise():
        ValidJSON(input_json_file)


# @pytest.mark.parametrize(
#     "invalid_json_file, expected_exception, log_message",
#     [
#         (
#             "invalid_VIA_JSON_sample_1.json",
#             FileNotFoundError,
#             "File not found: invalid_VIA_JSON_sample_1.json.",
#         ),
#         (
#             "invalid_VIA_JSON_sample_2.json",
#             ValueError,
#             "Error decoding JSON data from file: invalid_VIA_JSON_sample_2.",
#         ),
#     ],
# )
# def test_valid_json_errors(invalid_json_file,
# expected_exception, log_message):
#     """Test the ValidJSON validator on invalid data."""
#     with pytest.raises(expected_exception) as excinfo:
#         ValidJSON(invalid_json_file)

#     assert str(excinfo.value) == log_message
