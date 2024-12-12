# import json
from contextlib import nullcontext as does_not_raise

# import pooch
import pytest

from ethology.annotations.validators import ValidJSON

# @pytest.fixture()
# def


@pytest.mark.parametrize(
    "input_json_file, expected_exception, log_message",
    [
        ("VIA_JSON_sample_1.json", does_not_raise(), ""),
        ("VIA_JSON_sample_2.json", does_not_raise(), ""),
    ],
)
def test_valid_json(
    annotations_test_data,
    input_json_file,
    expected_exception,
    log_message,
):
    """Test the ValidJSON validator on valid data."""
    input_json_file = annotations_test_data[input_json_file]
    with expected_exception as excinfo:
        ValidJSON(input_json_file)

    if log_message:
        assert str(excinfo.value) == log_message


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
