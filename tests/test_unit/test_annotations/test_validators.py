# import json
from contextlib import nullcontext as does_not_raise

# import pooch
import pytest

from ethology.annotations.validators import ValidJSON


@pytest.fixture()
def via_json_valid_file(annotations_test_data):
    return annotations_test_data["VIA_JSON_sample_1.json"]


@pytest.fixture()
def coco_json_valid_file(annotations_test_data):
    return annotations_test_data["COCO_JSON_sample_1.json"]


@pytest.fixture()
def json_with_decode_error(tmp_path):
    """Return the path to a JSON file with a decoding error."""
    json_file = tmp_path / "JSON_decode_error.json"
    with open(json_file, "w") as f:
        f.write("invalid_json")
    return json_file


@pytest.fixture()
def json_file_not_found(tmp_path):
    """Return the path to a JSON file that does not exist."""
    return tmp_path / "JSON_file_not_found.json"


@pytest.mark.parametrize(
    "input_json_file, expected_exception, log_message",
    [
        (
            "via_json_valid_file",
            does_not_raise(),
            "",
        ),
        (
            "coco_json_valid_file",
            does_not_raise(),
            "",
        ),
        (
            "json_with_decode_error",
            pytest.raises(ValueError),
            "Error decoding JSON data from file: {}.",
        ),
        (
            "json_file_not_found",
            pytest.raises(FileNotFoundError),
            "File not found: {}.",
        ),
    ],
)
def test_valid_json(
    annotations_test_data,
    input_json_file,
    expected_exception,
    log_message,
    request,
):
    """Test the ValidJSON validator."""
    input_json_file = request.getfixturevalue(input_json_file)
    with expected_exception as excinfo:
        ValidJSON(input_json_file)

    if log_message:
        assert str(excinfo.value) == log_message.format(input_json_file)


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
