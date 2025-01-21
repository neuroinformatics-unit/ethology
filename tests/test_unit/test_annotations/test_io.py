from pathlib import Path
from unittest.mock import patch

import pytest

from ethology.annotations.io import _df_bboxes_from_single_file


@pytest.mark.parametrize(
    "input_format, function_to_mock, no_error_expected",
    [
        (
            "VIA",
            "ethology.annotations.io._df_bboxes_from_single_VIA_file",
            True,
        ),
        (
            "COCO",
            "ethology.annotations.io._df_bboxes_from_single_COCO_file",
            True,
        ),
        (
            "unsupported",
            None,
            False,
        ),
    ],
)
def test_df_bboxes_from_single_file(
    input_format, function_to_mock, no_error_expected
):
    """Test that the function delegates to the correct function."""
    file_path = Path("/mock/path/to/file")

    if no_error_expected:
        with patch(function_to_mock) as mock:
            _df_bboxes_from_single_file(file_path, input_format)
            mock.assert_called_once_with(file_path)
    else:
        with pytest.raises(ValueError) as excinfo:
            _df_bboxes_from_single_file(file_path, input_format)
        assert "Unsupported format" in str(excinfo.value)


def test_df_bboxes_from_single_VIA_file():
    pass


def test_df_bboxes_from_single_COCO_file():
    pass


def test_df_bboxes_from_file():
    pass
