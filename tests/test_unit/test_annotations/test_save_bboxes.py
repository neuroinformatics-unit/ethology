import filecmp

import pytest

from ethology.annotations.load_bboxes import df_bboxes_from_files
from ethology.annotations.save_bboxes import df_bboxes_to_COCO_file


@pytest.mark.parametrize(
    "input_format, filename",
    [
        ("VIA", "small_bboxes_duplicates_VIA.json"),
        # ("COCO", "small_bboxes_duplicates_COCO.json"),
    ],
)
def test_df_bboxes_to_COCO_file(
    input_format, filename, annotations_test_data, tmp_path
):
    # Get input JSON file
    input_file = annotations_test_data[filename]

    # Read dataframe
    df = df_bboxes_from_files(input_file, format=input_format)

    # Export to COCO format
    output_file = df_bboxes_to_COCO_file(
        df, output_dir=tmp_path / "output.json"
    )

    # Check input and output files are identical
    assert filecmp.cmp(input_file, output_file, shallow=False)
