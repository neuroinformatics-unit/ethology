import json

import pytest

from ethology.annotations.io.load_bboxes import from_files
from ethology.annotations.io.save_bboxes import df_bboxes_to_COCO_file


@pytest.mark.parametrize(
    "filename",
    [
        "small_bboxes_COCO.json",
        # "small_bboxes_duplicates_COCO.json",
    ],
)
def test_df_bboxes_to_COCO_file(filename, annotations_test_data, tmp_path):
    # Get input JSON file
    input_file = annotations_test_data[filename]

    # Read as bboxes dataframe
    df = from_files(input_file, format="COCO")

    # Export dataframe to COCO format
    output_file = df_bboxes_to_COCO_file(
        df, output_filepath=tmp_path / "output.json"
    )

    ########################################
    # Compare original and exported JSON files
    with open(input_file) as file:
        original_data = json.load(file)

    with open(output_file) as file:
        exported_data = json.load(file)

    ########################################
    # Check categories dictionaries match
    for categories_original, categories_exported in zip(
        original_data["categories"], exported_data["categories"], strict=True
    ):
        assert categories_original["id"] - 1 == categories_exported["id"]
        assert categories_original["name"] == categories_exported["name"]
        assert (
            categories_original["supercategory"]
            == categories_exported["supercategory"]
        )

    ########################################
    # Check images dictionaries match
    # Sort dicts in list of image dictionaries based on file_name
    original_img_dicts_sorted = sorted(
        original_data["images"], key=lambda x: x["file_name"]
    )
    exported_img_dicts_sorted = sorted(
        exported_data["images"], key=lambda x: x["file_name"]
    )

    # Compare the two lists of dictionaries
    for im_original, im_exported in zip(
        original_img_dicts_sorted,
        exported_img_dicts_sorted,
        strict=True,
    ):
        assert all(im_exported[ky] == im_original[ky] for ky in im_exported)

    ########################################
    # Check annotations
    # Sort annotations based on id
    exported_annot_dicts_sorted = sorted(
        exported_data["annotations"], key=lambda x: x["id"]
    )
    original_annot_dicts_sorted = sorted(
        original_data["annotations"], key=lambda x: x["id"]
    )

    # Compare the two lists of dictionaries
    for annot_original, annot_exported in zip(
        original_annot_dicts_sorted,
        exported_annot_dicts_sorted,
        strict=True,
    ):
        assert all(
            annot_exported[ky] == annot_original[ky]
            for ky in annot_exported
            if ky not in ["id", "category_id"]
            # id is expected to differ because we reindex
        )
        assert (
            annot_exported["category_id"] == annot_original["category_id"] - 1
        )
