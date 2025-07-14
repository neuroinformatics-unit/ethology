"""Example notebook showing how to export annotations to different formats."""

# %%
from pathlib import Path

from ethology.annotations.io import load_bboxes, save_bboxes

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Single file
ds = load_bboxes.from_files(
    (
        Path.home() / ".ethology-test-data/test_annotations/"
        "medium_bboxes_dataset_VIA/VIA_JSON_sample_2.json"
    ),
    format="VIA",
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export as COCO file

save_bboxes.to_COCO_file(ds, "test_coco.json")

# %%
