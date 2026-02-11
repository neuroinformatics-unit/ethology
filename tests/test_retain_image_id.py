import json
from pathlib import Path

from ethology.io.annotations.load_bboxes import from_files


def test_retain_image_id_coco(tmp_path: Path):
    # Create a small COCO-like json
    js = {
        "images": [
            {"id": 42, "file_name": "imgA.jpg", "width": 100, "height": 200},
            {"id": 99, "file_name": "imgB.jpg", "width": 100, "height": 200},
        ],
        "annotations": [
            {"id": 1, "image_id": 42, "bbox": [1, 2, 3, 4], "category_id": 1},
            {"id": 2, "image_id": 99, "bbox": [5, 6, 7, 8], "category_id": 1},
        ],
        "categories": [{"id": 1, "name": "cat", "supercategory": ""}],
    }
    p = tmp_path / "coco_test.json"
    p.write_text(json.dumps(js), encoding="utf-8")

    ds = from_files(str(p), format="COCO", retain_image_id=True)
    # ds.coords['image_id'] should contain the original COCO ids 42 and 99.
    mapping = ds.attrs["map_image_id_to_original"]
    assert set(mapping.values()) == {42, 99}
