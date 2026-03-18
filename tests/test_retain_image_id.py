"""Tests for the retain_image_id feature in load_bboxes."""

import json
from pathlib import Path

from ethology.io.annotations.load_bboxes import (
    _compute_filename_to_original_id,
    _compute_filename_to_original_id_coco,
    _compute_filename_to_original_id_via,
    from_files,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_coco(tmp_path: Path, name: str = "coco.json") -> Path:
    data = {
        "images": [
            {
                "id": 42,
                "file_name": "imgA.jpg",
                "width": 100,
                "height": 200,
            },
            {
                "id": 99,
                "file_name": "imgB.jpg",
                "width": 100,
                "height": 200,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 42,
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "category_id": 1,
            },
            {
                "id": 2,
                "image_id": 99,
                "bbox": [5.0, 6.0, 7.0, 8.0],
                "category_id": 1,
            },
        ],
        "categories": [{"id": 1, "name": "cat", "supercategory": "animal"}],
    }
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_via(
    tmp_path: Path,
    img_keys: list[tuple[str, str]] | None = None,
    name: str = "via.json",
) -> Path:
    if img_keys is None:
        img_keys = [("10", "imgA.jpg"), ("20", "imgB.jpg")]
    metadata = {
        key: {
            "filename": fname,
            "file_attributes": {"width": 100, "height": 200},
            "regions": [
                {
                    "shape_attributes": {
                        "name": "rect",
                        "x": 1,
                        "y": 2,
                        "width": 3,
                        "height": 4,
                    },
                    "region_attributes": {"animal": "1"},
                }
            ],
        }
        for key, fname in img_keys
    }
    data = {
        "_via_img_metadata": metadata,
        "_via_attributes": {
            "file": {},
            "region": {
                "animal": {
                    "type": "dropdown",
                    "options": {"1": "cat"},
                }
            },
        },
        "_via_settings": {},
    }
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# COCO tests
# ---------------------------------------------------------------------------


class TestRetainImageIdCOCO:
    """Tests for retain_image_id=True with COCO-format files."""

    def test_original_ids_preserved(self, tmp_path: Path) -> None:
        """COCO image IDs (42, 99) are preserved in the dataset."""
        p = _write_coco(tmp_path)
        ds = from_files(str(p), format="COCO", retain_image_id=True)
        mapping = ds.attrs["map_image_id_to_original"]
        assert set(mapping.values()) == {42, 99}

    def test_default_behaviour_unchanged(self, tmp_path: Path) -> None:
        """retain_image_id=False keeps 0-based ethology IDs."""
        p = _write_coco(tmp_path)
        ds = from_files(str(p), format="COCO", retain_image_id=False)
        assert set(ds.coords["image_id"].values) == {0, 1}
        assert ds.attrs["map_image_id_to_original"] == {}

    def test_nonexistent_file_is_skipped(self, tmp_path: Path) -> None:
        """A path that does not exist is silently skipped."""
        result = _compute_filename_to_original_id_coco(
            [tmp_path / "ghost.json"]
        )
        assert result == {}

    def test_invalid_json_is_skipped(self, tmp_path: Path) -> None:
        """A non-JSON file is silently skipped."""
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json }{", encoding="utf-8")
        result = _compute_filename_to_original_id_coco([bad])
        assert result == {}


# ---------------------------------------------------------------------------
# VIA tests
# ---------------------------------------------------------------------------


class TestRetainImageIdVIA:
    """Tests for retain_image_id=True with VIA-format files."""

    def test_numeric_keys_preserved(self, tmp_path: Path) -> None:
        """VIA metadata keys that are numeric ints are preserved."""
        p = _write_via(
            tmp_path,
            img_keys=[("10", "imgA.jpg"), ("20", "imgB.jpg")],
        )
        ds = from_files(str(p), format="VIA", retain_image_id=True)
        mapping = ds.attrs["map_image_id_to_original"]
        assert set(mapping.values()) == {10, 20}

    def test_non_numeric_keys_ignored(self, tmp_path: Path) -> None:
        """VIA keys that cannot be cast to int are silently skipped."""
        data = {
            "_via_img_metadata": {
                "some_string_key": {
                    "filename": "imgA.jpg",
                    "file_attributes": {},
                    "regions": [],
                }
            },
            "_via_attributes": {},
        }
        p = tmp_path / "via_nonnumeric.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = _compute_filename_to_original_id_via([p])
        assert result == {}

    def test_nonexistent_file_is_skipped(self, tmp_path: Path) -> None:
        """A path that does not exist is silently skipped (VIA)."""
        result = _compute_filename_to_original_id_via(
            [tmp_path / "ghost.json"]
        )
        assert result == {}

    def test_invalid_json_is_skipped(self, tmp_path: Path) -> None:
        """A non-JSON file is silently skipped (VIA)."""
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json }{", encoding="utf-8")
        result = _compute_filename_to_original_id_via([bad])
        assert result == {}


# ---------------------------------------------------------------------------
# Dispatcher fallback
# ---------------------------------------------------------------------------


class TestDispatcher:
    """Tests for the _compute_filename_to_original_id dispatcher."""

    def test_unknown_format_returns_empty_dict(self) -> None:
        """The fallback return {} fires for an unknown format."""
        result = _compute_filename_to_original_id(
            [],
            "UNKNOWN",  # type: ignore[arg-type]
        )
        assert result == {}
