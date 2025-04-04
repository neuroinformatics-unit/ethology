import json
import tempfile
from unittest.mock import patch

import pytest

from ethology.detectors.inference import run_detector


@patch("ethology.detectors.inference.run_detector.YOLODetector")
def test_run_detector_main(mock_yolo_cls):
    dummy_detector = mock_yolo_cls.return_value
    dummy_detector.run.return_value = [{"frame": 0, "class_name": "test"}]

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp_output:
        test_args = [
            "--video_path",
            "sample.mp4",
            "--detector_type",
            "yolo",
            "--model_path",
            "yolov8n.pt",
            "--output_path",
            tmp_output.name,
        ]
        with patch("sys.argv", ["run_detector.py"] + test_args):
            run_detector.main()

        with open(tmp_output.name) as f:
            data = json.load(f)
        assert data[0]["class_name"] == "test"


@patch(
    "sys.argv",
    [
        "run_detector.py",
        "--video_path",
        "a",
        "--detector_type",
        "invalid",
        "--model_path",
        "b",
        "--output_path",
        "out.json",
    ],
)
def test_unsupported_detector(capsys):
    with pytest.raises(SystemExit):
        run_detector.main()

    captured = capsys.readouterr()
    assert "invalid choice: 'invalid'" in captured.err
