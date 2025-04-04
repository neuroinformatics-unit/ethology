from unittest.mock import MagicMock, patch

import numpy as np

from ethology.detectors.inference.yolo_inference import YOLODetector


@patch("ethology.detectors.inference.yolo_inference.YOLO")
@patch("cv2.VideoCapture")
def test_yolo_detector_run(mock_capture, mock_yolo):
    dummy_frame = np.ones((640, 480, 3), dtype=np.uint8)

    mock_cap_instance = mock_capture.return_value
    mock_cap_instance.isOpened.side_effect = [
        True,
        False,
    ]
    mock_cap_instance.read.return_value = (True, dummy_frame)

    mock_model_instance = mock_yolo.return_value
    mock_model_instance.names = {1: "test_class"}

    mock_result = type("Result", (), {})()
    mock_result.boxes = [
        type(
            "Box",
            (),
            {
                "xyxy": [np.array([0, 0, 100, 100])],
                "conf": [np.array([0.9])],
                "cls": [np.array([1])],
            },
        )()
    ]
    mock_model_instance.predict.return_value = [mock_result]

    detector = YOLODetector("mock_path.pt")
    detections = detector.run("mock_video.mp4", visualize=False)

    assert len(detections) == 1
    assert detections[0]["class_name"] == "test_class"


@patch("ethology.detectors.inference.yolo_inference.YOLO")
@patch("cv2.VideoCapture")
def test_yolo_detector_video_not_opened(mock_capture, mock_yolo):
    mock_capture.return_value.isOpened.return_value = False

    detector = YOLODetector("mock.pt")
    detections = detector.run("invalid.mp4", visualize=False)

    assert detections == []


@patch("cv2.imshow")
@patch("cv2.rectangle")
@patch("cv2.putText")
@patch("cv2.VideoCapture")
@patch("ethology.detectors.inference.yolo_inference.YOLO")
def test_yolo_detector_visualize(
    mock_yolo, mock_capture, mock_text, mock_rect, mock_imshow
):
    frame = np.ones((640, 480, 3), dtype=np.uint8)
    mock_cap = mock_capture.return_value
    mock_cap.isOpened.side_effect = [True, False]
    mock_cap.read.return_value = (True, frame)

    mock_model = mock_yolo.return_value
    mock_model.names = {1: "test_class"}
    mock_result = MagicMock()
    mock_result.boxes = [
        MagicMock(
            xyxy=[np.array([0, 0, 100, 100])],
            conf=[np.array([0.9])],
            cls=[np.array([1])],
        )
    ]
    mock_model.predict.return_value = [mock_result]

    detector = YOLODetector("mock.pt")
    result = detector.run("dummy.mp4", visualize=True)

    assert len(result) == 1
    mock_rect.assert_called_once()
    mock_text.assert_called_once()
