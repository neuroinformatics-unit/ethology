"""YOLO detector inference logic."""

import cv2
from ultralytics import YOLO


class YOLODetector:
    """YOLOv8 detector for running inference on videos."""

    def __init__(self, model_path):
        """Initialize YOLO model from weights."""
        self.model = YOLO(model_path)

    def run(self, video_path, visualize=False):
        """Run detection on a video.

        Args:
            video_path (str): Path to input video.
            visualize (bool): Whether to show detections with OpenCV.

        Returns:
            list: List of detection dictionaries.

        """
        cap = cv2.VideoCapture(video_path)
        detections = []

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, verbose=False)[0]

            for det in results.boxes:
                bbox = det.xyxy[0].tolist()
                conf = float(det.conf[0])
                cls_id = int(det.cls[0])
                class_name = self.model.names[cls_id]

                detections.append(
                    {
                        "frame": frame_id,
                        "bbox": bbox,
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": class_name,
                    }
                )
                if visualize:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
            if visualize:
                cv2.imshow("Detections", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1

        cap.release()
        if visualize:
            cv2.destroyAllWindows()

        return detections
