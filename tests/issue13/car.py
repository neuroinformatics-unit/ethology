import cv2
import numpy as np
from boxmot.tracker_zoo import create_tracker
from ultralytics import YOLO
import tempfile

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# ByteTrack configuration
BYTETRACK_CONFIG = """
tracker_type: bytetrack
track_thresh: 0.5
track_buffer: 50
match_thresh: 0.8
aspect_ratio_thresh: 3.0
min_box_area: 10
mot20: false
frame_rate: 30
"""

# Create temporary config file
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
    f.write(BYTETRACK_CONFIG)
    config_path = f.name

# Initialize tracker
tracker = create_tracker(
    'bytetrack',
    config_path,
    None,  # reid_weights
    'cpu',
    False,
    False
)

cap = cv2.VideoCapture("video.mp4")
selected_id = None

def click_event(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        for track in param:
            x1, y1, x2, y2 = map(int, track[:4])
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_id = int(track[4])
                print(f"Selected ID: {selected_id}")
                break

cv2.namedWindow("Tracking")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)[0]
    
    # Filter for cars and get detection components
    car_mask = results.boxes.cls.cpu().numpy() == 2
    bboxes = results.boxes.xyxy.cpu().numpy()[car_mask]
    confs = results.boxes.conf.cpu().numpy()[car_mask]
    cls_ids = results.boxes.cls.cpu().numpy()[car_mask]  # Get class IDs

    # Format detections correctly [x1, y1, x2, y2, conf, class]
    detections = np.hstack([
        bboxes,
        confs.reshape(-1, 1),
        cls_ids.reshape(-1, 1)  # Add class IDs as final column
    ])

    # Update tracker
    tracks = tracker.update(detections, frame)

    # Rest of the code remains the same...
    cv2.setMouseCallback("Tracking", click_event, param=tracks)

    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        tid = int(track[4])
        color = (0, 255, 0) if tid == selected_id else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {tid}', (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if selected_id is not None:
        cv2.putText(frame, f"Tracking ID: {selected_id}",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
