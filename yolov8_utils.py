from ultralytics import YOLO
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)

# Path to the YOLOv8 face detection model
model_path = "models/yolov8n-face.pt"

try:
    logging.info("Attempting to load YOLOv8 face detection model...")
    model = YOLO(model_path)
    logging.info("YOLOv8 model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLOv8 model from {model_path}: {e}")
    raise ImportError(f"Could not load the YOLOv8 model. Please ensure the file '{model_path}' exists and is not corrupted.")

def detect_faces(frame):
    """
    Detects faces in a given frame using the YOLOv8 model.
    Returns a list of tuples, each containing the cropped face image and its coordinates.
    """
    results = model(frame)
    faces_with_coords = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_face = frame[y1:y2, x1:x2]
            faces_with_coords.append((cropped_face, (x1, y1, x2, y2)))
    return faces_with_coords
