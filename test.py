from ultralytics import YOLO

try:
    model = YOLO("yolo.pt")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
