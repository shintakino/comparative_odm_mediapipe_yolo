import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Constants for visualization
MARGIN = 25  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 2  # Increased font size
FONT_THICKNESS = 2  # Increased thickness for better visibility
TEXT_COLOR = (255, 255, 255)  # White text for better contrast
TEXT_BACKGROUND_COLOR = (0, 0, 0)  # Black background for text
RESIZE_WIDTH = 640  # Resize width (adjust for your needs)
RESIZE_HEIGHT = 360  # Resize height (adjust for your needs)

def visualize_mediapipe(image, detection_result, frame_count, fps):
    """Draws bounding boxes and processed frames info on the input image and returns it."""
    # Display the processed frames and FPS at the top-left corner
    processed_text = f"Frames: {frame_count}  FPS: {fps:.2f}"
    cv2.putText(image, processed_text, (MARGIN, MARGIN), cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    for detection in detection_result.detections:
        # Draw bounding box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        
        # Text location
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        
        # Compute text size for background
        (text_width, text_height), baseline = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, FONT_THICKNESS)
        
        # Draw a filled rectangle as background for text to improve visibility
        cv2.rectangle(image, (text_location[0], text_location[1] - text_height),
                      (text_location[0] + text_width, text_location[1] + baseline),
                      TEXT_BACKGROUND_COLOR, -1)
        
        # Put the text over the background
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

# Step 1: Setup MediaPipe Object Detection
base_options = python.BaseOptions(model_asset_path='media.tflite')  # Replace with path to your tflite model
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Step 2: Load YOLO Model
yolo_model = YOLO("yolo.pt")  # Replace with the path to your YOLO model

# Step 3: Load Video Input
video_file = 'analysis.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_file)

# Check if the video file is loaded correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Step 4: Process with MediaPipe first
frame_count = 0
total_time_mediapipe = 0
fps_values_mediapipe = []  # List to store FPS values for MediaPipe

while cap.isOpened():
    ret, frame = cap.read()  # Read the next frame
    if not ret:
        break  # Break the loop if no more frames

    # ==================== Run MediaPipe ====================
    start_time_mediapipe = time.time()

    # Convert the frame from BGR to RGB (MediaPipe uses RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect objects in the frame using MediaPipe
    detection_result = detector.detect(mp_image)

    # Calculate FPS for MediaPipe
    frame_count += 1
    end_time_mediapipe = time.time()
    total_time_mediapipe += (end_time_mediapipe - start_time_mediapipe)
    avg_frame_time_mediapipe = total_time_mediapipe / frame_count
    fps_mediapipe = 1 / avg_frame_time_mediapipe
    fps_values_mediapipe.append(fps_mediapipe)

    # Visualize the detections and frame info on the frame (MediaPipe)
    annotated_frame_mediapipe = visualize_mediapipe(frame, detection_result, frame_count, fps_mediapipe)

    # Display the frame with MediaPipe annotations
    resized_frame_mediapipe = cv2.resize(annotated_frame_mediapipe, (RESIZE_WIDTH, RESIZE_HEIGHT))
    rgb_annotated_frame_mediapipe = cv2.cvtColor(resized_frame_mediapipe, cv2.COLOR_BGR2RGB)
    cv2.imshow('Object Detection - MediaPipe', rgb_annotated_frame_mediapipe)

    # Wait for a short time to display the frame
    cv2.waitKey(1)

# After MediaPipe completes, reset video capture to start again for YOLO processing
cap.release()

# Step 5: Reload Video for YOLO Processing
cap = cv2.VideoCapture(video_file)

# Check if the video file is loaded correctly again
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0
total_time_yolo = 0
fps_values_yolo = []  # List to store FPS values for YOLO

while cap.isOpened():
    ret, frame = cap.read()  # Read the next frame
    if not ret:
        break  # Break the loop if no more frames

    # ==================== Run YOLO ====================
    start_time_yolo = time.time()

    # Run YOLO model inference on the frame
    yolo_results = yolo_model(frame)

    # Calculate FPS for YOLO
    frame_count += 1
    end_time_yolo = time.time()
    total_time_yolo += (end_time_yolo - start_time_yolo)
    avg_frame_time_yolo = total_time_yolo / frame_count
    fps_yolo = 1 / avg_frame_time_yolo
    fps_values_yolo.append(fps_yolo)

    # Draw bounding boxes and labels for YOLO detections
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[box.cls[0].item()]
            confidence = box.conf[0].item()
            label_text = f"{label} ({confidence:.2f})"
            text_location = (x1, y1 - 10)
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, FONT_THICKNESS)
            cv2.rectangle(frame, (text_location[0], text_location[1] - text_height),
                          (text_location[0] + text_width, text_location[1] + baseline),
                          TEXT_BACKGROUND_COLOR, -1)
            cv2.putText(frame, label_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Display the frame with YOLO annotations
    resized_frame_yolo = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    rgb_annotated_frame_yolo = cv2.cvtColor(resized_frame_yolo, cv2.COLOR_BGR2RGB)
    cv2.imshow('Object Detection - YOLO', rgb_annotated_frame_yolo)

    # Wait for a short time to display the frame
    cv2.waitKey(1)

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Step 6: Generate the FPS Comparison Graph
plt.figure(figsize=(10, 6))
plt.plot(fps_values_mediapipe, label='MediaPipe FPS', color='blue', linewidth=2)
plt.plot(fps_values_yolo, label='YOLO FPS', color='red', linewidth=2)
plt.xlabel('Frame Number')
plt.ylabel('FPS')
plt.title('FPS Comparison: MediaPipe vs YOLO')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Print the final FPS for both MediaPipe and YOLO after processing the video
if frame_count > 0:
    avg_frame_time_mediapipe = total_time_mediapipe / frame_count
    fps_mediapipe = 1 / avg_frame_time_mediapipe
    avg_frame_time_yolo = total_time_yolo / frame_count
    fps_yolo = 1 / avg_frame_time_yolo

    print(f"Final FPS for MediaPipe: {fps_mediapipe:.2f}")
    print(f"Final FPS for YOLO: {fps_yolo:.2f}")
