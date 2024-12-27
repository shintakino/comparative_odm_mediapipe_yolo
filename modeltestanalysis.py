import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Constants for visualization
MARGIN = 25
ROW_SIZE = 10
FONT_SIZE = 2
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND_COLOR = (0, 0, 0)
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360

def visualize_mediapipe(image, object_result, hand_result, frame_count, fps):
    """Draws both object detection and hand landmarks on the input image."""
    # Display the processed frames and FPS
    processed_text = f"Frames: {frame_count}  FPS: {fps:.2f}"
    cv2.putText(image, processed_text, (MARGIN, MARGIN), cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Draw object detection results
    if object_result.detections:
        for detection in object_result.detections:
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
            
            text_location = (MARGIN + bbox.origin_x,
                           MARGIN + ROW_SIZE + bbox.origin_y)
            
            (text_width, text_height), baseline = cv2.getTextSize(
                result_text, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, FONT_THICKNESS)
            
            cv2.rectangle(image, 
                         (text_location[0], text_location[1] - text_height),
                         (text_location[0] + text_width, text_location[1] + baseline),
                         TEXT_BACKGROUND_COLOR, -1)
            
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                       FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Draw hand landmarks
    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in hand_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

    return image

# Initialize MediaPipe detectors
def initialize_mediapipe_detectors():
    # Object detector setup
    object_base_options = python.BaseOptions(model_asset_path='media.tflite')
    object_options = vision.ObjectDetectorOptions(
        base_options=object_base_options,
        score_threshold=0.5)
    object_detector = vision.ObjectDetector.create_from_options(object_options)

    # Hand landmarker setup
    hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    return object_detector, hand_detector

# Load YOLO Model
yolo_model = YOLO("yolo.pt")

# Load Video Input
video_file = 'analysis.mp4'
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize MediaPipe detectors
object_detector, hand_detector = initialize_mediapipe_detectors()

# Process with MediaPipe
frame_count = 0
total_time_mediapipe = 0
fps_values_mediapipe = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe processing
    start_time_mediapipe = time.time()

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run both MediaPipe detectors
    object_result = object_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)

    # Calculate FPS for MediaPipe
    frame_count += 1
    end_time_mediapipe = time.time()
    total_time_mediapipe += (end_time_mediapipe - start_time_mediapipe)
    avg_frame_time_mediapipe = total_time_mediapipe / frame_count
    fps_mediapipe = 1 / avg_frame_time_mediapipe
    fps_values_mediapipe.append(fps_mediapipe)

    # Visualize MediaPipe results
    annotated_frame = visualize_mediapipe(
        frame.copy(), object_result, hand_result, frame_count, fps_mediapipe)
    resized_frame = cv2.resize(annotated_frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    cv2.imshow('MediaPipe Detection', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# Process with YOLO
cap = cv2.VideoCapture(video_file)
frame_count = 0
total_time_yolo = 0
fps_values_yolo = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO processing
    start_time_yolo = time.time()
    yolo_results = yolo_model(frame)

    # Calculate FPS for YOLO
    frame_count += 1
    end_time_yolo = time.time()
    total_time_yolo += (end_time_yolo - start_time_yolo)
    avg_frame_time_yolo = total_time_yolo / frame_count
    fps_yolo = 1 / avg_frame_time_yolo
    fps_values_yolo.append(fps_yolo)

    # Draw YOLO results
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[box.cls[0].item()]
            confidence = box.conf[0].item()
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), TEXT_COLOR, 3)
            label_text = f"{label} ({confidence:.2f})"
            text_location = (x1, y1 - 10)
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, FONT_THICKNESS)
            
            cv2.rectangle(frame, 
                         (text_location[0], text_location[1] - text_height),
                         (text_location[0] + text_width, text_location[1] + baseline),
                         TEXT_BACKGROUND_COLOR, -1)
            
            cv2.putText(frame, label_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                       FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    resized_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    cv2.imshow('YOLO Detection', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Generate FPS comparison graph
plt.figure(figsize=(10, 6))
plt.plot(fps_values_mediapipe, label='MediaPipe (Object + Hand)', color='blue', linewidth=2)
plt.plot(fps_values_yolo, label='YOLO', color='red', linewidth=2)
plt.xlabel('Frame Number')
plt.ylabel('FPS')
plt.title('FPS Comparison: MediaPipe vs YOLO')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Print final FPS statistics
print(f"Final MediaPipe FPS (Object + Hand): {fps_mediapipe:.2f}")
print(f"Final YOLO FPS: {fps_yolo:.2f}")