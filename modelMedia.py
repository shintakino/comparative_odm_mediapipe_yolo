import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
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

def draw_detections_and_landmarks(image, object_result, hand_result, frame_count, fps):
    """Draws object detection boxes and hand landmarks on the input image."""
    annotated_image = np.copy(image)
    
    # Display the processed frames and FPS
    processed_text = f"Frames: {frame_count}  FPS: {fps:.2f}"
    cv2.putText(annotated_image, processed_text, (MARGIN, MARGIN), 
                cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Draw object detection boxes and labels
    if object_result.detections:
        for detection in object_result.detections:
            # Draw bounding box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = f"{category_name} ({probability})"
            
            text_location = (MARGIN + bbox.origin_x,
                           MARGIN + ROW_SIZE + bbox.origin_y)
            
            # Draw text background
            (text_width, text_height), baseline = cv2.getTextSize(
                result_text, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, FONT_THICKNESS)
            cv2.rectangle(annotated_image, 
                         (text_location[0], text_location[1] - text_height),
                         (text_location[0] + text_width, text_location[1] + baseline),
                         TEXT_BACKGROUND_COLOR, -1)
            
            # Draw text
            cv2.putText(annotated_image, result_text, text_location, 
                       cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Draw hand landmarks (without labels)
    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            # Convert landmarks to proto format
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in hand_landmarks
            ])

            # Draw hand landmarks and connections
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image

# Initialize both detectors
def initialize_detectors():
    # Object detector setup
    object_base_options = python.BaseOptions(model_asset_path='mediaNew.tflite')
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

# Main processing loop
def process_video(video_path):
    # Initialize detectors
    object_detector, hand_detector = initialize_detectors()
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Initialize metrics
    frame_count = 0
    total_time = 0
    fps_values = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run both detections
        object_result = object_detector.detect(mp_image)
        hand_result = hand_detector.detect(mp_image)

        # Update metrics
        frame_count += 1
        end_time = time.time()
        total_time += (end_time - start_time)
        avg_frame_time = total_time / frame_count
        fps = 1 / avg_frame_time
        fps_values.append(fps)

        # Draw detections and display
        annotated_frame = draw_detections_and_landmarks(
            frame, object_result, hand_result, frame_count, fps)
        resized_frame = cv2.resize(annotated_frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        
        cv2.imshow('Combined Detection - Video', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Print final FPS
    if frame_count > 0:
        avg_frame_time = total_time / frame_count
        fps = 1 / avg_frame_time
        print(f"Final FPS: {fps:.2f}")

    # Plot FPS analysis
    plt.figure(figsize=(10, 6))
    plt.plot(fps_values, label="FPS", color='b')
    plt.xlabel("Frame Number")
    plt.ylabel("FPS")
    plt.title("FPS Analysis Over Video Frames")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the program
if __name__ == "__main__":
    video_file = 'analysis.mp4'  # Replace with your video file path
    process_video(video_file)