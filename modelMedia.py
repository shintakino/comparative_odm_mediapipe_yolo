import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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

def visualize(image, detection_result, frame_count, fps):
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

# Step 2: Load Video Input
video_file = 'analysis.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_file)

# Check if the video file is loaded correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Step 3: Measure Inference Speed
frame_count = 0
total_time = 0
fps_values = []  # List to store FPS values for plotting

while cap.isOpened():
    ret, frame = cap.read()  # Read the next frame
    if not ret:
        break  # Break the loop if no more frames

    # Start the timer for the current frame
    start_time = time.time()

    # Convert the frame from BGR to RGB (MediaPipe uses RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect objects in the frame
    detection_result = detector.detect(mp_image)

    # Update frame count and calculate FPS
    frame_count += 1
    end_time = time.time()
    total_time += (end_time - start_time)
    avg_frame_time = total_time / frame_count
    fps = 1 / avg_frame_time
    fps_values.append(fps)  # Store the FPS for this frame

    # Visualize the detections and frame info on the frame
    annotated_frame = visualize(frame, detection_result, frame_count, fps)

    # Resize the annotated frame
    resized_frame = cv2.resize(annotated_frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Convert frame from BGR to RGB (for displaying with OpenCV)
    rgb_annotated_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Display the frame with annotations
    cv2.imshow('Object Detection - Video', rgb_annotated_frame)

    # Wait for a short time to display the frame
    cv2.waitKey(1)

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the final FPS after processing the video
if frame_count > 0:
    avg_frame_time = total_time / frame_count
    fps = 1 / avg_frame_time
    print(f"Final FPS: {fps:.2f}")

# Step 4: Plot FPS over time
plt.figure(figsize=(10, 6))
plt.plot(fps_values, label="FPS", color='b')
plt.xlabel("Frame Number")
plt.ylabel("FPS")
plt.title("FPS Analysis Over Video Frames")
plt.legend()
plt.grid(True)
plt.show()
