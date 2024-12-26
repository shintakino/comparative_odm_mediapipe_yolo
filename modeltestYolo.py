import cv2
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("yolo.pt")

# Setup constants for visualization
MARGIN = 25  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 2  # Increased font size
FONT_THICKNESS = 2  # Increased thickness for better visibility
TEXT_COLOR = (255, 255, 255)  # White text for better contrast
TEXT_BACKGROUND_COLOR = (0, 0, 0)  # Black background for text
RESIZE_WIDTH = 640  # Resize width (adjust for your needs)
RESIZE_HEIGHT = 360  # Resize height (adjust for your needs)

# Step 1: Open the video file
video_file = 'analysis.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_file)

# Check if the video file is opened correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Step 2: Measure Inference Speed
frame_count = 0
total_time = 0
fps_values = []  # List to store FPS values for plotting

while cap.isOpened():
    ret, frame = cap.read()  # Read the next frame
    if not ret:
        break  # Break the loop if no more frames

    # Start the timer for the current frame
    start_time = time.time()

    # Run YOLO model inference on the frame
    results = model(frame)

    # Update frame count and calculate FPS
    frame_count += 1
    end_time = time.time()
    total_time += (end_time - start_time)
    avg_frame_time = total_time / frame_count
    fps = 1 / avg_frame_time
    fps_values.append(fps)  # Store the FPS for this frame

    # Draw bounding boxes for each detection
    for result in results:
        for box in result.boxes:
            # Get the bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates from tensor
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), TEXT_COLOR, 3)
            
            # Get the label and confidence
            label = result.names[box.cls[0].item()]  # Class name
            confidence = box.conf[0].item()  # Confidence score

            # Prepare text with label and confidence
            label_text = f"{label} ({confidence:.2f})"
            
            # Text location
            text_location = (x1, y1 - 10)
            
            # Compute text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, FONT_THICKNESS)
            
            # Draw a filled rectangle as background for text to improve visibility
            cv2.rectangle(frame, (text_location[0], text_location[1] - text_height),
                          (text_location[0] + text_width, text_location[1] + baseline),
                          TEXT_BACKGROUND_COLOR, -1)
            
            # Put the text over the background
            cv2.putText(frame, label_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Display processed frames and FPS at the top-left corner
    processed_text = f"Frames: {frame_count}  FPS: {fps:.2f}"
    cv2.putText(frame, processed_text, (MARGIN, MARGIN), cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Resize the frame
    resized_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Convert frame from BGR to RGB for visualization
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Display the frame with annotations
    cv2.imshow('YOLO Object Detection - Video', rgb_frame)

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

# Step 3: Plot FPS over time
plt.figure(figsize=(10, 6))
plt.plot(fps_values, label="FPS", color='b')
plt.xlabel("Frame Number")
plt.ylabel("FPS")
plt.title("FPS Analysis Over Video Frames")
plt.legend()
plt.grid(True)
plt.show()
