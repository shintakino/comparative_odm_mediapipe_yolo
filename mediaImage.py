import json
import cv2
import os
import mediapipe as mp

# Step 1: Load JSON data from a file
json_path = r'media_datasets\valid\labels.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Step 2: Initialize MediaPipe Object Detection
mp_object_detection = mp.solutions.object_detection
object_detection = mp_object_detection.ObjectDetection(min_detection_confidence=0.5)

# Step 3: Directory containing test images
image_directory = r'media_datasets\valid\images'

# Step 4: Select a single image (Modify this to your target image file name)
target_image_name = "04_04381_png.rf.a8faf624ff77b01d720dae6ea7c502dc.jpg"  # Change this to the image you want to test
image_info = next((img for img in data['images'] if img['file_name'] == target_image_name), None)

if not image_info:
    print(f"Error: Image '{target_image_name}' not found in JSON metadata.")
else:
    image_id = image_info['id']  # Add 1 to the image ID
    image_path = os.path.join(image_directory, target_image_name)

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
    else:
        # Load the image
        image = cv2.imread(image_path)

        # Check if the image is loaded correctly
        if image is None:
            print(f"Error: Could not open image file {image_path}.")
        else:
            # Convert the image to RGB (as MediaPipe expects RGB input)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run MediaPipe object detection
            results = object_detection.process(rgb_image)

            # Step 5: Process MediaPipe detections and print results
            if results.detections:
                for detection in results.detections:
                    # Get the predicted class name and confidence
                    label = detection.label
                    confidence = round(detection.score, 2)

                    # Get the actual class name from annotations
                    actual_class_name = "Unknown"
                    for annotation in data.get("annotations", []):
                        if annotation["image_id"] == image_info["id"]:
                            actual_class_name = data["categories"][annotation["category_id"]]["name"]
                            break

                    # Print the result
                    print(f"Image ID: {image_id}, "  # Now starts from 1 instead of 0
                          f"Actual Class: {actual_class_name}, "
                          f"Predicted Class: {label}, "
                          f"Confidence: {confidence}")
            else:
                print(f"No detections found for image: {image_path}")

