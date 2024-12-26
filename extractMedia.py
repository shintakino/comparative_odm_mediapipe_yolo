import json
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Step 1: Load JSON data from a file
json_path = r'datasets\valid\labels.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Step 2: Setup MediaPipe Object Detection
model_path = r'media.tflite'  # Replace with your TFLite model path
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Step 3: Directory containing test images
image_directory = r'datasets\valid\images'

# Prepare a mapping of actual classes
categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
annotations = data.get("annotations", [])

# Prepare a list to store results
results = []

# Step 4: Loop through each image in the JSON data based on ID
for image_info in sorted(data['images'], key=lambda x: x['id']):
    image_name = image_info['file_name']
    image_id = image_info['id']  # Get the image ID
    image_path = os.path.join(image_directory, image_name)
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        continue

    # Load the image
    image = cv2.imread(image_path)

    # Check if the image is loaded correctly
    if image is None:
        print(f"Error: Could not open image file {image_path}.")
        continue

    # Convert the image from BGR to RGB (MediaPipe uses RGB format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Step 5: Detect objects in the image
    detection_result = detector.detect(mp_image)

    # Step 6: Store Detected Labels and Probabilities
    for detection in detection_result.detections:
        predicted_class_name = detection.categories[0].category_name  # Get the predicted class name
        probability = round(detection.categories[0].score, 2)  # Get the probability

        # Get the actual class name from annotations
        actual_class_name = "Unknown"
        for annotation in annotations:
            if annotation["image_id"] == image_id:
                actual_class_name = categories.get(annotation["category_id"], "Unknown")
                break

        # Append the results to the list
        results.append({
            "S.No": image_id + 1,  # Adjusting image_id to start from 1
            "Actual_class": actual_class_name,
            "Predicted_class": predicted_class_name,
            "Prob": probability
        })

# Step 7: Create a DataFrame from the results
df = pd.DataFrame(results)

# Step 8: Save to Excel
excel_file = "media_output_labels.xlsx"
df.to_excel(excel_file, index=False, sheet_name="Labels")

print(f"Data successfully saved to {excel_file}")