import json
import pandas as pd
import cv2
import os
from ultralytics import YOLO

# Step 1: Load JSON data from a file
json_path = r'datasets\valid\labels.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Step 2: Load the YOLO model
model_path = "yolo.pt"  # Replace with your YOLO model path
model = YOLO(model_path)

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

    # Step 5: Run YOLO model inference on the image
    results_yolo = model(image)

    # Step 6: Process YOLO detections
    for result in results_yolo:
        for box in result.boxes:
            # Get the label and confidence
            label = result.names[box.cls[0].item()]  # Predicted class name
            confidence = round(box.conf[0].item(), 2)  # Confidence score

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
                "Predicted_class": label,
                "Prob": confidence
            })

# Step 7: Create a DataFrame from the results
df = pd.DataFrame(results)

# Step 8: Save to Excel
excel_file = "yolo_output_labels.xlsx"
df.to_excel(excel_file, index=False, sheet_name="Labels")

print(f"Data successfully saved to {excel_file}")
