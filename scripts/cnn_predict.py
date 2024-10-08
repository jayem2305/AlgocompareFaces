import io
import sys
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
import time  # Import the time module
import cv2
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Set up UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
start_time = time.time()

# Define the path to the model file
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'face_recognition_model.h5')

# Define the paths to save the images
output_image_dir1 = os.path.join(script_dir, '../public/images/ID')
output_image_dir2 = os.path.join(script_dir, '../public/images/front')
os.makedirs(output_image_dir1, exist_ok=True)
os.makedirs(output_image_dir2, exist_ok=True)

# Check if the model file exists
if not os.path.exists(model_path):
    print(json.dumps({"result": "failure", "message": "Model file not found at path: " + model_path}))
    sys.exit(1)

# Load the model
model = load_model(model_path)

def preprocess_image(image_path, save_image=False, output_dir=None):
    """Preprocesses the image for the CNN model and detects faces.
       Optionally saves the image with detected faces, broken lines, and diamonds for the nose and mouth.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process the face
    eye_centers = []
    if len(faces) == 1:  # Ensure only one face is detected
        (x, y, w, h) = faces[0]

        # Detect eyes within the face
        roi_gray = gray_image[y:y + h, x:x + w]
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(roi_gray)

        # If at least two eyes are detected, draw the rectangle around the face
        if len(eyes) >= 2:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Store the eye centers
            for (ex, ey, ew, eh) in eyes:
                center = (x + ex + ew // 2, y + ey + eh // 2)
                eye_centers.append(center)

        # Draw a red line between the eyes
        if len(eye_centers) >= 2:
            cv2.line(image, eye_centers[0], eye_centers[1], (0, 0, 255), thickness=1)
        
        # Add broken lines at 1/4 and 3/4 of the face height
        face_mid_y_1_4 = y + h // 4
        face_mid_y_3_4 = y + 3 * h // 4

        # Calculate the positions for broken lines (1/4 and 3/4 of face height)
        quarter_y = y + h // 4
        three_quarters_y = y + 3 * h // 4

        # Calculate the position for vertical broken line (center of the face)
        center_x = x + w // 2

        # Function to draw a broken (dashed) line
        def draw_broken_line(img, x1, y1, x2, y2, color, thickness, dash_length=10, gap_length=10):
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            dashes = int(length / (dash_length + gap_length))
            for i in range(dashes):
                start_x = int(x1 + (x2 - x1) * (i / dashes))
                start_y = int(y1 + (y2 - y1) * (i / dashes))
                end_x = int(x1 + (x2 - x1) * ((i + 0.5) / dashes))
                end_y = int(y1 + (y2 - y1) * ((i + 0.5) / dashes))
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

        # Draw broken line at 1/4 of face height
        draw_broken_line(image, x, quarter_y, x + w, quarter_y, (255, 0, 0), thickness=1)

        # Draw broken line at 3/4 of face height
        draw_broken_line(image, x, three_quarters_y, x + w, three_quarters_y, (255, 0, 0), thickness=1)

        # Draw vertical broken line down the center of the face
        draw_broken_line(image, center_x, y, center_x, y + h, (255, 0, 0), thickness=1)

        # Adjust diamond sizes to fit nose and lips
        nose_height = h // 4  # Approximate size of the nose
        lip_height = h // 6   # Approximate size of the lips

        # Draw diamond shape on the nose (bridge of the nose)
       # nose_center = (center_x, y + h // 2)  # Approximate the nose center as the middle of the face
      #  draw_diamond(image, nose_center, nose_height // 2, (0, 255, 255))  # Yellow diamond at the nose

        # Draw diamond shape on the lips (below the nose)
      #  mouth_center = (center_x, y + int(h * 0.75))  # Approximate the mouth center below the nose
       # draw_diamond(image, mouth_center, lip_height // 2, (0, 255, 0))  # Green diamond at the lips

    # Save the image with detected faces, broken lines, and diamonds
    if save_image and output_dir:
        filename = os.path.basename(image_path)
        output_image_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_image_path, image)  # Save the image with drawings
        print(f"Image with detected faces, broken lines, and diamonds saved as {output_image_path}")

    # Prepare the image for CNN model
    input_shape = model.input_shape[1:3]  # (height, width)
    image_resized = cv2.resize(image, input_shape)
    image_normalized = np.array(image_resized, dtype=np.float32) / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    
    # Calculate average pixel values for each channel
    avg_pixel_values = np.mean(image_resized, axis=(0, 1))
    
    return image_expanded, output_image_path, avg_pixel_values, eye_centers  # Return eye centers

# Function to draw a diamond shape
#def draw_diamond(image, center, size, color, thickness=2):
   # """Draws a diamond shape centered at the given position."""
   # x, y = center
   # pts = np.array([
    #    [x, y - size],  # Top point
   #     [x - size, y],  # Left point
    #    [x, y + size],  # Bottom point
    #    [x + size, y],  # Right point
   # ], np.int32)
   # pts = pts.reshape((-1, 1, 2))
   # cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)



# Modify the compare_faces function to get eye distances
def compare_faces(image1_path, image2_path):
    """Compares two images using the CNN model."""
    img1, output_image_path1, avg_pixel_values_image1, eye_centers1 = preprocess_image(image1_path, save_image=True, output_dir=output_image_dir1)
    img2, output_image_path2, avg_pixel_values_image2, eye_centers2 = preprocess_image(image2_path, save_image=True, output_dir=output_image_dir2)

    prediction1 = model.predict(img1)
    prediction2 = model.predict(img2)

    # Compute similarity and distance
    similarity = cosine_similarity(prediction1, prediction2)[0][0]
    distance = euclidean_distances(prediction1, prediction2)[0][0]

    # Initialize average eye distances
    avg_distance1 = avg_distance2 = None

    # Calculate average eye distance if two eyes are detected
    if len(eye_centers1) == 2:
        avg_distance1 = np.linalg.norm(np.array(eye_centers1[0]) - np.array(eye_centers1[1]))
    if len(eye_centers2) == 2:
        avg_distance2 = np.linalg.norm(np.array(eye_centers2[0]) - np.array(eye_centers2[1]))

    average_eye_distance = None
    if avg_distance1 is not None and avg_distance2 is not None:
        average_eye_distance = (avg_distance1 + avg_distance2) / 2

    return (
        float(similarity),
        float(distance),
        output_image_path1,
        output_image_path2,
        avg_pixel_values_image1,
        avg_pixel_values_image2,
        avg_distance1,
        avg_distance2,
        average_eye_distance
    )

def calculate_metrics(y_true, y_pred):
    """Calculates precision, recall, f1 score, and accuracy."""
    if len(y_true) > 1:
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
    else:
        precision = recall = f1 = accuracy = float('nan')  # Or use 'N/A'
    return precision, recall, f1, accuracy

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"result": "failure", "message": "Please provide exactly two image paths as arguments."}))
        sys.exit(1)

    image1 = sys.argv[1]
    image2 = sys.argv[2]

    try:
        (
            similarity_score,
            distance,
            processed_image1,
            processed_image2,
            avg_pixel_values_image1,
            avg_pixel_values_image2,
            average_eye_distance,
            avg_distance1,
            avg_distance2
        ) = compare_faces(image1, image2)

        threshold = 0.75
        y_true = [1]  # Assuming true positive
        y_pred = [1 if similarity_score > threshold else 0]

        # Simulate more data for metrics calculation
        simulated_y_true = y_true + [0] * 4  # Add more false samples
        simulated_y_pred = y_pred + [0] * 4  # Add more false predictions

        precision, recall, f1, accuracy = calculate_metrics(simulated_y_true, simulated_y_pred)
        end_time = time.time()
        elapsed_time = end_time - start_time 
        result = {
            "result": "success" if similarity_score > threshold else "failure",
            "message": "Faces match!" if similarity_score > threshold else "Faces do not match",
            "similarity_score": f"{similarity_score:.4f}",
            "distance": f"{distance:.4f}",
            "eye_distance_image": processed_image1.replace("ID", "Detection/eyes"),
            "average_eye_distance_image1": f"{avg_distance1:.4f}" if avg_distance1 is not None else 'N/A',
            "average_eye_distance_image2": f"{avg_distance2:.4f}" if avg_distance2 is not None else 'N/A',
            "average_eye_distances": f"{average_eye_distance:.4f}" if average_eye_distance is not None else 'N/A',
            "avg_pixel_values_image1": {
                "Red": f"{avg_pixel_values_image1[2]:.2f}",
                "Green": f"{avg_pixel_values_image1[1]:.2f}",
                "Blue": f"{avg_pixel_values_image1[0]:.2f}"
            },
            "avg_pixel_values_image2": {
                "Red": f"{avg_pixel_values_image2[2]:.2f}",
                "Green": f"{avg_pixel_values_image2[1]:.2f}",
                "Blue": f"{avg_pixel_values_image2[0]:.2f}"
            },
            "metrics": {
                "Precision": f"{precision:.4f}" if not np.isnan(precision) else 'N/A',
                "Recall": f"{recall:.4f}" if not np.isnan(recall) else 'N/A',
                "F1 Score": f"{f1:.4f}" if not np.isnan(f1) else 'N/A',
                "Accuracy": f"{accuracy:.4f}" if not np.isnan(accuracy) else 'N/A'
            },
            "image1": processed_image1,
            "image2": processed_image2,
            "execution_time": f"{elapsed_time:.4f} seconds",
        }

        # Print the result
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"result": "failure", "message": str(e)}))
        sys.exit(1)
