# evaluate_model.py

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import json

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file at path {image_path} does not exist or cannot be read.")
    img = cv2.resize(img, (64, 64))  # Resize to match training images
    img = img / 255.0  # Normalize
    return img

def load_data(data_folder):
    images = []
    labels = []
    label_map = {}
    label_counter = 0
    
    # Iterate over each class folder
    for class_id in os.listdir(data_folder):
        class_folder = os.path.join(data_folder, class_id)
        if not os.path.isdir(class_folder):
            continue
        
        # Create a numeric label for each class
        if class_id not in label_map:
            label_map[class_id] = label_counter
            label_counter += 1
        
        # Process images in each class folder
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(label_map[class_id])  # Use numeric label

    images = np.array(images)
    labels = np.array(labels)
    
    # Ensure images have the correct shape
    images = np.reshape(images, (images.shape[0], 64, 64, 3))
    
    return images, labels, label_map

def evaluate_model(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    metrics = {
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred)
    }
    return metrics

def main():
    model_path = 'scripts/face_recognition_model.h5'
    data_folder = 'dataset/'
    
    # Load model
    model = load_model(model_path)
    
    # Load and prepare data
    images, labels, label_map = load_data(data_folder)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save metrics and label map to JSON files
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    with open('label_map.json', 'w') as f:
        json.dump(label_map, f, indent=4)
    
    print("Evaluation completed. Metrics saved to model_metrics.json.")
    print("Label map saved to label_map.json.")

if __name__ == "__main__":
    main()
