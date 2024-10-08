import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data_from_folder(folder_path):
    images = []
    labels = []
    for label, sub_folder in enumerate(['front', 'left', 'right']):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        for filename in os.listdir(sub_folder_path):
            img_path = os.path.join(sub_folder_path, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Resize images to 64x64
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
    return np.array(images), np.array(labels)

def preprocess_data():
    images, labels = load_data_from_folder('dataset')
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    np.save('scripts/x_train.npy', x_train)
    np.save('scripts/x_val.npy', x_val)
    np.save('scripts/y_train.npy', y_train)
    np.save('scripts/y_val.npy', y_val)

if __name__ == "__main__":
    preprocess_data()
