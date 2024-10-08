import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model file
model_path = os.path.join(script_dir, '..', 'models', 'face_recognition_model_r.pth')

# Resolve the absolute path
model_path = os.path.abspath(model_path)

# Print the resolved model path for debugging
print(f"Resolved model path: {model_path}")

# Check if the file exists
if os.path.isfile(model_path):
    print("Model file found.")
else:
    print("Model file not found.")
