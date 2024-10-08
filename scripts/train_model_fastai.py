from fastai.vision.all import *
import os

# Define paths
dataset_path = 'dataset/'
image_size = (224, 224)  # Larger image size for detailed features
batch_size = 64
num_workers = 4

def get_face_files_and_labels():
    files = get_image_files(dataset_path)
    labels = [file.parent.name for file in files]
    return files, labels

def train_model():
    # Create a DataBlock for image classification
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_face_files_and_labels,
        splitter=RandomSplitter(valid_pct=0.2),
        get_y=lambda x: x.parent.name,
        item_tfms=Resize(image_size),
        batch_tfms=Normalize.from_stats(*imagenet_stats)
    )

    # Create DataLoaders
    dls = dblock.dataloaders(dataset_path, bs=batch_size, num_workers=num_workers)

    # Build and train the CNN model
    learn = vision_learner(dls, resnet34, metrics=accuracy)

    # Train for more epochs initially
    learn.fine_tune(20)

    # Save the trained model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, 'face_recognition_model.pkl')
    learn.export(model_save_path)

    print(f"Model training completed and saved to {model_save_path}!")

if __name__ == "__main__":
    train_model()
