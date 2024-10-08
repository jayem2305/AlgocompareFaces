import os

def train_yolo():
    # Change to YOLOv5 directory (make sure the path is correct)
    yolov5_path = os.path.join(os.getcwd(), 'yolov5')
    os.chdir(yolov5_path)

    # Define command for training YOLOv5 using CPU
    train_command = (
        "python train.py --img 640 --batch 16 --epochs 50 "
        "--data ../dataset2/data.yaml --weights yolov5s.pt --device cpu"
    )

    # Execute the training command
    os.system(train_command)

    print("YOLO training completed!")

if __name__ == "__main__":
    train_yolo()
