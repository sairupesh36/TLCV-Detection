# Traffic Sign Detection using YOLOv8



## Overview
This project implements traffic sign detection using the YOLOv8 (You Only Look Once) model. The dataset is downloaded from Roboflow, and the model is trained using the Ultralytics YOLO library. The trained model is then used for real-time traffic sign detection in videos.

## Installation
To set up the environment, install the required dependencies:

```bash
!pip install ultralytics==8.0.0
!pip install roboflow
```

## GPU Check
Check if a GPU is available to speed up model training:

```bash
!nvidia-smi
```

## Directory Setup
Create and navigate to the dataset directory:

```python
HOME = "/content/"
!mkdir {HOME}/datasets
%cd {HOME}/datasets
```

## Dataset Download
Download the traffic sign dataset using Roboflow API:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("roboflow-100").project("road-signs-6ih4y")
dataset = project.version(2).download("yolov5")
```

## Model Training
Train the YOLOv8 model using the dataset:

```bash
%cd {HOME}
!yolo task=detect mode=train model=yolov8s.pt data='/content/datasets/road_signs/data.yaml' epochs=100 imgsz=640
```

## Model Evaluation
Display training results such as the confusion matrix and validation predictions:

```python
from IPython.display import Image
Image(filename=f"{HOME}/runs/detect/train2/confusion_matrix.png")
Image(filename=f"{HOME}/runs/detect/train3/val_batch2_pred.jpg", height=500)
Image(filename=f"{HOME}/runs/detect/train3/results.png", width=600)
```

## Inference on Video
Run the trained model to detect traffic signs in a video:

```bash
!yolo task=detect mode=predict model=/content/runs/detect/train2/weights/best.pt conf=0.25 source='Traffic Sign.mp4'
```

## Compress and Display Video
Compress the output video and display it in the notebook:

```python
from IPython.display import HTML
from base64 import b64encode
import os

save_path = f'{HOME}/runs/detect/predict3/Traffic_sign.mp4'
compressed_path = f'{HOME}/compressed_traffic_sign.mp4'

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

mp4 = open(compressed_path, 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f'<video width=400 controls><source src="{data_url}" type="video/mp4"></video>')
```

## Load Trained Model
Load the trained YOLO model using OpenCV and Ultralytics:

```python
import cv2
from IPython.display import display, Image
import torch
from ultralytics import YOLO

model = YOLO(f"{HOME}/runs/detect/train2/weights/best.pt")
```

## Acknowledgments
- **Ultralytics YOLOv8** for object detection.
- **Roboflow** for dataset management.
- **OpenCV** for image processing.
- **Google Colab** for providing a cloud-based training environment.

## License
This project is for educational purposes and follows the open-source guidelines of the utilized tools and datasets.

