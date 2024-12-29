from ultralytics import YOLO
from roboflow import Roboflow

#AUTO GENERATE DARI ROBOFLOW
rf = Roboflow(api_key="JANGAN SAMPAI API KEY ANDA PUBLIK!!!!!!!!!!!!!!")
project = rf.workspace("DATASET ROBOFLOW").project("NAMA DATASET")
version = project.version(1)
dataset = version.download("yolov11")

#UBAH SESUAI PATH DATASET ANDA SENDIRI
data_path = 'C:/Users/Workload13/Documents/yoloproject/cs2dataset-3/data.yaml'

model = YOLO("yolo11s.pt")

# Train the model
results = model.train(data=data_path, epochs=100, imgsz=640, workers = 0)#workers = 0 for crash fix

version.deploy("yolov11", "C:/Users/Workload13/Documents/yoloproject/runs/detect/train10/weights", "best.pt")