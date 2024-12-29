from ultralytics import YOLO

model = YOLO("best.pt")
model.export(
    format="engine",
    batch=1, 
    int8=True,
    data="C:/Users/Workload13/Documents/yoloproject/cs2dataset-3/data.yaml",  
)

# Load the exported TensorRT INT8 model
model = YOLO("best.engine", task="detect")

# Run inference
result = model.predict("CT.png", save=True)