#convert default yolo model to tensorrt using coco dataset calibration
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.export(
    format="engine",
    dynamic=True,  
    batch=8,  
    workspace=3, 
    imgsz=640, 
    int8=True,
    data="coco128.yaml", 
    device=0 
)

# Load the exported TensorRT INT8 model
model = YOLO("yolo11s.engine")

# Run inference
result = model.predict("bus.jpg", verbose = True)