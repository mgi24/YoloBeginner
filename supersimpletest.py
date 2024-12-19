from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.predict(source="screen", show=True)