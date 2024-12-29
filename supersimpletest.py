from ultralytics import YOLO

model = YOLO('yolo11s.pt')

model.predict(source=0, show=True)