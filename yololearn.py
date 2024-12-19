from ultralytics import YOLO
from mss import mss
import numpy as np
import cv2 as cv

model = YOLO('yolo11n.pt')
cropx = 416
cropy = 288
gamex=448
gamey=448

monitor  = {"top":cropy,"left":cropx, "width":gamex, "height":gamey, "monitor":0}
sct = mss()


def main():
    while True:
        image = sct.grab(monitor)
        image = np.array(image)
        frame_bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
        scanner(frame_bgr)
        if cv.waitKey(1) & 0xFF == 27:
            break

def scanner(frame):
    
    results = model.predict(source=frame, imgsz=448, device="cuda:0",  verbose=False, stream=True, conf=0.3, iou=0.7, batch =1, max_det=300)
    position = []
    confidence = []
    cls = []
    for r in results:
        output = r.boxes.cpu().numpy()
        for data in output:
            position.append(data.xyxy)
            confidence.append(data.conf[0])
            cls.append(data.cls[0])
    for i, data in enumerate(position):
        x1, y1, x2, y2 = data[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(cls[i])
    
        
    cv.imshow("frame", frame)
main()