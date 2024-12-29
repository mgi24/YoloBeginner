from ultralytics import YOLO
from mss import mss
import numpy as np
import cv2 as cv


model = YOLO('yolo11s.pt') #pilih model https://docs.ultralytics.com/tasks/detect/

#Crop layar
cropx = 416
cropy = 288
gamex=448
gamey=448

#masukkan data crop (khusus mss)
monitor  = {"top":cropy,"left":cropx, "width":gamex, "height":gamey, "monitor":0}
sct = mss()

#loop input frame ke yolo
def main():
    while True:
        image = sct.grab(monitor)#mengambil gambar dari layar BGRA
        image = np.array(image)#convert ke array
        frame_bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)#hapus alpha channel
        
        scanner(frame_bgr)#deteksi yolo
        if cv.waitKey(1) & 0xFF == 27:#debugger opencv
            break

# #deteksi yolo dan output deteksi
def scanner(frame):
    
    results = model.predict(
        #INFERENCE SETTINGS
        source=frame, imgsz=448, 
        device="cuda:0",  classes=[0,2],
        
        #VISUALIZATION SETTINGS
        verbose=False,
        stream=True) #deteksi yolo
    
    #array output data
    position = []
    confidence = []
    cls = []
    #mengambil data per frame
    
    for r in results:
        output = r.boxes.cpu().numpy()#pindah data dari gpu ke CPU dan convert ke numpy
        
        for data in output:
            position.append(data.xyxy[0])#masukkan data posisi
            confidence.append(data.conf[0])#masukkan data confidence
            cls.append(data.cls[0])#masukkan data class
    
    for i, data in enumerate(position):#ambil data per object per frame
        x1, y1, x2, y2 = data#ambil data posisi
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)#convert ke int agar bisa di draw di opencv
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)#draw rectangle
        
        label = f"{int(cls[i])}: {confidence[i]:.1f}"#cls[i] dan confidence[i] adalah data class dan confidence
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        
    cv.imshow("frame", frame)#output frame opencv

main()