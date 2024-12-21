import cv2 as cv
import bettercam
from ultralytics import YOLO


camera = bettercam.create(output_color="BGR")
camera.start()

model = YOLO('yolo11n.pt')
#loop input frame ke yolo
def main():
    while True:
        
        image = camera.get_latest_frame()
        #frame_bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)#hapus alpha channel
        scanner(image)#deteksi yolo
        if cv.waitKey(1) & 0xFF == 27:#debugger opencv
            break

#deteksi yolo dan output deteksi
def scanner(frame):
    
    results = model.predict(
        source=frame, imgsz=448, device="cuda:0",  
        verbose=False, stream=True, conf=0.3, 
        iou=0.7, batch =1, max_det=300) #deteksi yolo
    
    #array output data
    position = []
    confidence = []
    cls = []

    #mengambil data per frame
    for r in results:
        output = r.boxes.cpu().numpy()#pindah data dari gpu ke CPU dan convert ke numpy
        for data in output:
            position.append(data.xyxy)#masukkan data posisi
            confidence.append(data.conf[0])#masukkan data confidence
            cls.append(data.cls[0])#masukkan data class
    for i, data in enumerate(position):#ambil data per object per frame
        x1, y1, x2, y2 = data[0]#ambil data posisi
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)#convert ke int agar bisa di draw di opencv
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)#draw rectangle
        print(cls[i])
    
        
    cv.imshow("frame", frame)#output frame opencv

main()