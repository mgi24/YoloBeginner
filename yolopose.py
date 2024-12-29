from ultralytics import YOLO
import cv2 as cv
import numpy as np
from mss import mss

cropx = 416
cropy = 288
gamex=448
gamey=448

monitor  = {"top":cropy,"left":cropx, "width":gamex, "height":gamey, "monitor":0}
sct = mss()

model = YOLO('yolo11n-pose.pt')

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def scanner(frame):
    global framecount
    
    results = model.predict(source=frame,imgsz=448,device = "cuda:0" ,classes = [0], verbose=True)
    
    keypoint = results[0].keypoints

    nose = 0
    left_eye = 1
    right_eye = 2
    left_ear = 3
    right_ear = 4
    left_shoulder = 5
    right_shoulder = 6
    left_elbow = 7
    right_elbow = 8
    left_wrist = 9
    right_wrist = 10
    left_hip = 11
    right_hip = 12
    left_knee = 13
    right_knee = 14
    left_ankle = 15
    right_ankle = 16
    xy = keypoint.xy.cpu().numpy()#all information
    
    if(xy.size != 0):
        for point in xy:#each detected person
            for i, (x,y) in enumerate (point):#each keypoint
                if i == nose and x != 0 and y != 0:
                    cv.putText(frame, 'Nose', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    print(f"Nose = {x} {y}")
                elif i == left_eye and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Left Eye', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == right_eye and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Right Eye', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == left_ear and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Left Ear', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == right_ear and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Right Ear', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == left_shoulder and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Left Shoulder', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == right_shoulder and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Right Shoulder', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == left_elbow and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Left Elbow', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == right_elbow and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Right Elbow', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == left_wrist and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Left Wrist', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == right_wrist and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Right Wrist', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == left_hip and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Left Hip', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == right_hip and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Right Hip', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == left_knee and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Left Knee', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == right_knee and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Right Knee', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == left_ankle and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Left Ankle', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif i == right_ankle and x != 0 and y != 0:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv.putText(frame, 'Right Ankle', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow("frame", frame)
    
def main():
    cv.startWindowThread()
    
    while True:
        
        image = sct.grab(monitor)
        image = np.array(image)
        frame_bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
        scanner(frame_bgr)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
main()