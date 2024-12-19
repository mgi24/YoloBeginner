import cv2 as cv
import torch

build = cv.getBuildInformation()
if 'CUDA' in build:
    print("OpenCV is built with CUDA support.")
if torch.cuda.is_available():
    print("Torch is using CUDA.")