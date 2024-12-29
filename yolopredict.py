from ultralytics import YOLO

model = YOLO('yolo11s.pt')

model.predict(
    #INFERENCE SETTINGS
    source="screen", 
    # conf=0.3,
    # iou=0.9, 
    imgsz=448, 
    half=False,
    device="0", 
    batch =1,
    max_det=300,
    vid_stride = 1,
    stream_buffer=False,
    visualize = False,
    agnostic_nms=False, 
    # classes=[0],
    # project="Detektor",
    # name="orang",
    
    # #VISUALIZATION SETTINGS
    show = True,
    # save=True,
    # save_frames=True,
    # save_txt=True,
    # save_conf=True,
    # save_crop=True,
    # show_labels=True,
    # show_conf=False,
    # show_boxes=False,
    line_width=1,
    verbose=False,
    # stream=True 
    ) #deteksi yolo