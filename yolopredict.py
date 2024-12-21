from ultralytics import YOLO

model = YOLO('yolo11s.pt')

model.predict(
    #INFERENCE SETTINGS
    source="screen", 
    conf=0.8,
    # iou=0.7, 
    # imgsz=448, 
    # half=False,
    # device="cuda:0", 
    # batch =1,
    # max_det=300,
    # vid_stride = 1,
    # stream_buffer=False,
    # visualize = False,
    # agnostic_nms=False, 
    # classes=[0],
    # project=None,
    # Name=None,
    
    # #VISUALIZATION SETTINGS
    show = True,
    # save=False,
    # save_frames=False,
    # save_txt=False,
    # save_conf=False,
    # save_crop=False,
    # show_labels=True,
    # show_conf=True,
    # show_boxes=True,
    # line_width=None,
    # verbose=False,
    # stream=True 
    ) #deteksi yolo