import ultralytics


ultralytics.YOLO("yolov8n-seg.pt", task="segment")
ultralytics.YOLO("yolov8n.pt", task="detect")
