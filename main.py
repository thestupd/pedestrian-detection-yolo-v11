from ultralytics import YOLO

model = YOLO("model.pt")

model.predict(source="video.mp4", iou=0.5, conf=0.7, save=True, show=True)