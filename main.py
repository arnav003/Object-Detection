from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.predict(source="/mnt/d/Software/Python/Python Projects/Object_Detection/cctv.mp4", save=True)