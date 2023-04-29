from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.predict(source=0, save=True)

#yolo task=detect mode=predict model=yolov8n.pt source=cctv.mp4 conf=0.5 show=True