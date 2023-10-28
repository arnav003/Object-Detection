from ultralytics import YOLO
model = YOLO("yolov8l.pt")
model.predict(source='drone_camera.mp4', save=True)

#yolo task=detect mode=predict model=yolov8n.pt source=cctv.mp4 conf=0.5 show=True