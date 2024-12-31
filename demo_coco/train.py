from ultralytics import YOLO

model = YOLO("../model/yolo11n.pt")

results = model.train(data="coco.yaml", epochs=100, imgsz=640)
