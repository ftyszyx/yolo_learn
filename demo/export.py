from ultralytics import YOLO
model = YOLO("yolov8n.pt")
# model = YOLO("runs/detect/crack_detection4/weights/best.pt")
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
