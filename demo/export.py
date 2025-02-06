from ultralytics import YOLO
model = YOLO("yolo11s.pt")
model = YOLO("runs/detect/crack_detection4/weights/best.pt")
model.export(format="onnx")
