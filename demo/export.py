from ultralytics import YOLO
# model = YOLO("yolo11n.pt")
model = YOLO("best.pt")
# export rknn 
# 'name' can be one of rk3588, rk3576, rk3566, rk3568, rk3562, rv1103, rv1106, rv1103b, rv1106b, rk2118

# model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
model.export(format="rknn", imgsz=640, name="rk3588")
