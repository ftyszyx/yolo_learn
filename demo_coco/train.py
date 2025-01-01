from ultralytics import YOLO


def train_model():
    model = YOLO("../model/yolo11n.pt")
    results = model.train(data="coco.yaml", epochs=100, imgsz=640)

if __name__ == "__main__":
    train_model()
