from ultralytics import YOLO


def train_model():
    # 创建YOLO模型
    model = YOLO("yolo11n.pt")  # 加载预训练模型

    # 开始训练
    results = model.train(data="data.yaml", epochs=2, imgsz=2048, batch=16, name="crack_detection")  # 训练轮数  # 图片大小  # 批次大小  # 实验名称

    print("训练完成！")


if __name__ == "__main__":
    train_model()
