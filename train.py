from ultralytics import YOLO


def train_model():
    # 创建YOLO模型
    model = YOLO("yolo11n.pt")  # 加载预训练模型

    # 开始训练
    results = model.train(
        data="data.yaml",
        epochs=2,
        imgsz=1024,          # 增大尺寸以适应不同大小的图片
        batch=8,
        patience=20,
        name="crack_detection",
        # 数据增强参数
        augment=True,
        mixup=0.1,
        mosaic=0.5,
        degrees=10.0,
        scale=0.5,
        # 自适应参数
        rect=True,          # 使用矩形训练，减少填充
        overlap_mask=True,  # 允许边界框重叠
    )
    
    print("训练完成！")


if __name__ == "__main__":
    train_model()
