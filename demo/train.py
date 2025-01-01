import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import torch


def train_model():
    # 检查GPU是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 使用更大的模型
    model = YOLO("yolo11n.pt")  # 从n改为m，使用更大的模型

    # 开始训练
    results = model.train(
        data="data.yaml",
        epochs=50,  # 增加训练轮数
        imgsz=1280,  # 增加图片尺寸
        batch=8,  # 减小batch size以适应更大的模型
        name="crack_detection",
        # 学习率调整
        lr0=0.001,  # 降低初始学习率
        lrf=0.0001,  # 降低最终学习率
        conf=0.001,  # 降低置信度阈值，训练时检测更多目标

        # hsv_h=0.015,  # 增加hsv_h以增加颜色变化
    )

    print("训练完成！")


if __name__ == "__main__":
    train_model()
