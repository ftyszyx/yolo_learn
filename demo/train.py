import torch
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
first train:2025-2-6
yolo11n  res:runs\detect\crack_detection14\weights\best.pt  
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 85/85 [00:09<00:00,   
    all       1696       2637      0.579      0.523       0.54      0.241

"""


def train_model():
    # 检查GPU是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 使用更大的模型
    model = YOLO("yolo11s.pt")  # 从n改为m，使用更大的模型

    # 开始训练
    results = model.train(
        data="data.yaml",
        epochs=100,  # 增加训练轮数
        imgsz=640,  # 增加图片尺寸
        batch=16,  # 减小batch size以适应更大的模型
        name="crack_detection",
        # 学习率调整
        lr0=0.001,  # 降低初始学习率
        lrf=0.0001,  # 降低最终学习率

        conf=0.001,  # 降低置信度阈值，训练时检测更多目标
        hsv_h=0.02,  # 增加hsv_h以增加颜色变化
        optimizer="NAdam",  # 使用自动优化器 默认就是auto
    )

    print("训练完成！")


if __name__ == "__main__":
    train_model()
