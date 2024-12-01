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
    model = YOLO("yolo11l.pt")  # 从n改为m，使用更大的模型

    # 开始训练
    results = model.train(
        data="data.yaml",
        epochs=500,  # 增加训练轮数
        imgsz=1280,  # 增加图片尺寸
        batch=8,  # 减小batch size以适应更大的模型
        patience=100,  # 增加早停耐心值
        name="crack_detection",
        # 学习率调整
        lr0=0.001,  # 降低初始学习率
        lrf=0.0001,  # 降低最终学习率
        warmup_epochs=5,  # 增加预热轮数
        # 数据增强
        augment=True,
        mixup=0.1,  # 减小mixup强度，避免过度增强
        mosaic=0.5,  # 减小mosaic强度
        degrees=10.0,  # 减小旋转角度
        scale=0.5,  # 调整缩放范围
        shear=0.2,  # 减小剪切强度
        perspective=0.0,  # 移除透视变换
        flipud=0.3,  # 减小翻转概率
        fliplr=0.3,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # 优化器设置
        optimizer="AdamW",
        weight_decay=0.0005,  # 减小权重衰减
        # 其他设置
        rect=True,
        overlap_mask=True,
        single_cls=True,  # 如果只关注一个类别，设为True
        close_mosaic=20,
        # 新增参数
        label_smoothing=0.1,  # 添加标签平滑
        cos_lr=True,  # 使用余弦学习率调度
        max_det=100,  # 增加最大检测数
        nms=True,  # 使用NMS
        iou=0.7,  # 提高IOU阈值
        conf=0.001,  # 降低置信度阈值，训练时检测更多目标
    )

    print("训练完成！")


if __name__ == "__main__":
    train_model()
