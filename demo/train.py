import torch
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
first train:2025-2-6
yolo11n  res:runs\detect\crack_detection14\weights\best.pt  
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 85/85 [00:09<00:00,   
    all       1696       2637      0.579      0.523       0.54      0.241


second train:2025-2-6
yolo11s  res:runs\detect\crack_detection4\weights\best.pt
YOLO11s summary (fused): 238 layers, 9,414,348 parameters, 0 gradients, 21.3 GFLOPs
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 66/66 [00:07<00:00,  8.72it/s]
all       2097       3368      0.559       0.55      0.554      0.258
纵向裂缝        570        822       0.51      0.499      0.471      0.211
横向裂缝        449        784       0.48      0.395      0.401      0.157
龟裂        938       1237      0.621      0.657      0.684      0.358
坑洞        303        525      0.624      0.649       0.66      0.306
"""


def train_model():
    # 检查GPU是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 使用更大的模型
    model = YOLO("yolo11m.pt")  # 从n改为m，使用更大的模型

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
