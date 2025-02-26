import torch
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""


2028-2-15
yolo11n  res:runs\detect\crack_detection5\weights\best.pt
YOLO11n summary (fused): 238 layers, 2,582,932 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 66/66 [00:06<00:00, 10.02it/s]
                   all       2097       3368      0.571      0.511      0.539      0.252
       zongxiang_crack        570        822      0.533      0.449      0.472      0.216
       hengxiang_crack        449        784       0.48      0.375      0.406       0.16
          guilie_crack        938       1237      0.658       0.64       0.68      0.359
        kengdong_crack        303        525      0.615      0.579      0.597      0.272

yolo11n  res:runs\detect\crack_detection\weights\best.pt
        YOLO11n summary (fused): 238 layers, 2,582,932 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 30/30 [00:06<00:00,  4.92it/s]
                   all       2097       3368      0.574      0.508      0.528      0.242
       zongxiang_crack        570        822      0.529      0.448       0.46      0.203
       hengxiang_crack        449        784      0.455      0.395      0.386      0.151
          guilie_crack        938       1237      0.672      0.629      0.672      0.343
        kengdong_crack        303        525      0.638      0.562      0.595      0.272
Speed: 0.1ms preprocess, 0.6ms inference, 0.0ms loss, 0.6ms postprocess per image


yolo11s Validating runs\detect\crack_detection2\weights\best.pt...
Ultralytics 8.3.75 🚀 Python-3.10.16 torch-2.5.1 CUDA:0 (NVIDIA GeForce RTX 4070 Ti SUPER, 16376MiB)
YOLO11s summary (fused): 238 layers, 9,414,348 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 30/30 [00:08<00:00,  3.67it/s]
                   all       2097       3368      0.572      0.553      0.557      0.262
       zongxiang_crack        570        822       0.53      0.485       0.48       0.22
       hengxiang_crack        449        784      0.471      0.432      0.404      0.159
          guilie_crack        938       1237      0.648      0.669      0.695      0.358
        kengdong_crack        303        525      0.637      0.627       0.65      0.309


yolo8n  res:runs\detect\crack_detection3\weights\best.pt
Ultralytics 8.3.75 🚀 Python-3.10.16 torch-2.5.1 CUDA:0 (NVIDIA GeForce RTX 4070 Ti SUPER, 16376MiB)
Model summary (fused): 168 layers, 3,006,428 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 30/30 [00:06<00:00,  4.93it/s]
                   all       2097       3368      0.564      0.513      0.535      0.249
       zongxiang_crack        570        822      0.529      0.475      0.489       0.22
       hengxiang_crack        449        784      0.475       0.38      0.391      0.156
          guilie_crack        938       1237      0.654      0.631      0.671      0.347
        kengdong_crack        303        525      0.599      0.568       0.59      0.275
Speed: 0.1ms preprocess, 0.6ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to runs\detect\crack_detection3
训练完成！

yolo8n Ultralytics 8.3.75 🚀 Python-3.10.16 torch-2.5.1 CUDA:0 (NVIDIA GeForce RTX 4070 Ti SUPER, 16376MiB)
Model summary (fused): 168 layers, 3,006,428 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 36/36 [00:10<00:00,  3.42it/s]
                   all       2563       5148      0.616      0.552      0.566      0.259
       zongxiang_crack        811       1421      0.601      0.547      0.547       0.24
       hengxiang_crack        877       1862      0.578      0.513      0.525      0.221
          guilie_crack       1015       1356      0.647      0.619      0.641      0.318
        kengdong_crack        310        509      0.637      0.527      0.554      0.256
Speed: 0.2ms preprocess, 1.5ms inference, 0.0ms loss, 0.7ms postprocess per image
Results saved to runs\detect\crack_detection8
"""


def train_model():
    # 检查GPU是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 使用更大的模型
    # model = YOLO("yolov8n.pt")  # 从n改为m，使用更大的模型
    model = YOLO("best.pt")

    # 开始训练
    results = model.train(
        data="data.yaml",
        epochs=80,  # 增加训练轮数
        imgsz=1024,  # 增加图片尺寸
        batch=36,  # 减小batch size以适应更大的模型
        name="crack_detection",
        # 学习率调整
        lr0=0.001,  # 降低初始学习率
        lrf=0.01,  # 降低最终学习率

        conf=0.001,  # 降低置信度阈值，训练时检测更多目标
        hsv_h=0.02,  # 增加hsv_h以增加颜色变化
        # optimizer="NAdam",  # 使用自动优化器 默认就是auto
    )

    print("训练完成！")


if __name__ == "__main__":
    train_model()
