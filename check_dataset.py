import os
import cv2
import numpy as np
from collections import Counter
from pathlib import Path

def check_dataset():
    train_images = os.path.join('dataset', 'train', 'images')
    train_labels = os.path.join('dataset', 'train', 'labels')
    val_images = os.path.join('dataset', 'val', 'images')
    val_labels = os.path.join('dataset', 'val', 'labels')
    
    class_counts = Counter()
    box_sizes = []
    aspect_ratios = []
    
    def analyze_label(label_path, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            return
        
        img_height, img_width = img.shape[:2]
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    class_counts[int(class_id)] += 1
                    
                    # 计算实际像素尺寸
                    w_pixels = width * img_width
                    h_pixels = height * img_height
                    box_sizes.append(w_pixels * h_pixels)
                    aspect_ratios.append(w_pixels / h_pixels if h_pixels > 0 else 0)
                except:
                    print(f"警告：标签文件 {label_path} 格式错误")
    
    # 分析训练集
    print("分析训练集...")
    for label_file in os.listdir(train_labels):
        if label_file.endswith('.txt'):
            img_file = Path(label_file).stem + '.jpg'
            label_path = os.path.join(train_labels, label_file)
            img_path = os.path.join(train_images, img_file)
            if os.path.exists(img_path):
                analyze_label(label_path, img_path)
    print("\n数据集统计：")
    print("-" * 50)
    print("类别分布：")
    for class_id, count in class_counts.items():
        print(f"类别 {class_id}: {count} 个样本")
    
    if box_sizes:
        print("\n边界框统计：")
        print(f"最小框面积: {min(box_sizes):.0f} 像素")
        print(f"最大框面积: {max(box_sizes):.0f} 像素")
        print(f"平均框面积: {np.mean(box_sizes):.0f} 像素")
        print(f"中位数框面积: {np.median(box_sizes):.0f} 像素")
        
        print("\n宽高比统计：")
        print(f"最小宽高比: {min(aspect_ratios):.2f}")
        print(f"最大宽高比: {max(aspect_ratios):.2f}")
        print(f"平均宽高比: {np.mean(aspect_ratios):.2f}")
    
    # 检查空标签文件
    empty_files = []
    for label_file in os.listdir(train_labels):
        if os.path.getsize(os.path.join(train_labels, label_file)) == 0:
            empty_files.append(label_file)
    
    if empty_files:
        print("\n警告：发现空标签文件：")
        for f in empty_files:
            print(f"- {f}")
    
    # 检查标签值是否合法
    invalid_labels = []
    for label_file in os.listdir(train_labels):
        with open(os.path.join(train_labels, label_file), 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    values = list(map(float, line.strip().split()))
                    if len(values) != 5:
                        invalid_labels.append((label_file, line_num, "格式错误"))
                    elif not (0 <= values[1] <= 1 and 0 <= values[2] <= 1 and 
                             0 <= values[3] <= 1 and 0 <= values[4] <= 1):
                        invalid_labels.append((label_file, line_num, "坐标超出范围"))
                except:
                    invalid_labels.append((label_file, line_num, "数据格式错误"))
    
    if invalid_labels:
        print("\n警告：发现无效标签：")
        for file, line, reason in invalid_labels:
            print(f"- {file} 第{line}行: {reason}")

if __name__ == "__main__":
    check_dataset() 