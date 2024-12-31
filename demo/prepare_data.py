import os
import shutil
from sklearn.model_selection import train_test_split


def prepare_dataset(pics_dir="pics", labels_dir="pics"):
    """准备训练数据集"""
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    # 创建dataset目录及其子目录
    dataset_dir = os.path.join(current_dir, "dataset")
    train_images_dir = os.path.join(dataset_dir, "train", "images")
    train_labels_dir = os.path.join(dataset_dir, "train", "labels")
    val_images_dir = os.path.join(dataset_dir, "val", "images")
    val_labels_dir = os.path.join(dataset_dir, "val", "labels")

    # 创建必要的目录
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 获取所有已标注的图片文件
    image_files = []
    for f in os.listdir(pics_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            # 检查是否有对应的标注文件
            label_file = os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt")
            if os.path.exists(label_file):
                image_files.append(f)

    if not image_files:
        print("错误：没有找到已标注的图片！请先使用LabelImg进行标注。")
        return

    # 分割训练集和验证集
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    # 处理训练集
    for img_file in train_files:
        # 复制图片
        src_img = os.path.join(pics_dir, img_file)
        dst_img = os.path.join(train_images_dir, img_file)
        shutil.copy2(src_img, dst_img)

        # 复制标注文件
        src_label = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
        dst_label = os.path.join(train_labels_dir, os.path.splitext(img_file)[0] + ".txt")
        shutil.copy2(src_label, dst_label)

    # 处理验证集
    for img_file in val_files:
        # 复制图片
        src_img = os.path.join(pics_dir, img_file)
        dst_img = os.path.join(val_images_dir, img_file)
        shutil.copy2(src_img, dst_img)

        # 复制标注文件
        src_label = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
        dst_label = os.path.join(val_labels_dir, os.path.splitext(img_file)[0] + ".txt")
        shutil.copy2(src_label, dst_label)

    print(f"数据集准备完成！")
    print(f"训练集：{len(train_files)}张图片")
    print(f"验证集：{len(val_files)}张图片")
    print(f"\n数据集目录结构：")
    print(f"dataset/")
    print(f"  ├─train/")
    print(f"  │  ├─images/  ({len(train_files)}张图片)")
    print(f"  │  └─labels/  ({len(train_files)}个标注文件)")
    print(f"  └─val/")
    print(f"     ├─images/  ({len(val_files)}张图片)")
    print(f"     └─labels/  ({len(val_files)}个标注文件)")


if __name__ == "__main__":
    prepare_dataset()
