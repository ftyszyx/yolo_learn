# 从xlabeling 标注的json文件中提取图片和标注信息，准备训练和验证和test数据集
import os
import shutil
import json
from sklearn.model_selection import train_test_split
g_class_2_label_map = {
    "纵向裂缝": 0,
    "横向裂缝": 1,
    "龟裂": 2,
    "坑洞": 3,
}
g_datalist = [
    "F:/BaiduNetdiskDownload/train_data/Japan_road",
    "F:/BaiduNetdiskDownload/train_data/my_label"
]


def write_label_file(label_filelist, out_dir, sub_name):
    images_dir = os.path.join(out_dir, sub_name, "images")
    labels_dir = os.path.join(out_dir, sub_name, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for label_file in label_filelist:
        parent_dir = os.path.dirname(label_file)
        with open(label_file, "r") as f:
            data = json.load(f)
            image_file_name = data["imagePath"]
            image_width = data["imageWidth"]
            image_height = data["imageHeight"]
            shape_list = data["shapes"]
            image_src_name = os.path.join(parent_dir, image_file_name)
            image_dest_name = os.path.join(images_dir, image_file_name)
            # print(f"copy image:{image_src_name} to {image_dest_name}")
            shutil.copy2(image_src_name, image_dest_name)
            label_text_file = os.path.join(
                labels_dir, os.path.splitext(image_file_name)[0]+".txt")
            with open(label_text_file, "w") as f:
                for shape in shape_list:
                    label = shape["label"]
                    if label not in g_class_2_label_map:
                        continue
                    class_id = g_class_2_label_map[label]
                    points = shape["points"]
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x3, y3 = points[2]
                    x4, y4 = points[3]
                    x1 = round(x1/image_width, 2)
                    y1 = round(y1/image_height, 2)
                    x2 = round(x2/image_width, 2)
                    y2 = round(y2/image_height, 2)
                    x3 = round(x3/image_width, 2)
                    y3 = round(y3/image_height, 2)
                    x4 = round(x4/image_width, 2)
                    y4 = round(y4/image_height, 2)
                    f.write(
                        f"{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
                # print(f"write label file:{label_text_file}")
                f.close()


def clean_dataset():
    # 获取当前工作目录
    current_dir = os.getcwd()
    # 创建dataset目录及其子目录
    dataset_dir = os.path.join(current_dir, "datasets")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    return dataset_dir


def prepare_dataset(datadir, out_dir):
    # 获取所有已标注的图片文件
    label_files = []
    for f in os.listdir(datadir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            # 检查是否有对应的标注文件
            label_file = os.path.join(
                datadir, os.path.splitext(f)[0] + ".json")
            if os.path.exists(label_file):
                label_files.append(label_file)
    if not label_files:
        print("错误：没有找到已标注的图片！请先使用LabelImg进行标注。")
        return
    # 分割训练集和验证集
    train_val_files, test_files = train_test_split(
        label_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(
        train_val_files, test_size=0.2, random_state=42)
    write_label_file(train_files, out_dir, "train")
    write_label_file(val_files, out_dir, "val")
    write_label_file(test_files, out_dir, "test")
    print(f"数据集准备完成！")
    print(f"训练集：{len(train_files)}张图片")
    print(f"验证集：{len(val_files)}张图片")
    print(f"测试集：{len(test_files)}张图片")


if __name__ == "__main__":
    out_dir = clean_dataset()
    for datadir in g_datalist:
        prepare_dataset(datadir, out_dir)
