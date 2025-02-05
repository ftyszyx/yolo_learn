import os
import json
import shutil
from PIL import Image
g_output_dir = "F:/BaiduNetdiskDownload/train_data"
g_dataset_list = [
    {
        "image_dir": "F:/BaiduNetdiskDownload/dat/Japan_road/Japan_road/images",
        "annotation_dir": "F:/BaiduNetdiskDownload/dat/Japan_road/Japan_road/labels",
        "prefix": "Japan_road",
        "class_name_list": ["纵向裂缝", "纵向拼接缝", "错误标签", "横向裂缝",
                            "横向拼接缝", "龟裂", "坑洞", "十字路口模糊", "白线模糊", "井盖"],
        "sub_dir_list": ["test", "train", "val"]
    },
    {
        "image_dir": "F:/BaiduNetdiskDownload/dat/test1/images",
        "annotation_dir": "F:/BaiduNetdiskDownload/dat/test1/labels",
        "prefix": "my_label",
        "class_name_list": ["坑洞"],
        "sub_dir_list": []
    },
    {
        "image_dir": "F:/BaiduNetdiskDownload/dat/yolo8_other/test/images",
        "annotation_dir": "F:/BaiduNetdiskDownload/dat/yolo8_other/test/labels",
        "prefix": "my_data_yolo8_test",
        "class_name_list": ['crocodile_crack', 'lateral_crack', 'longitudinal_crack', 'pothole'],
        "sub_dir_list": []
    },
    {
        "image_dir": "F:/BaiduNetdiskDownload/dat/yolo8_other/train/images",
        "annotation_dir": "F:/BaiduNetdiskDownload/dat/yolo8_other/train/labels",
        "prefix": "my_data_yolo8_train",
        "class_name_list": ['crocodile_crack', 'lateral_crack', 'longitudinal_crack', 'pothole'],
        "sub_dir_list": []
    },
    {
        "image_dir": "F:/BaiduNetdiskDownload/dat/yolo8_other/valid/images",
        "annotation_dir": "F:/BaiduNetdiskDownload/dat/yolo8_other/valid/labels",
        "prefix": "my_data_yolo8_valid",

        "class_name_list": ['crocodile_crack', 'lateral_crack', 'longitudinal_crack', 'pothole'],
        "sub_dir_list": []
    }
]


def convert_yolo_to_label_studio(image_dir, annotation_dir, out_prefix, out_dir, class_name_list):
    pic_num = 0
    for filename in os.listdir(annotation_dir):
        if filename.endswith(".txt"):
            image_name = filename[:-4] + ".jpg"  # 或者 .png, 根据你的图像格式修改
            image_path = os.path.join(image_dir, image_name)
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                image_name = filename[:-4] + ".png"  # 或者 .png, 根据你的图像格式修改
                image_path = os.path.join(image_dir, image_name)
                if not os.path.exists(image_path):
                    print(f"警告: 图像文件 {image_path} 不存在，跳过 {filename}")
                    continue
            # print(f"开始转换 {image_name} path:{image_path}")
            label_file_name = image_name[:-4]+".json"
            label_full_path = os.path.join(image_dir, label_file_name)
            annotation_path = os.path.join(annotation_dir, filename)
            # 获取图像尺寸
            try:

                image = Image.open(image_path)
                width, height = image.size
            except FileNotFoundError:
                print(f"错误: 无法打开图像文件 {image_path}")
                continue

            # 读取 YOLO 标注
            annotations = []
            with open(annotation_path, "r") as f:
                for line in f:
                    linearr = line.strip().split()
                    if len(linearr) <= 5:
                        class_id, center_x, center_y, bbox_width, bbox_height = map(
                            float, linearr)
                        # 将归一化的 YOLO 坐标转换为像素坐标
                        center_x *= width
                        center_y *= height
                        bbox_width *= width
                        bbox_height *= height
                        # 计算左上角坐标
                        x = center_x - bbox_width / 2
                        y = center_y - bbox_height / 2
                        annotations.append({
                            "label": class_name_list[int(class_id)],
                            "shape_type": "rectangle",
                            "flags": {},
                            "points": [[x, y], [x + bbox_width, y], [x+bbox_width, y+bbox_height], [x, y+bbox_height]],
                            "group_id": None,
                            "description": None,
                            "difficult": False,
                            "attributes": {}
                        })
                    else:
                        class_id, x1, y1, x2, y2, x3, y3, x4, y4 = map(
                            float, linearr)
                        annotations.append({
                            "label": class_name_list[int(class_id)],
                            "shape_type": "rectangle",
                            "flags": {},
                            "points": [[x1*width, y1*height], [x2*width, y2*height], [x3*width, y3*height], [x4*width, y4*height]],
                            "group_id": None,
                            "description": None,
                            "difficult": False,
                            "attributes": {}

                        })
                        # 构建 Label Studio JSON 格式
            label_studio_json = {
                "version": "2.5.3",
                "flags": {},
                "shapes": [],
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width,

            }

            for annotation in annotations:
                label_studio_json["shapes"].append(annotation)
            json.dump(label_studio_json, open(label_full_path, "w"), indent=4)
            shutil.copyfile(image_path, os.path.join(
                out_dir, out_prefix+"_"+image_name))
            shutil.copyfile(label_full_path, os.path.join(
                out_dir, out_prefix+"_"+label_file_name))
            pic_num += 1

            # print(f"转换完成，JSON 文件保存在 {label_full_path}")

    return pic_num


if __name__ == "__main__":
    info_log = ""
    pic_num = 0
    for dataset in g_dataset_list:
        dataset_name = dataset["prefix"]
        print(f"开始转换数据集 {dataset_name}")
        output_dir = os.path.join(g_output_dir, dataset_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dataset_pic_num = 0
        if len(dataset["sub_dir_list"]) > 0:
            for sub_dir in dataset["sub_dir_list"]:

                image_dir = os.path.join(dataset["image_dir"], sub_dir)

                annotation_dir = os.path.join(
                    dataset["annotation_dir"], sub_dir)
                dataset_pic_num += convert_yolo_to_label_studio(image_dir, annotation_dir, dataset_name, output_dir,
                                                                dataset["class_name_list"])

        else:
            dataset_pic_num += convert_yolo_to_label_studio(dataset["image_dir"], dataset["annotation_dir"], dataset_name,
                                                            output_dir, dataset["class_name_list"])
        print(f"数据集 {dataset_name} 转换完成，共 {dataset_pic_num} 张图片")
        pic_num += dataset_pic_num
    print(f"所有数据集转换完成，共 {pic_num} 张图片")

    info_log += f"所有数据集转换完成，共 {pic_num} 张图片\n"

    with open(os.path.join(g_output_dir, "export_info.txt"), "w") as f:
        f.write(info_log)
