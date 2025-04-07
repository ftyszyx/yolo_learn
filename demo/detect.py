import numpy as np
import cv2
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def detect_cracks(image_path,output_path):
    # 加载训练好的模型
    # model = YOLO("runs/detect/crack_detection8/weights/best.pt")
    # model = YOLO("best_yolo8n.pt")
    model = YOLO("./model/yolo11x.pt")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    # model = YOLO("yolo11s.pt")

    # get all image in image_path
    image_list = os.listdir(image_path)
    for image_name in image_list:
        pic_path= os.path.join(image_path, image_name)
        # 读取图片
        image = cv2.imread(pic_path)
        # 运行检测
        results = model(image)

        # 在图片上标注检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # 获取置信度
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]  # 获取类别名称
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 添加标签
                label = f"{class_name} {conf:.2f}"
                print(f'label:{class_name}  conf:{conf:.2f} id:{cls} ')
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 保存结果
        out_pic_path= os.path.join(output_path, image_name)
        cv2.imwrite(out_pic_path, image)
        print(f"检测结果已保存至: {out_pic_path}")


if __name__ == "__main__":
    # image_path = "./test/test2.jpg"
    # image_path = "./test/4.jpg"
    # image_path = "./test/mypic2.jpg"
    detect_cracks("./test","./output")
