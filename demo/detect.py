import numpy as np
import cv2
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cv2AddChineseText(img, text, position, textColor=(0, 0, 255), textSize=30):
    """
    在图片上添加中文文字
    :param img: OpenCV图片（numpy.ndarray）
    :param text: 添加的文字
    :param position: 文字添加的位置
    :param textColor: 文字颜色
    :param textSize: 文字大小
    :return: 添加文字后的图片
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)

    # 设置字体和大小
    fontStyle = ImageFont.truetype("huawen.ttf", textSize, encoding="utf-8")

    # 绘制文字
    draw.text(position, text, textColor, font=fontStyle)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def detect_cracks(image_path):
    # 加载训练好的模型
    model = YOLO("runs/detect/crack_detection14/weights/best.pt")

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

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

            cls = int(box.cls[0])  # 获取类别ID
            class_name = result.names[cls]  # 获取类别名称

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 添加标签
            label = f"{class_name} {conf:.2f}"
            print(f'label:{class_name}  conf:{conf:.2f} id:{cls} ')
            # cv2.putText(image, label, (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            image = cv2AddChineseText(image, label, (x1, y1 - 30))

    # 保存结果
    output_path = "result.jpg"
    cv2.imwrite(output_path, image)
    print(f"检测结果已保存至: {output_path}")


if __name__ == "__main__":
    # image_path = "./datasets/val/images/5d32bffc-DSCN0957.JPG"
    image_path = "./test/test1.jpg"
    detect_cracks(image_path)
