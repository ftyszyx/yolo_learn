from ultralytics import YOLO
import cv2
import numpy as np


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

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 添加标签
            label = f"Crack {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 保存结果
    output_path = "result.jpg"
    cv2.imwrite(output_path, image)
    print(f"检测结果已保存至: {output_path}")


if __name__ == "__main__":
    image_path = "./test/020.jpg"
    detect_cracks(image_path)
