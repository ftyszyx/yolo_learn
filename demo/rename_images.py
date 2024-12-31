import os
from pathlib import Path


def rename_images(directory="pics"):
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在！")
        return

    # 获取所有图片文件
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

    # 排序文件名以确保重命名的一致性
    image_files.sort()

    # 重命名文件
    for index, old_name in enumerate(image_files, 1):
        # 获取文件扩展名
        file_ext = Path(old_name).suffix
        # 创建新文件名 (例如: 001.jpg, 002.jpg, ...)
        new_name = f"{index:03d}{file_ext}"

        # 构建完整的文件路径
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)

        # 如果新文件名已存在，跳过该文件
        if os.path.exists(new_path):
            print(f"跳过 {old_name}: 文件 {new_name} 已存在")
            continue

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {old_name} -> {new_name}")

    print(f"\n完成！共重命名 {len(image_files)} 个文件。")


if __name__ == "__main__":
    rename_images()
