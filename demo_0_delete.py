"""
只保留500个图片
"""
import os
import random
import shutil


def keep_random_images(folder_path, n=500):
    # 遍历主文件夹下的所有子文件夹
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            # 遍历当前子文件夹中的所有图片文件
            image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
            if image_files:
                # 如果图片数量大于n，随机选择要保留的图片，其余删除
                if len(image_files) > n:
                    files_to_keep = random.sample(image_files, n)
                    files_to_delete = [f for f in image_files if f not in files_to_keep]
                    for file_to_delete in files_to_delete:
                        os.remove(os.path.join(subdir_path, file_to_delete))
                    print(f"Deleted {len(files_to_delete)} files in {subdir_path}")


# 指定包含所有图片文件的主文件夹路径
main_folder_path = "data"
keep_random_images(main_folder_path, n=600)
