"""
划分训练和验证
"""
import os
import random
import shutil


def split_data(source_folder, train_folder, val_folder, split_ratio=0.6):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            random.shuffle(files)
            split_index = int(len(files) * split_ratio)
            train_files = files[:split_index]
            test_files = files[split_index:]

            for file_name in train_files:
                src = os.path.join(folder_path, file_name)
                dest = os.path.join(train_folder, folder_name, file_name)
                if not os.path.exists(os.path.join(train_folder, folder_name)):
                    os.makedirs(os.path.join(train_folder, folder_name))
                shutil.copy(src, dest)

            for file_name in test_files:
                src = os.path.join(folder_path, file_name)
                dest = os.path.join(val_folder, folder_name, file_name)
                if not os.path.exists(os.path.join(val_folder, folder_name)):
                    os.makedirs(os.path.join(val_folder, folder_name))
                shutil.copy(src, dest)


source_folder = "data"
train_folder = "./data_train_val/train"
val_folder = "./data_train_val/val"
split_data(source_folder, train_folder, val_folder)
