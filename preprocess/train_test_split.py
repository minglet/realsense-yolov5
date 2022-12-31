import os
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

image_files = glob("C:/Users/user/aug/*.jpg")
images = [name.replace(".jpg","") for name in image_files]

train_names, test_names = train_test_split(images, test_size=0.2, random_state=42, shuffle=True)
val_names, test_names = train_test_split(test_names, test_size=0.5, random_state=42, shuffle=True)

def batch_move_files(file_list, source_path, destination_path):
    for file in file_list:
        image = file.split('/')[-1] + '.jpg'
        txt = file.split('/')[-1] + '.txt'
        shutil.copy(os.path.join(source_path, image), destination_path)
        shutil.copy(os.path.join(source_path, txt), destination_path)
    return

source_dir = "./"

test_dir = "C:/Users/user/aug/test/"
train_dir = "C:/Users/user/aug/train/"
val_dir = "C:/Users/user/aug/val/"
batch_move_files(train_names, source_dir, train_dir)
batch_move_files(test_names, source_dir, test_dir)
batch_move_files(val_names, source_dir, val_dir)
