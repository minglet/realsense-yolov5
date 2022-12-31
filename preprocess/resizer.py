from PIL import Image
import os
file_list = os.listdir("C:/Users/user/OneDrive - Chonnam National University/바탕 화면/data_minah/wait")
img_list = [file for file in file_list if ".jpg" in file]
print(img_list)

BASE_DIR = "C:/Users/user/OneDrive - Chonnam National University/바탕 화면/data_minah/wait"

for i, img_file in enumerate(img_list):
    img = Image.open(BASE_DIR + "\\" + img_file).convert("RGB")
    # img2 = img.transpose(Image.ROTATE_270)
    new_img = img.resize((1280, 720), Image.ANTIALIAS)
    new_img.save(BASE_DIR + "\\" + "{}.jpg".format(img_file.replace(".jpg", "")), format='jpeg', quality=90)