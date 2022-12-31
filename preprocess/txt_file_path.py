import os
import glob

image_files = glob.glob("C:/Users/user/aug/*")
image_list = [image_dir for image_dir in image_files]
print(image_list)
f = open("./new_train.txt", 'w')

f.writelines('\n'.join(image_list))
f.close()

