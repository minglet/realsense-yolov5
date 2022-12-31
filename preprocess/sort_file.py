import os

file_path = "C:/Users/user/OneDrive - Chonnam National University/바탕 화면/data_minah/total_data"
file_names = os.listdir(file_path)
file_names

i = 1
for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(i) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1