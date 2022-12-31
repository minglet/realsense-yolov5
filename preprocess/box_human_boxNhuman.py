import os
import shutil

#make all list
mix = []
box = []
human = []
boxNhuman = []

for i in range(300,501): #need to change the range
    with open('./box_img 600/' + str(i) + '.txt', 'r') as f:
        line = f.readline()
        content_split = line.split()
        if content_split[0] == str(0):
            mix.append("./box_img 600/" + str(i))
        else:
            human.append("./box_img 600/" + str(i))

for x in mix: #in "mix" list will have box or boxNhuman
    with open(x + '.txt','r') as f:
        line = f.readlines()
        a = str(line[:]) #get the whole content in the txt
        a = a.split()
        if "'1" in a:
            boxNhuman.append(str(x))
        else:
            box.append(str(x))

def move_files(file_list, source_path, destination_path): #file_list는 리스트 형태여야함.
    for file in file_list:
        image = file.split('/')[-1] + '.jpg'
        txt = file.split('/')[-1] + '.txt'
        shutil.copy(os.path.join(source_path, image), destination_path)
        shutil.copy(os.path.join(source_path,txt), destination_path)

    return

source_dir = "./box_img 600"

box_dir = "./data/box/"
human_dir = "./data/human/"
boxNhuman_dir = "./data/boxNhuman/"

move_files(box, source_dir, box_dir)
move_files(human, source_dir, human_dir)
move_files(boxNhuman, source_dir, boxNhuman_dir)

# print(human)
# print(box)
# print(boxNhuman)
# print(mix)