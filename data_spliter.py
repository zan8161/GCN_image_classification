import os
import shutil
import random


path = "D:\\Code\\py\\GNN\\data\\animal\\train\\raw"


for folder in os.listdir(path):
    filelist = []
    l = os.walk(path + "\\" + folder)
    for subfolder, _, file in l:
        for sfile in file:
            filelist.append(os.path.join(subfolder, sfile))

    num = int(0.2 * len(filelist))
    mfilelist = random.sample(filelist, k=num)

    for file in mfilelist:
        shutil.move(file, "D:\\Code\\py\\GNN\\data\\animal\\test\\raw" + "\\" + folder)
