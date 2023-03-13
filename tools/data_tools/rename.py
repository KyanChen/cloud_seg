import glob
import os
import shutil

files = glob.glob('/expand_data/datasets/GFCS/train/rgb/*')
for file in files:
    print(0)
    os.rename(file, file.replace('_rgb', ''))
