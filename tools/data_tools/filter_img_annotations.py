import glob
import cv2
import numpy as np

in_path = '/expand_data/datasets/ZY_cloud/S3_trainset/total/'
files = glob.glob(in_path+'/label/*.png')
for file in files:
    img = cv2.imread(file)
    h, w, _ = img.shape
    jpg_file = file.replace('label', 'img').replace('.png', '.jpg')
    jpg = cv2.imread(jpg_file)
    h1, w1, _ = jpg.shape
    if h1 != h or w1 != w:
        print(file)
        jpg = cv2.resize(jpg, (w, h))
        cv2.imwrite(jpg_file, jpg)

