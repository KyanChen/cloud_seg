import glob

import cv2
import numpy as np

in_path = '/expand_data/datasets/ZY_cloud/S3_trainset/total/label/'
files = glob.glob(in_path+'/*.png')
for file in files:
    img = cv2.imread(file)
    img[img==128] = 1
    img[img==255] = 2
    print(np.sum(img>2))
    img[img>2] = 0
    cv2.imwrite(file, img)

