import os
import time

os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 "
              "sh dist_train.sh "
              "../my_configs/segformer_mit-b1_8xb2-160k_cloud-512x512.py "
              "results/CS_E20230309_0 "
              "4")

