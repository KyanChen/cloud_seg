import os
import time

os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 "
              "sh dist_train.sh "
              "../my_config/segformer_mit-b0_512x512_160k_cloud.py "
              "results/CS_E20230311_0 "
              "4")

