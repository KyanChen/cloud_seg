# Copyright (c) OpenMMLab. All rights reserved.
import glob
import sys
import mmengine
import numpy as np
import tqdm

sys.path.insert(0, sys.path[0]+'/../')
import os
from argparse import ArgumentParser

import cv2
from mmengine.model.utils import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot


def main():
    parser = ArgumentParser()
    # parser.add_argument('--folder', default='/expand_data/datasets/ZY_cloud/S3_trainset/total/img', help='Video file or webcam id')
    parser.add_argument('--folder', default='results/error_img',
                        help='Video file or webcam id')
    parser.add_argument('--config', default='../my_configs/segformer_mit-b1_8xb2-160k_cloud-512x512.py', help='Config file')
    parser.add_argument('--checkpoint', default='results/CS_wo_backbone/epoch_40.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default=None,
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--show', default=False, action='store_true', help='Whether to show draw result')
    parser.add_argument(
        '--show-wait-time', default=0.1, type=int, help='Wait time after imshow')
    parser.add_argument(
        '--output_folder', default='results/vis_error_wobackbone', type=str, help='Output video file path')
    parser.add_argument(
        '--output-fourcc',
        default='MJPG',
        type=str,
        help='Fourcc of the output video')
    parser.add_argument(
        '--output-fps', default=-1, type=int, help='FPS of the output video')
    parser.add_argument(
        '--output-height',
        default=-1,
        type=int,
        help='Frame height of the output video')
    parser.add_argument(
        '--output-width',
        default=-1,
        type=int,
        help='Frame width of the output video')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    assert args.show or args.output_folder, \
        'At least one output should be enabled.'

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    files = glob.glob(args.folder + '/*.jpg')
    os.makedirs(args.output_folder, exist_ok=True)
    if args.palette is not None:
        model.dataset_meta['palette'] = args.palette
    # start looping
    for file in tqdm.tqdm(files):
        # test a single image
        result = inference_model(model, file)

        # blend raw image and prediction
        draw_img = show_result_pyplot(model,
                                      file,
                                      result,
                                      opacity=0.8,
                                      draw_gt=True,
                                      # save_dir=args.output_folder,
                                      show=False
                                      )

        if args.show:
            cv2.imshow('video_demo', draw_img)
            cv2.waitKey(args.show_wait_time)
        if args.output_folder:
            ori_img = cv2.imread(file)
            ori_img = cv2.resize(ori_img, (draw_img.shape[1], draw_img.shape[0]))
            draw_img = np.concatenate((ori_img, draw_img))
            cv2.imwrite(args.output_folder + '/' + os.path.basename(file).replace('.jpg', '.png'), draw_img)


if __name__ == '__main__':
    main()
