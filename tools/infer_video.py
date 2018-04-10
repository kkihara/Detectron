#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import json

import numpy as np
import cv2
from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
import utils.keypoints as keypoint_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'video_file', help='path to mp4', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    cam_name = os.path.splitext(os.path.basename(args.video_file))[0]

    frame_poses = []
    for frame_num, im in get_frames(args.video_file):
        logger.info('Processing frame num: {}'.format(frame_num))
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        poses = [convert_pose_data(pose) for pose in cls_keyps[1]]
        frame_poses.append(poses)
    with open(os.path.join(args.output_dir, cam_name + '-frame-poses.txt'), 'w') as f:
        f.write('\n'.join(get_json(frame_pose) for frame_pose in frame_poses))


KEYPS_IDX = [
    'nose',
    'neck', # special case for neck
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'bkg', # special case for bkg
]


def get_json(data):
    return json.dumps(data, indent=None, separators=(',', ':'))


coco_keyps, _ = keypoint_utils.get_keypoints()

KEYP_THRESH = 2

def convert_pose_data(data):
    new_data = np.full(38, np.nan)

    for i, keyp in enumerate(KEYPS_IDX):
        idx = i * 2
        if keyp == 'bkg':
            pass
        elif keyp == 'neck':
            sc_neck = np.minimum(
                data[2, coco_keyps.index('right_shoulder')],
                data[2, coco_keyps.index('left_shoulder')])
            if sc_neck > KEYP_THRESH:
                left_shoulder = data[:2, coco_keyps.index('left_shoulder')]
                right_shoulder = data[:2, coco_keyps.index('right_shoulder')]
                new_data[idx:idx+2] = (left_shoulder + right_shoulder) / 2.0
        else:
            if data[2, coco_keyps.index(keyp)] > KEYP_THRESH:
                new_data[idx:idx+2] = data[:2, coco_keyps.index(keyp)]
    return new_data.tolist()


def get_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_num, frame
        frame_num += 1


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)