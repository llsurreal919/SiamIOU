# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import time

import numpy as np

from SiamIOU import MOTracker
from Util import Util

def main(args):
    if args.dataset == 'UA-DETRAC':
        detections_base = "./UA-DETRAC/UA_test"
        img_path = 'path/to/images'
        file_list = os.listdir(detections_base)
        data_ = './output/EB/test'
        print('Operation method: {}'.format(data_.split('/')[-1]))
        Speed = []
        util = Util()
        mot_tracker = MOTracker(args)
        for x in file_list:
            if x == 'EB':continue
            dirbase = os.path.join(detections_base, x)
            for y in os.listdir(dirbase):
                subdir = os.path.join(dirbase, y, x+'_Det_EB.txt')
                det_file = subdir
                detections = np.genfromtxt(det_file, delimiter=' ')
                try:
                    detections.shape[1]
                except:
                    detections = detections.reshape(-1,detections.shape[0])
                print('/'.join(det_file.split('/')[-3:]))
                dets = util.load_mot(detections)
                frame_name_list = util.init_video(img_path, x)
                start_time = time.time()
                out = mot_tracker.mot_track(dets, frame_name_list)
                end_time = time.time()
                num_frames = len(dets)
                speed = num_frames / (end_time - start_time)
                Speed.append(speed)
                a = x + '.txt'
                dirout = os.path.join(data_, x,y)
                if not os.path.isdir(dirout):
                    os.makedirs(dirout)
                target_dir = os.path.join(dirout, a)
                util.save_to_ua(target_dir, out)
        print(sum(Speed)/len(Speed))
    else:
        print('Wrong Input...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file', default='path/to/config.yaml')
    parser.add_argument('--snapshot', type=str, help='model name', default='path/to/model.pth')
    parser.add_argument('--dataset', type=str, help='UA-DETRAC', default='UA-DETRAC')
    args = parser.parse_args()
    main(args)
