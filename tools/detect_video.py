#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from utils.cython_nms import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, glob, cv2
import argparse
from visualization import draw_bounding_boxes

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__', 'corrosion', 'cable_dangle', 'paint_peel', 'rf_', 'rru_', 'mw_', 'aol', 'platform_mesh', 'antenna_boom', 'lightning_rod')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_32000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'horus': ('horus_trainval',), 'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

NUM_CLASSES = len(CLASSES)

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_path = os.path.join(cfg.DATA_DIR, 'demo_video', 'video_to_images')
    im_file = os.path.join(im_path, image_name + '.jpg')
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print("Object detection for frame: " + image_name)

    # Visualize detections for each class
    CONF_THRESH = 0.6
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        inds = np.where(scores[:, cls_ind] >= CONF_THRESH)[0]
        cls_boxes = boxes[inds, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[inds, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, cfg.TEST.NMS)
        dets = dets[keep, :]
        if len(dets) != 0:
            im = draw_bounding_boxes(im, dets, cls_ind, cls)

        cv2.imwrite(cfg.DATA_DIR + '/output/output_' + image_name, im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", NUM_CLASSES,
                          tag='default', anchor_scales=[8, 16, 32, 64], anchor_ratios=[0.25, 0.5, 1, 2, 4])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['DJI_0623.JPG', 'DJI_0027.JPG', 'DJI_0037.JPG', 'DJI_0043.JPG', 'DJI_0043_XT.JPG', 'DJI_0074_T.JPG', 'DJI_0027_T.JPG', 'DJI_0037_T.JPG', 'DJI_0043_T.JPG', 'DJI_0074.JPG']
    im_path = os.path.join(cfg.DATA_DIR, 'demo_video', 'video_to_images')
    
    files = glob.glob(im_path + '/*.jpg')
    for f in files:
        os.remove(f)
    # Convert from video to images
    # vc = cv2.VideoCapture(os.path.join(cfg.DATA_DIR, 'demo_video', 'Z30_1_10Aug2017.mp4'))
    vc = cv2.VideoCapture(os.path.join(cfg.DATA_DIR, 'demo_video', 'X4S_1_10Aug2017-1.mov'))
    c = 1

    rval, frame = vc.read()

    while rval:
        rval, frame = vc.read()
        cv2.imwrite(os.path.join(im_path, str(c).zfill(7) + '.jpg'), frame)
        c = c + 1
        cv2.waitKey(1)

    vc.release()
    
    # Get list of image names
    im_names = []
    for img in os.listdir(im_path):
        if img.endswith('.jpg'):
            im_names.append(os.path.splitext(img)[0].strip())

    im_names.sort()

    '''
    for idx in range(len(im_names)):
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_names[idx])
    '''
    # Convert images to video

    '''
    eg = cv2.imread(im_path + '/0000001.jpg')
    height, width, _ = eg.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(os.path.join(cfg.DATA_DIR, 'demo_video', 'output.avi'), fourcc, 24.0, (width, height))
    
    for img in im_names:
        frame = cv2.imread(im_path + '/' + img + '.jpg')
        video.write(frame)
    video.release()
    '''

    cv2.destroyAllWindows()

    #clip = ImageSequenceClip(os.path.join(cfg.DATA_DIR, 'demo_video', 'video_to_images'), fps=24)
    #clip.to_videofile(os.path.join(cfg.DATA_DIR, 'demo_video', 'output.mp4'), fps=24)
