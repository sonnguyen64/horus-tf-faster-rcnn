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
import os, cv2
import argparse
from utils.visualization import draw_bounding_boxes

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1


NETS = {'vgg16': ('vgg16_faster_rcnn_iter_32000.ckpt',),'res101': ('res101_faster_rcnn_iter_4000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'horus': ('horus_trainval',)}
NUM_CLASSES = len(cfg.CLASSES)

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    for i in inds:
        bbox = dets[i, :4]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
        cv2.putText(im, class_name, (bbox[0], bbox[1]-2), 0, 0.3, (0,255,0))

def demo(sess, net, image_name):
    # Load the demo image
    im_path = os.path.join(cfg.DATA_DIR, 'horus-test', 'images')
    im_file = os.path.join(im_path, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.6
    for cls_ind, cls in enumerate(cfg.CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        inds = np.where(scores[:, cls_ind] >= CONF_THRESH)[0]
        cls_boxes = boxes[inds, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[inds, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, cfg.TEST.NMS)
        dets = dets[keep, :]
        print(dets)
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)
        # inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
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
    anchor_scales = cfg.ANCHOR_SCALES
    anchor_ratios = cfg.ANCHOR_RATIOS

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
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1,num_layers=101)
    else:
        raise NotImplementedError

    net.create_architecture(sess, "TEST", NUM_CLASSES,
            tag='default', anchor_scales=anchor_scales, anchor_ratios=anchor_ratios, image=None, im_info=None)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    im_names = []
    for file_name in os.listdir(os.path.join(cfg.DATA_DIR, 'horus-test', 'images')):
      if file_name.endswith('.JPG'):
        im_names.append(os.path.splitext(file_name)[0].strip() + '.JPG')

    # im_names = ['DJI_0737.JPG', 'DJI_0732.JPG', 'DJI_0609.JPG', 'DJI_0039.JPG', 'DJI_0031.JPG']

    im_names = ['DJI_0634.JPG']
    # im_names = ['DJI_0034.JPG', 'DJI_0061.JPG', 'DJI_0082.JPG', 'DJI_0090.JPG', 'DJI_0623.JPG', 'DJI_0027.JPG', 'DJI_0037.JPG', 'DJI_0043.JPG', 'DJI_0043_XT.JPG', 'DJI_0074_T.JPG', 'DJI_0027_T.JPG', 'DJI_0037_T.JPG', 'DJI_0043_T.JPG', 'DJI_0074.JPG']
    
    
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

