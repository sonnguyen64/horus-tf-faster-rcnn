# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

STANDARD_COLORS = [
    (255,255,255), (0,255,0), (0,0,255), (255,0,0), (255,255,0),
    (0,255,255), (255,0,255), (0,0,128), (0,128,0), (128,0,0),
    (30,144,255),(138,43,226), (128,0,128), (255,20,147),
    (210,105,30), (0,0,0)
]

NUM_COLORS = len(STANDARD_COLORS)

def draw_bounding_boxes(im, dets, cls_ind, cls):
  for det in dets:
    bbox = det[:4]
    h, w, _ = im.shape
    color = STANDARD_COLORS[cls_ind % NUM_COLORS]
    thick = int((h + w) // 350)
    cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick)
    cv2.putText(im, cls, (int(bbox[0]), int(bbox[1]-12)), 0, 1e-3 * h, color, thick//3)
  return im
