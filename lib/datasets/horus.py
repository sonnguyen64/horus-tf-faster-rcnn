# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from model.config import cfg
from .horus_eval import horus_eval


class horus(imdb):
  def __init__(self, image_set, use_diff=False):
    image_set = 'horus_' + image_set 
    if use_diff:
        image_set += '_diff'
    imdb.__init__(self, image_set)
    self._image_set = image_set
    # self._data_path = self._get_default_path()
    self._data_path = self._get_data_path(self._image_set)
    # self._classes = ('__background__', 'corrosion', 'cable_dangle', 'paint_peel', 'rf_', 'rru_', 'mw_', 'aol', 'platform_mesh', 'antenna_boom', 'lightning_rod')
    self._classes = cfg.CLASSES
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = ['.JPG', '.jpg']
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'
    
    self._dets = {}
    for cls in self._classes:
        self._dets[cls] = []

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': use_diff}

    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_direc = os.path.join(self._data_path, 'images')
    for ext in self._image_ext:
        image_path = os.path.join(image_direc, index + ext)
        if os.path.exists(image_path):
            return image_path

  def _load_image_set_index(self):
    dirPath = os.path.join(self._data_path, 'labels')
    image_index = []
    for file_name in os.listdir(dirPath):
      if file_name.endswith('.xml'):
        image_index.append(os.path.splitext(file_name)[0].strip())
    return image_index

  def _get_default_path(self):
    """
    Return the default path where Horus data is expected to be placed.
    """
    return os.path.join(cfg.DATA_DIR, 'horus')

  def _get_data_path(self, dataset):
    """
    Return the default path where Horus data is expected to be placed.
    """
    if dataset == 'horus_trainval':
      return os.path.join(cfg.DATA_DIR, 'horus-train')
    elif dataset == 'horus_test':
      return os.path.join(cfg.DATA_DIR, 'horus-test')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self._image_set + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self._image_set, cache_file))
      return roidb

    gt_roidb = [self._load_horus_labels(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if self._image_set != 'horus_test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_horus_labels(self, index):
    """
    Load image and bounding boxes info from XML file in the Horus data
    format.
    """
    filename = os.path.join(self._data_path, 'labels', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    if not self.config['use_diff']:
      # Exclude the samples labeled as difficult
      non_diff_objs = [
        obj for obj in objs if int(obj.find('difficult').text) == 0]
      objs = non_diff_objs

    # Count numer of objects
    num_objs = 0
    for obj in objs:
      if obj.find('name').text.lower().strip() in self._classes:
        num_objs += 1

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for horus is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    ix = 0
    for obj in objs:
      if obj.find('name').text.lower().strip() in self._classes:
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        cls = self._class_to_ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
        ix += 1

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _do_python_eval(self, output_dir='output', iou_thresh=0.5):
    cachedir = os.path.join(self._data_path, 'annotations_cache')
    # cachedir = os.path.join(cfg.DATA_DIR, 'horus-test', 'annotations_cache')
    datapath = os.path.join(self._data_path)
    # annopath = os.path.join(cfg.DATA_DIR, 'horus-test', 'labels')
    tps = 0
    fps = 0
    gts = 0

    aps = []

    for i, cls in enumerate(self._classes):
      if cls != '__background__':
        tp, fp, npos, rec, prec, ap = horus_eval(self._dets, datapath, cls, cachedir, ovthresh=iou_thresh, use_diff=self.config['use_diff'])
        if ap == ap:
          aps += [ap]
        # print('TP = ', tp)
        # print('FP = ', fp)
        # print('NP = ', npos)
        print('==========')
        print('Class: ', cls)
        # print('FP: ', fp_eval)
        print('TP = {:.1f}\tFP = {:.1f}\tNP = {:.1f}'.format(tp, fp, npos))
        tps += tp
        fps += fp
        gts += npos
        print('AP: {:.4f}'.format(ap))
        if tp > 0:
            print('Precision: {:.4f}\tRecall: {:.4f}'.format(tp / (tp + fp), tp / npos))
        print('==========')
    # print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    # for ap in aps:
    #  print(('{:.3f}'.format(ap)))
    print(('mAP: {:.3f}'.format(np.mean(aps))))
    print('TP = {:.1f}\tFP = {:.1f}\tNP = {:.1f}'.format(tps, fps, gts))
    print(('Precision: {:.4f}\tRecall: {:.4f}'.format(tps / (tps + fps), tps / gts))) 
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')



  def _write_voc_results_file(self, all_boxes):

    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      # print('Writing {} VOC results file'.format(cls))
      # filename = self._get_voc_results_file_template().format(cls)
      # with open(filename, 'wt') as f:
      for im_ind, index in enumerate(self.image_index):
        dets = all_boxes[cls_ind][im_ind]
        if dets == []:
          continue
        for k in range(dets.shape[0]):
          self._dets[cls].append([index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1])
          # print('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

  def evaluate_detections(self, all_boxes, output_dir, iou_thresh=0.5):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir, iou_thresh=iou_thresh)


if __name__ == '__main__':
  from datasets.horus import horus

  d = horus('trainval')
  res = d.roidb
  from IPython import embed;

  embed()
