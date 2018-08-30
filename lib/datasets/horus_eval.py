# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np

def parse_rec(filename):
  """ Parse Horus xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects


def horus_ap(rec, prec):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  # correct AP calculation
  # first append sentinel values at the end
  mrec = np.concatenate(([0.], rec, [1.]))
  mpre = np.concatenate(([0.], prec, [0.]))

  # compute the precision envelope
  for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

  # to calculate area under PR curve, look for points
  # where X axis (recall) changes value
  i = np.where(mrec[1:] != mrec[:-1])[0]

  # and sum (\Delta recall) * prec
  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def horus_eval(dets,
             datapath,
             classname,
             cachedir,
             ovthresh=0.5,
             use_diff=False):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, 'horus_annots.pkl')
  # read list of images
  imagenames = []
  for file_name in os.listdir(datapath + '/labels'):
    if file_name.endswith('.xml'):
      imagenames.append(os.path.splitext(file_name)[0].strip()) 

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(os.path.join(datapath, 'labels', imagename + '.xml'))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'w') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')
  
  # extract gt objects for this class
  class_recs = {}
  npos = 0

  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if str.lower(obj['name']) == classname]
    bbox = np.array([x['bbox'] for x in R])
    if use_diff:
      difficult = np.array([False for x in R]).astype(np.bool)
    else:
      difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}
  # read dets
  #detfile = detpath.format(classname)
  #with open(detfile, 'r') as f:
  #  lines = f.readlines()

  #splitlines = [x.strip().split(' ') for x in lines]
  #image_ids = [x[0] for x in splitlines]
  #confidence = np.array([float(x[1]) for x in splitlines])
  #BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  detection = dets[classname]
  image_ids = []
  confidence = []
  BB = []

  # fp_eval = []

  for x in detection:
    image_ids.append(x[0])
    confidence.append(float(x[1]))
    BB.append([float(i) for i in x[2:]])

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  confidence = np.array(confidence)
  BB = np.array(BB)
  
  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

  # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
            # fp_eval.append(image_ids[d])
            #if classname == 'antenna_boom':
            #  crop_img(datapath + '/images', image_ids[d] + '.JPG', classname + '_' + image_ids[d] + '_'+ str(d) + '.jpg', bb[0], bb[2], bb[1], bb[3])
      else:
        fp[d] = 1.
        # fp_eval.append(image_ids[d])
        #if classname == 'antenna_boom':
        #  crop_img(datapath + '/images', image_ids[d] + '.JPG', classname + '_' + image_ids[d] + '_' + str(d) + '.jpg', bb[0], bb[2], bb[1], bb[3])
      
  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)

  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = horus_ap(rec, prec)

  tp_count = 0
  fp_count = 0

  if len(tp) > 0:
    tp_count = np.max(tp)
  
  if len(fp) > 0:
    fp_count = np.max(fp)

  return tp_count, fp_count, npos, rec, prec, ap

import cv2

def crop_img(im_path, im_name, crop_im_name, xmin, xmax, ymin, ymax):
  img = cv2.imread(im_path + '/' + im_name)
  crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
  cv2.imwrite(im_path + '/fp/' + crop_im_name, crop_img)


