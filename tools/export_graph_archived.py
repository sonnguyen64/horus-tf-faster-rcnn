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
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from layer_utils.proposal_layer_tf import bbox_transform_inv_tf, clip_boxes_tf


CLASSES = ('__background__', 'corrosion', 'cable_dangle', 'paint_peel', 'rf_', 'rru_', 'mw_', 'aol', 'platform_mesh', 'antenna_boom', 'lightning_rod')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_32000.ckpt',),
        'res101': ('res101_faster_rcnn_iter_110000.ckpt',
                   'res101_faster_rcnn_iter_1190000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',),
            'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),
            'horus': ('horus_trainval',)}

NUM_CLASSES = len(CLASSES)

CONF_THRESH = 0.6
NMS_THRESH = 0.3


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def postprocess(rois, bbox_pred, scores, im_info):
    # Post processing
    boxes = rois[:, 1:5] / im_info[0, 2]

    # Do NMS and visualize results
    all_cls, all_scores, all_bboxs = [], [], []
    for cls_ind, cls in enumerate(CLASSES):
        if cls_ind == 0:
            continue  # because we skipped background

        # Select interesting boxes
        cls_scores = scores[:, cls_ind]
        inds = tf.where(cls_scores >= CONF_THRESH)[:, 0]
        cls_scores = tf.gather(cls_scores, inds)

        cls_deltas = tf.gather(bbox_pred[:, 4 * cls_ind:4 * (cls_ind + 1)],
                               inds)
        cls_boxes = bbox_transform_inv_tf(tf.gather(boxes, inds), cls_deltas)
        cls_boxes = clip_boxes_tf(cls_boxes, im_info[0, :2]/im_info[0, 2])

        keep = tf.image.non_max_suppression(cls_boxes,
                                            cls_scores,
                                            500,
                                            iou_threshold=NMS_THRESH)
        all_cls.append(tf.ones_like(keep) * cls_ind)
        all_scores.append(tf.gather(cls_scores, keep))
        all_bboxs.append(tf.gather(cls_boxes, keep))
    return tf.concat(all_cls, 0), tf.concat(all_scores, 0), tf.concat(
        all_bboxs, 0)


def preprocess(im):
    # Cast and remove mean
    im_orig = tf.cast(im, dtype=tf.float32)
    im_orig -= cfg.PIXEL_MEANS

    # Get shape and size
    im_shape = tf.cast(tf.shape(im_orig), dtype=tf.float32)
    im_size_min = tf.reduce_min(im_shape[:2])
    im_size_max = tf.reduce_max(im_shape[:2])

    # Resize image
    im_scale = cfg.TEST.SCALES[0] / im_size_min
    # Prevent the biggest axis from being more than MAX_SIZE
    im_scale = tf.cond(tf.round(im_scale * im_size_max) > float(cfg.TEST.MAX_SIZE),
                       lambda: cfg.TEST.MAX_SIZE / im_size_max,
                       lambda: im_scale)
    im_resized = tf.image.resize_images(im_orig,
                                        tf.cast(tf.round(im_shape[:2] * im_scale), dtype=tf.int32),
                                        method=tf.image.ResizeMethod.BILINEAR,
                                        align_corners=True)
    im_blob = im_resized[tf.newaxis]
    im_info = tf.concat([tf.cast(tf.shape(im_resized)[:2], dtype=tf.float32),
                         [im_scale]], 0)[np.newaxis]
    return im_blob, im_info


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712 coco]',
                        choices=DATASETS.keys(), default='coco')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    demonet = args.demo_net
    dataset = args.dataset

    # model path
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                           NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly? If you want something '
                       'simple and handy, try ./tools/demo_depre.py first.').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    # raw_image = tf.placeholder(tf.uint8, shape=[None, None, 3])
    # image, im_info = preprocess(raw_image)

    
    im_blob = tf.placeholder(tf.float32, shape=[None, None, 3])[tf.newaxis]
    # im_blob, im_info = preprocess(im_blob)
    im_info = tf.placeholder(tf.float32, shape=[3])[np.newaxis]

    net.create_architecture(sess, "TEST", NUM_CLASSES, tag='default', anchor_scales=[8, 16, 32, 64], anchor_ratios=[0.25, 0.5, 1, 2, 4], image=im_blob, im_info=im_info)

    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']
    # for im_name in im_names:
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print('Demo for data/demo/{}'.format(im_name))
    #     demo(sess, net, im_name)
    #
    # plt.show()

    # export the model to make it loadable with TF serving
    export_path = 'frozen_model_{}_{}'.format(demonet, dataset)
    print('Exporting trained model to', export_path)
    builder = saved_model_builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    # tensor_info_img = utils.build_tensor_info(raw_image)

    tensor_info_im_blob = utils.build_tensor_info(im_blob)
    tensor_info_im_info = utils.build_tensor_info(im_info)

    tensor_info_cls = utils.build_tensor_info(net._predictions['cls_prob'])
    tensor_info_bbox = utils.build_tensor_info(net._predictions['bbox_pred'])
    tensor_info_rois = utils.build_tensor_info(net._predictions['rois'])
    
    '''
    prediction_signature = signature_def_utils.build_signature_def(
        inputs={'image': tensor_info_img},
        outputs={'cls_prob': tensor_info_cls,
                 'bbox_pred': tensor_info_bbox,
                 'rois': tensor_info_rois,
                 },
        method_name=signature_constants.PREDICT_METHOD_NAME)
    '''
    
    prediction_signature = signature_def_utils.build_signature_def(
        inputs={'im_blob': tensor_info_im_blob,
                'im_info': tensor_info_im_info},
        outputs={'cls_prob': tensor_info_cls,
                 'bbox_pred': tensor_info_bbox,
                 'rois': tensor_info_rois,
                 },
        method_name=signature_constants.PREDICT_METHOD_NAME)

    # Do it with post processing
    fin_cls, fin_scores, fin_bbox = postprocess(net._predictions['rois'],
                                                net._predictions['bbox_pred'],
                                                net._predictions['cls_prob'],
                                                net._im_info)

    tensor_info_fin_cls = utils.build_tensor_info(fin_cls)
    tensor_info_fin_score = utils.build_tensor_info(fin_scores)
    tensor_info_fin_bbox = utils.build_tensor_info(fin_bbox)

    '''
    prediction_post_signature = signature_def_utils.build_signature_def(
        inputs={'image': tensor_info_img},
        outputs={'fin_cls': tensor_info_fin_cls,
                 'fin_score': tensor_info_fin_score,
                 'fin_bbox': tensor_info_fin_bbox,
                 },
        method_name=signature_constants.PREDICT_METHOD_NAME)
    '''

    prediction_post_signature = signature_def_utils.build_signature_def(
        inputs={'im_blob': tensor_info_im_blob,
                'im_info': tensor_info_im_info},
        outputs={'fin_cls': tensor_info_fin_cls,
                 'fin_score': tensor_info_fin_score,
                 'fin_bbox': tensor_info_fin_bbox,
                 },
        method_name=signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.initialize_all_tables(),
                              name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            'predict_bbox':
                prediction_signature,
            'predict_post':
                prediction_post_signature
        },
        legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!')
