#! /usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg
from utils.utils_track import *
import math

# NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
# STRIDES         = np.array(cfg.YOLO.STRIDES)
# IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
# XYSCALE = cfg.YOLO.XYSCALE
# ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

def YOLO(input_layer, NUM_CLASS, model='yolov4', is_tiny=False):
    if is_tiny:
        if model == 'yolov4':
            return YOLOv4_tiny(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3_tiny(input_layer, NUM_CLASS)
    else:
        if model == 'yolov4':
            return YOLOv4(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3(input_layer, NUM_CLASS)

def YOLOv3(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.cspdarknet53(input_layer)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4_tiny(input_layer, NUM_CLASS):
    route_1, conv = backbone.cspdarknet53_tiny(input_layer)
#route_1:26*26*256
#conv:13*13*512
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, NUM_CLASS * (NUM_CLASS + 8)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, NUM_CLASS * (NUM_CLASS + 8)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def YOLOv3_tiny(input_layer, NUM_CLASS):
    route_1, conv = backbone.darknet53_tiny(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def decode(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE=[1,1,1], FRAMEWORK='tf'):
    if FRAMEWORK == 'trt':
        return decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    elif FRAMEWORK == 'tflite':
        return decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    else:
        return decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)

def decode_train(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    conv_output = tf.reshape(conv_output,
                             (tf.shape(conv_output)[0], output_size, output_size, 2, 8 + NUM_CLASS))

    #conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
    #                                                                      axis=-1)
    conv_raw_dxdydz, conv_raw_dhdwdl, conv_raw_conf, conv_raw_r, conv_raw_prob = tf.split(conv_output, (3, 3, 1, 1, NUM_CLASS),
                                                                          axis=-1)
    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    z_grid = tf.expand_dims(tf.cast(tf.ones_like(xy_grid[0]),tf.float32)*(-1.7),axis=-1)
    xy_grid = tf.stack(xy_grid, axis=-1)
    xy_grid = tf.cast(xy_grid, tf.float32)
    xyz_grid = tf.concat([xy_grid,z_grid],axis=-1)
    xyz_grid = tf.expand_dims(xyz_grid,axis=2)
    xyz_grid = tf.tile(tf.expand_dims(xyz_grid, axis=0), [batch_size, 1, 1, 2, 1])


    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output,
                             (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    conv_raw_dxdy_0, conv_raw_dwdh_0, conv_raw_score_0,\
    conv_raw_dxdy_1, conv_raw_dwdh_1, conv_raw_score_1,\
    conv_raw_dxdy_2, conv_raw_dwdh_2, conv_raw_score_2 = tf.split(conv_output, (2, 2, 1+NUM_CLASS, 2, 2, 1+NUM_CLASS,
                                                                                2, 2, 1+NUM_CLASS), axis=-1)

    conv_raw_score = [conv_raw_score_0, conv_raw_score_1, conv_raw_score_2]
    for idx, score in enumerate(conv_raw_score):
        score = tf.sigmoid(score)
        score = score[:, :, :, 0:1] * score[:, :, :, 1:]
        conv_raw_score[idx] = tf.reshape(score, (1, -1, NUM_CLASS))
    pred_prob = tf.concat(conv_raw_score, axis=1)

    conv_raw_dwdh = [conv_raw_dwdh_0, conv_raw_dwdh_1, conv_raw_dwdh_2]
    for idx, dwdh in enumerate(conv_raw_dwdh):
        dwdh = tf.exp(dwdh) * ANCHORS[i][idx]
        conv_raw_dwdh[idx] = tf.reshape(dwdh, (1, -1, 2))
    pred_wh = tf.concat(conv_raw_dwdh, axis=1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.stack(xy_grid, axis=-1)  # [gx, gy, 2]
    xy_grid = tf.expand_dims(xy_grid, axis=0)
    xy_grid = tf.cast(xy_grid, tf.float32)

    conv_raw_dxdy = [conv_raw_dxdy_0, conv_raw_dxdy_1, conv_raw_dxdy_2]
    for idx, dxdy in enumerate(conv_raw_dxdy):
        dxdy = ((tf.sigmoid(dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
        conv_raw_dxdy[idx] = tf.reshape(dxdy, (1, -1, 2))
    pred_xy = tf.concat(conv_raw_dxdy, axis=1)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    # x = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.float32), axis=0), [output_size, 1])
    # y = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.float32), axis=1), [1, output_size])
    # xy_grid = tf.expand_dims(tf.stack([x, y], axis=-1), axis=2)  # [gx, gy, 1, 2]
    # xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    # pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
    #           STRIDES[i]
    pred_xy = (tf.reshape(tf.sigmoid(conv_raw_dxdy), (-1, 2)) * XYSCALE[i] - 0.5 * (XYSCALE[i] - 1) + tf.reshape(xy_grid, (-1, 2))) * STRIDES[i]
    pred_xy = tf.reshape(pred_xy, (batch_size, output_size, output_size, 3, 2))
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob

    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


# def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    # conv_shape  = tf.shape(conv)
    # batch_size  = conv_shape[0]
    # output_size = conv_shape[1]
    # input_size  = STRIDES[i] * output_size
    # conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # conv_raw_conf = conv[:, :, :, :, 4:5]
    # conv_raw_prob = conv[:, :, :, :, 5:]

    # pred_xywh     = pred[:, :, :, :, 0:4]
    # pred_conf     = pred[:, :, :, :, 4:5]

    # label_xywh    = label[:, :, :, :, 0:4]
    # respond_bbox  = label[:, :, :, :, 4:5]
    # label_prob    = label[:, :, :, :, 5:]

    # giou = tf.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
    # input_size = tf.cast(input_size, tf.float32)

    # bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    # giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    # iou = utils.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    # conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # conf_loss = conf_focal * (
            # respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            # +
            # respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    # )

    # prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    # conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    # prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    # return giou_loss, conf_loss, prob_loss

def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1


def compute_loss(conv,pred,label,bboxes,i,IOU_LOSS_THRESH=0.3):
    STRIDES=[2,4]
    XYSCALE=[1.05,1.05]
    ANCHORS = np.array([[1.53,1.63,3.88],[3.53,2.54,16.09],[1.76,0.66,0.84],[1.73,0.6,1.76]])
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    
    conv = tf.reshape(conv, (batch_size, output_size, output_size, cfg.NUM_CLASS, 8 + cfg.NUM_CLASS))
    encode_pred = encode_lable_gzp(pred,i,NUM_CLASS=4)
    pred_conf     = pred[:, :, :, :, 7:8]
    pred_r = pred[:, :, :, :, 6:7]
    
    
    conv_raw_conf = conv[:, :, :, :, 7:8]
    conv_raw_prob = conv[:, :, :, :, 8:]
    
    
    conv_raw_conf = conv[:, :, :, :, 7:8]
    conv_raw_prob = conv[:, :, :, :, 8:]
    pred_xyzhwlr     = pred[:, :, :, :, 0:7]
    pred_conf     = pred[:, :, :, :, 7:8]
    pred_r = pred[:, :, :, :, 6:7]

    label_xyzhwlr    = label[:, :, :, :, 0:7]
    respond_bbox  = label[:, :, :, :, 7:8]
    label_prob    = label[:, :, :, :, 8:]
    label_r = label[:, :, :, :, 6:7]
    giou_3d = tf.expand_dims(bbox_iou_from_pre(pred_xyzhwlr, label_xyzhwlr,mode='giou3d'), axis=-1)
    input_size = tf.cast(input_size, tf.float32)
    bbox_loss_scale = 2.0 - 1.0 * label_xyzhwlr[:, :, :, :, 4:5] * label_xyzhwlr[:, :, :, :, 5:6] / (1.26*3.01)  #w与l的统计均值
    #盒体面积越大，权重越小，有助于小物体检测
    giou_loss = respond_bbox *  (1- giou_3d) #bbox_loss_scale *
    delt_r = pred_r-label_r
    r_loss = respond_bbox * tf.pow((tf.math.sin(delt_r)-tf.math.cos(delt_r)),2)/2
    iou = bbox_iou_from_pre(pred_xyzhwlr[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    r_loss = tf.reduce_mean(tf.reduce_sum(r_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
    return giou_loss, r_loss, conf_loss, prob_loss

#模型输出到预测
def decode_train_gzp(conv_output,NUM_CLASS,i):
    STRIDES=np.array([2,4],dtype=np.int32)
    XYSCALE=np.array([1.00,1.00],dtype=np.float32)
    ANCHORS = np.array([[1.31,1.59,3.33],[2.54,2.28,6.81],[1.19,0.51,0.45],[1.22,0.64,1.46]],dtype=np.float32)#hwl

    output_size = tf.shape(conv_output)[1]
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, NUM_CLASS , 8 + NUM_CLASS))
    conv_raw_dxdydz, conv_raw_dhdwdl, conv_raw_r, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (3, 3, 1, 1, NUM_CLASS),
                                                                              axis=-1)
    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    z_grid = tf.expand_dims(tf.cast(tf.ones_like(xy_grid[0]),tf.float32)*(-0.84),axis=-1)
    xy_grid = tf.stack(xy_grid, axis=-1)
    xy_grid = tf.cast(xy_grid, tf.float32)
    xyz_grid = tf.concat([xy_grid,z_grid],axis=-1)
    xyz_grid = tf.expand_dims(xyz_grid,axis=2)
    xyz_grid = tf.tile(tf.expand_dims(xyz_grid, axis=0), [batch_size, 1, 1, NUM_CLASS, 1])

    pred_xyz = ((tf.sigmoid(conv_raw_dxdydz) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xyz_grid) * STRIDES[i]
    pred_hwl = tf.exp(conv_raw_dhdwdl) * ANCHORS
    pred_r = (tf.sigmoid(conv_raw_r)-0.5)*math.pi*2

    pred_xyzhwlr = tf.concat([pred_xyz, pred_hwl, pred_r], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_result = tf.concat([pred_xyzhwlr, pred_conf, pred_prob], axis=-1)
    return pred_result 

def bbox_iou_from_pre(pre,label,mode='iou'):
    pre_xmin = tf.minimum(pre[..., 0:1] - pre[..., 5:6] * 0.5*tf.math.cos(pre[..., 6:7])-pre[..., 4:5] * 0.5*tf.math.sin(pre[..., 6:7]),
              pre[..., 0:1] + pre[..., 5:6] * 0.5*tf.math.cos(pre[..., 6:7])+pre[..., 4:5] * 0.5*tf.math.sin(pre[..., 6:7]))
    pre_ymin = tf.minimum( pre[..., 1:2] - pre[..., 5:6] * 0.5*tf.math.sin(pre[..., 6:7])-pre[..., 4:5] * 0.5*tf.math.cos(pre[..., 6:7]),
              pre[..., 1:2] + pre[..., 5:6] * 0.5*tf.math.sin(pre[..., 6:7])+pre[..., 4:5] * 0.5*tf.math.cos(pre[..., 6:7]))
    pre_zmin = pre[..., 2:3]
    
    pre_xmax = tf.maximum(pre[..., 0:1] - pre[..., 5:6] * 0.5*tf.math.cos(pre[..., 6:7])-pre[..., 4:5] * 0.5*tf.math.sin(pre[..., 6:7]),
              pre[..., 0:1] + pre[..., 5:6] * 0.5*tf.math.cos(pre[..., 6:7])+pre[..., 4:5] * 0.5*tf.math.sin(pre[..., 6:7]))
    pre_ymax = tf.maximum( pre[..., 1:2] - pre[..., 5:6] * 0.5*tf.math.sin(pre[..., 6:7])-pre[..., 4:5] * 0.5*tf.math.cos(pre[..., 6:7]),
              pre[..., 1:2] + pre[..., 5:6] * 0.5*tf.math.sin(pre[..., 6:7])+pre[..., 4:5] * 0.5*tf.math.cos(pre[..., 6:7]))
    pre_zmax = pre[..., 2:3] + pre[..., 3:4]

    label_xmin = tf.minimum(label[..., 0:1] - label[..., 5:6] * 0.5*tf.math.cos(label[..., 6:7])-label[..., 4:5] * 0.5*tf.math.sin(label[..., 6:7]),
              label[..., 0:1] + label[..., 5:6] * 0.5*tf.math.cos(label[..., 6:7])+label[..., 4:5] * 0.5*tf.math.sin(label[..., 6:7]))
    label_ymin = tf.minimum(label[..., 1:2] - label[..., 5:6] * 0.5*tf.math.sin(label[..., 6:7])-label[..., 4:5] * 0.5*tf.math.cos(label[..., 6:7]),
              label[..., 1:2] + label[..., 5:6] * 0.5*tf.math.sin(label[..., 6:7])+label[..., 4:5] * 0.5*tf.math.cos(label[..., 6:7]))
    label_zmin = label[..., 2:3]
    label_xmax = tf.maximum(label[..., 0:1] - label[..., 5:6] * 0.5*tf.math.cos(label[..., 6:7])-label[..., 4:5] * 0.5*tf.math.sin(label[..., 6:7]),
              label[..., 0:1] + label[..., 5:6] * 0.5*tf.math.cos(label[..., 6:7])+label[..., 4:5] * 0.5*tf.math.sin(label[..., 6:7]))
    label_ymax = tf.maximum(label[..., 1:2] - label[..., 5:6] * 0.5*tf.math.sin(label[..., 6:7])-label[..., 4:5] * 0.5*tf.math.cos(label[..., 6:7]),
              label[..., 1:2] + label[..., 5:6] * 0.5*tf.math.sin(label[..., 6:7])+label[..., 4:5] * 0.5*tf.math.cos(label[..., 6:7]))
    label_zmax = label[..., 2:3] + label[..., 3:4]
    
    bboxes1_area = pre[..., 4] * pre[..., 5]
    bboxes1_volume = pre[..., 4] * pre[..., 5] * pre[..., 3]
    bboxes2_area = label[..., 4] * label[..., 5]
    bboxes2_volume = label[..., 4] * label[..., 5] * label[..., 3]

    pre_min_xyz = tf.concat([pre_xmin,pre_ymin,pre_zmin],axis=-1)
    pre_max_xyz = tf.concat([pre_xmax,pre_ymax,pre_zmax],axis=-1)
    label_min_xyz = tf.concat([label_xmin,label_ymin,label_zmin],axis=-1)
    label_max_xyz = tf.concat([label_xmax,label_ymax,label_zmax],axis=-1)

    left_up = tf.maximum(pre_min_xyz, label_min_xyz)
    right_down = tf.minimum(pre_max_xyz, label_max_xyz)
    inter_section = tf.maximum(right_down - left_up, 0.0)
    
    
    if mode=='iou':
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = bboxes1_area + bboxes2_area - inter_area
        iou = tf.math.divide_no_nan(inter_area, union_area)
        return iou    
    if mode=='giou':
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = bboxes1_area + bboxes2_area - inter_area
        iou = tf.math.divide_no_nan(inter_area, union_area)
        enclose_left_up = tf.minimum(pre_min_xyz, label_min_xyz)
        enclose_right_down = tf.maximum(pre_max_xyz, label_max_xyz)
        enclose_section = enclose_right_down - enclose_left_up
        enclose_area = enclose_section[..., 0] * enclose_section[..., 1]
        giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)
        return giou
    if mode =='giou3d':
        inter_volume = inter_section[..., 0] * inter_section[..., 1] * inter_section[..., 2]
        union_volume = bboxes1_volume + bboxes2_volume - inter_volume        
        iou = tf.math.divide_no_nan(inter_volume, union_volume)
        enclose_left_up = tf.minimum(pre_min_xyz, label_min_xyz)
        enclose_right_down = tf.maximum(pre_max_xyz, label_max_xyz)
        enclose_section = enclose_right_down - enclose_left_up
        enclose_volume = enclose_section[..., 0] * enclose_section[..., 1] * enclose_section[..., 2]
        giou_3d = iou - tf.math.divide_no_nan(enclose_volume - union_volume, enclose_volume)
        return giou_3d
    if mode =='iou3d':
        inter_volume = inter_section[..., 0] * inter_section[..., 1] * inter_section[..., 2]
        union_volume = bboxes1_volume + bboxes2_volume - inter_volume        
        iou = tf.math.divide_no_nan(inter_volume, union_volume)
        return iou        
    return iou

def encode_lable_gzp(pred_result,i,NUM_CLASS=4):
    STRIDES=np.array([2,4],dtype=np.int32)
    XYSCALE=np.array([1.00,1.00],dtype=np.float32)
    ANCHORS = np.array([[1.31,1.59,3.33],[2.54,2.28,6.81],[1.19,0.51,0.45],[1.22,0.64,1.46]],dtype=np.float32)#hwl

    pred_xyz = pred_result[:,:,:,:,0:3]
    pred_hwl = pred_result[:,:,:,:,3:6]
    pred_r = pred_result[:,:,:,:,6:7]
    pred_conf = pred_result[:,:,:,:,7:8]
    pred_prob = pred_result[:,:,:,:,8:]

    conv_r_sigmoid = pred_r/(2*math.pi)+0.5
    conv_hwl = tf.log(pred_hwl/ANCHORS)

    batch_size = tf.shape(pred_result)[0]
    output_size = tf.shape(pred_result)[1]
    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    z_grid = tf.expand_dims(tf.cast(tf.ones_like(xy_grid[0]),tf.float32)*(-0.84),axis=-1)
    xy_grid = tf.stack(xy_grid, axis=-1)
    xy_grid = tf.cast(xy_grid, tf.float32)
    xyz_grid = tf.concat([xy_grid,z_grid],axis=-1)
    xyz_grid = tf.expand_dims(xyz_grid,axis=2)
    xyz_grid = tf.tile(tf.expand_dims(xyz_grid, axis=0), [batch_size, 1, 1, NUM_CLASS, 1])
    conv_xyz_sigmoid = (pred_xyz/STRIDES[i]-xyz_grid+0.5* (XYSCALE[i] - 1))/ XYSCALE[i]
    conv=tf.concat([conv_xyz_sigmoid,conv_hwl,conv_r_sigmoid,pred_conf,pred_prob],axis=-1)
    return conv 