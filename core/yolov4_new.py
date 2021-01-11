import tensorflow as tf
import numpy as np
import math
import core.common as common
from utils.utils_track import *
from core.config import cfg
from utils.colorize import colorize
anchors = cal_anchors()

def cspdarknet53(input_data):
    input_data = common.convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = common.convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 64, 128), activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = common.residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 128, 256), activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = common.residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    # input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            # , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    # input_data = common.convolutional(input_data, (1, 1, 2048, 512))
    # input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    # input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data

def YOLOv4(input_layer):
    NUM_CLASS = cfg.NUM_CLASS
    route_1, route_2, conv = cspdarknet53(input_layer) #1,1/2,1/4

    route = conv  # route = 1/4
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv) # conv = 1/2
    route_2 = common.convolutional(route_2, (1, 1, 512, 256)) 
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256)) 

    route_2 = conv # route_2 = 1/2
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv) #conv = 1
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    
    route_1 = conv # 1    
    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1) 

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256)) #1/2
    conv = common.upsample(conv) #1

    route_2 = conv #1

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)#1/2
    route = common.upsample(route)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.upsample(conv)
    
    conv = tf.concat([route_1,route_2,conv], axis=-1)

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 16 + cfg.NUM_CLASS ), activate=False, bn=False)
    return conv_lbbox

def cspdarknet53_tiny(input_data):
    input_data = common.convolutional(input_data, (3, 3, 3, 32))#, downsample=True
    input_data = common.convolutional(input_data, (3, 3, 32, 64))#, downsample=True
    input_data = common.convolutional(input_data, (3, 3, 64, 64))

    route = input_data
    input_data = common.route_group(input_data, 2, 1)#split，只取一半
    input_data = common.convolutional(input_data, (3, 3, 32, 32))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 32, 32))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 32, 64))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 64, 128))
    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 64, 128))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 64, 128))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 64, 256))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 512, 512))#512
    return route_1, input_data #1/2, 1/4

def YOLOv4_tiny(input_layer):
    route_1, conv = cspdarknet53_tiny(input_layer)
    #route_1:26*26*128
    #conv:13*13*512
    conv = common.convolutional(conv, (1, 1, 512, 512))
    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lobj_branch = common.upsample(conv_lobj_branch)   
    conv = common.convolutional(conv, (1, 1, 256, 256))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)
    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 512))
    conv_branch=tf.concat([conv_lobj_branch, conv_mobj_branch], axis=-1)    
    conv_bbox = common.convolutional(conv_branch, (1, 1, 256, 18 + cfg.NUM_CLASS), activate=False, bn=False)
    #r_map,a_map, p_map, c_map=tf.split(conv_bbox, (12, 2 , 2, cfg.NUM_CLASS), axis=-1)
    #angle = tf.math.tanh(a_map)*math.pi
    #conv_bbox = tf.concat([r_map, angle, p_map, c_map],axis=-1)
    
    return conv_bbox,conv_branch

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

def compute_loss_and_result(conv_bbox,point_cls,box_reg,lable,train=True):
    targets,pos_equal_one,pos_equal_one_sum,pos_equal_one_for_reg,neg_equal_one,neg_equal_one_sum,cls_onehot,point_cls_gt_placeholder,box_gt_placeholder=lable
    r_map, p_map, c_map=tf.split(conv_bbox, (16 , 2, cfg.NUM_CLASS), axis=-1)
    p_pos = tf.sigmoid(p_map)
    cls_pred = tf.nn.softmax(c_map, axis=-1)
    small_addon_for_BCE=1e-10
    #point loss
    point_cls_loss = -point_cls_gt_placeholder*tf.math.log(tf.clip_by_value(point_cls, small_addon_for_BCE,1.0))
    point_cls_loss = tf.reduce_mean(tf.reduce_sum(point_cls_loss,axis=-1))
    box_reg_loss = tf.reduce_sum(smooth_l1(box_reg,box_gt_placeholder,3),axis=-1)
    box_reg_loss = tf.reduce_mean(box_reg_loss)
    #loss
    
    cls_pos_loss = (-pos_equal_one * tf.math.log(p_pos + small_addon_for_BCE)) / pos_equal_one_sum
    cls_neg_loss = (-neg_equal_one * tf.math.log(1 - p_pos + small_addon_for_BCE)) / neg_equal_one_sum
    cls_pred_loss = tf.reduce_max(pos_equal_one,axis=-1)*tf.reduce_mean((-cls_onehot * tf.math.log(cls_pred + small_addon_for_BCE)),axis=-1) / pos_equal_one_sum
    cls_loss = tf.reduce_mean(tf.reduce_sum( (1* cls_pos_loss + 2 * cls_neg_loss),axis=[1,2,3] ))
    cls_pos_loss_rec = tf.reduce_mean(tf.reduce_sum( cls_pos_loss ,axis=[1,2,3]))
    cls_neg_loss_rec = tf.reduce_mean(tf.reduce_sum( cls_neg_loss ,axis=[1,2,3]))
    cls_pred_loss =tf.reduce_mean(tf.reduce_sum( cls_pred_loss ,axis=[1,2,3]))
    reg_loss = smooth_l1(r_map * pos_equal_one_for_reg, targets *
                                      pos_equal_one_for_reg, 3) / pos_equal_one_sum
    reg_loss = tf.reduce_mean(tf.reduce_sum(reg_loss,axis=[1,2,3]))
    delta_output = r_map
    prob_output = p_pos
    cls_output = cls_pred
    loss = tf.reduce_sum(cls_loss + reg_loss + cls_pred_loss) + point_cls_loss# + box_reg_loss
    result=[delta_output,prob_output,cls_output]
    child_loss = [cls_loss,cls_pos_loss_rec,cls_neg_loss_rec,reg_loss,cls_pred_loss]
    train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', loss),
            tf.summary.scalar('train/reg_loss', reg_loss),
            tf.summary.scalar('train/conf_loss', cls_loss),
            tf.summary.scalar('train/cls_loss', cls_pred_loss),
            tf.summary.scalar('train/point_cls_loss', point_cls_loss),
            tf.summary.scalar('train/box_reg_loss', box_reg_loss)
        ])
    valid_summary = tf.summary.merge([
            tf.summary.scalar('valid/loss', loss),
            tf.summary.scalar('valid/reg_loss', reg_loss),
            tf.summary.scalar('valid/conf_loss', cls_loss),
            tf.summary.scalar('valid/cls_loss', cls_pred_loss),
            tf.summary.scalar('valid/point_cls_loss', point_cls_loss),
            tf.summary.scalar('valid/box_reg_loss', box_reg_loss)
        ])
    if train:
        return loss,result,child_loss,train_summary
    else:
        return loss,result,child_loss,valid_summary

def caculate_nms(boxes2d,boxes2d_scores):
    box2d_ind_after_nms = tf.image.non_max_suppression(
                boxes2d, boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)
    return box2d_ind_after_nms

boxes2d_placeholder = tf.placeholder(tf.float32, [None, 4])
boxes2d_scores = tf.placeholder(tf.float32, [None])
box2d_ind_after_nms = caculate_nms(boxes2d_placeholder,boxes2d_scores)
#模型输出到预测

def predict(sess,probs,deltas,cls_pred,gtbox3d_with_ID,gtbox3d_with_ID_notcare,raw_lidar,bev_f,img, rpn_threshold=cfg.RPN_SCORE_THRESH, evaluation=True,vis=True):
    gtbox3d_with_ID_all = np.concatenate([gtbox3d_with_ID,gtbox3d_with_ID_notcare],axis=-2)
    gt_class = gtbox3d_with_ID_all[0,:,1]
    gt_occluded = gtbox3d_with_ID_all[0,:,0]
    gtbox3d = gtbox3d_with_ID_all[0,:,2::]
    gtbox3ddraw = gtbox3d_with_ID[0,:,2::]
    gtbox3ddraw_notcare = gtbox3d_with_ID_notcare[0,:,2::]

    cls_pred_trans=np.transpose(cls_pred[0],[2,0,1])
    cls_max_arg=np.argmax(cls_pred_trans,axis=0)
    anchors_pick=anchors[cls_max_arg,
                         np.tile(np.arange(cfg.FEATURE_WIDTH)[:,np.newaxis],[1,cfg.FEATURE_HEIGHT]),
                         np.tile(np.arange(cfg.FEATURE_HEIGHT)[np.newaxis,:],[cfg.FEATURE_WIDTH,1]),:]

    batch_size = probs.shape[0]
    batch_probs = probs.reshape([batch_size, -1])
    batch_boxes3d = delta_to_boxes3d_for_tan_angle(deltas, anchors_pick)
    #batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
    ind_b , ind = np.where(batch_probs >= rpn_threshold)
    if len(ind) !=0:
        tmp_boxes3d = batch_boxes3d[0, ind, ...]
        #tmp_boxes2d = batch_boxes2d[ind_b, ind, ...]
        tmp_scores = batch_probs[0, ind]
        cls_max_arg = np.tile(cls_max_arg[...,np.newaxis],2)
        tmp_cls=cls_max_arg.reshape(-1)[ind]

        boxes3d_corner = box3Dcorner_from_gtbox(tmp_boxes3d)
        boxes2d_standard = corner_to_standup_box2d(np.array(boxes3d_corner))

        ind = sess.run(box2d_ind_after_nms, {
                    boxes2d_placeholder: boxes2d_standard,
                    boxes2d_scores: tmp_scores})

        tmp_boxes3d = np.array(tmp_boxes3d[ind, ...])
        tmp_scores =  np.array(tmp_scores[ind])
        tmp_cls= np.array(tmp_cls[ind])
        boxes2d_standard =  np.array(boxes2d_standard[ind,...])
    else:
        tmp_boxes3d=np.array([])
        tmp_scores=np.array([])
        tmp_cls=np.array([])
        boxes2d_standard=np.array([])
        
    # if tmp_boxes3d.shape[0] !=0:
        # regress_angle = regression_angle(tmp_boxes3d,bev_f)
        # tmp_boxes3d[:,6] = regress_angle
        # boxes3d_corner = box3Dcorner_from_gtbox(tmp_boxes3d)
        # boxes2d_standard = corner_to_standup_box2d(np.array(boxes3d_corner))
    
    batch_gt_boxes2d = gtbox3d[:, [0, 1, 4, 5, 6]]
    batch_gt_boxes3d_to_corner = box3Dcorner_from_gtbox(gtbox3d)
    gt_boxes2d_standard = corner_to_standup_box2d(np.array(batch_gt_boxes3d_to_corner))
    ret_box3d = tmp_boxes3d
    ret_score = tmp_scores
    ret_cls = tmp_cls
    boxes2d = boxes2d_standard
    #预测结束部分
    #评价指标
    if evaluation:
        gt_box_num = gtbox3d.shape[0]
        pre_box_num = ret_box3d.shape[0]
        reg_TP=[0 for i in range(cfg.NUM_CLASS)]
        reg_TR=[0 for i in range(cfg.NUM_CLASS)]
        reg_TP_strict=[0 for i in range(cfg.NUM_CLASS)]
        reg_TR_strict=[0 for i in range(cfg.NUM_CLASS)]
        cls_TP=[0 for i in range(cfg.NUM_CLASS)]
        cls_TR=[0 for i in range(cfg.NUM_CLASS)]
        pre_box_cls_num = [0 for i in range(cfg.NUM_CLASS)]
        gt_box_cls_num = [0 for i in range(cfg.NUM_CLASS)]

        if gt_box_num !=0 and pre_box_num !=0:
            iou = bbox_overlaps(
                np.ascontiguousarray(boxes2d).astype(np.float32),
                np.ascontiguousarray(gt_boxes2d_standard).astype(np.float32),
                )
            iou_argmax_for_pred=np.argmax(iou, axis=1)
            iou_argmax_for_gt=np.argmax(iou, axis=0)
            iou_max_for_pred=np.max(iou,axis=1)
            iou_max_for_gt=np.max(iou,axis=0)
            for i in range(pre_box_num):
                iou_maxarg=iou_argmax_for_pred[i]
                iou_max = iou_max_for_pred[i]
                gt_cls = int(gt_class[iou_maxarg])
                pred_cls = int(tmp_cls[i])
                if gt_cls !=4:
                    pre_box_cls_num[pred_cls] += 1
                    
                    if iou_max>0.5:
                        reg_TP[pred_cls] += 1
                    if iou_max>0.7:
                        reg_TP_strict[pred_cls] += 1
                    if gt_cls == pred_cls:
                        cls_TP[pred_cls] += 1
            for i in range(gt_box_num):
                iou_maxarg=iou_argmax_for_gt[i]
                iou_max = iou_max_for_gt[i]
                pred_cls = int(tmp_cls[iou_maxarg])
                gt_cls = int(gt_class[i])
                if gt_cls !=4:
                    gt_box_cls_num[gt_cls] += 1
                    if iou_max>0.5:
                        reg_TR[gt_cls] += 1
                    if iou_max>0.7:
                        reg_TR_strict[gt_cls] += 1 
                    if gt_cls == pred_cls:
                        cls_TR[gt_cls] += 1
        elif gt_box_num !=0:
            for i in range(gt_box_num):
                gt_cls = int(gt_class[i])
                if gt_cls !=4:
                    gt_box_cls_num[gt_cls] += 1
        elif pre_box_num !=0:
            for i in range(pre_box_num):
                pred_cls = int(tmp_cls[i])
                pre_box_cls_num[pred_cls] += 1

        reg_TP_batch = np.array(reg_TP)
        reg_TR_batch = np.array(reg_TR)
        cls_TP_batch = np.array(cls_TP)
        cls_TR_batch = np.array(cls_TR)
        reg_TP_strict_batch = np.array(reg_TP_strict)
        reg_TR_strict_batch = np.array(reg_TR_strict)

        gt_box_num_batch = np.array(gt_box_cls_num)
        pre_box_num_batch = np.array(pre_box_cls_num)
        criteria_set=[reg_TP_batch, reg_TR_batch, reg_TP_strict_batch, reg_TR_strict_batch, cls_TP_batch, cls_TR_batch, pre_box_num_batch, gt_box_num_batch]
    else:
        criteria_set=[]
    if ret_box3d.shape[0] == 0:
        ret_box3d_score = np.array([])
    else:
        ret_box3d_score=np.concatenate([ret_cls[:, np.newaxis],ret_box3d, ret_score[:, np.newaxis]], axis=-1)
    #评价指标结束
    #可视化
    if vis:
        img,P, Tr_velo_to_cam, R_cam_to_rect =img
        front_img = draw_lidar_box3d_on_image(img, ret_box3d, ret_score, ret_cls,gtbox3ddraw,gtbox3ddraw_notcare,P2 = P, T_VELO_2_CAM=Tr_velo_to_cam, R_RECT_0=R_cam_to_rect )
        bird_view = lidar_to_bird_view_img(raw_lidar)
        bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d, ret_score, ret_cls, gtbox3ddraw,gtbox3ddraw_notcare,factor=4)
        heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)
        vis_set = [bird_view,heatmap,front_img]
    else:
        vis_set = []
    return ret_box3d_score,criteria_set,vis_set
