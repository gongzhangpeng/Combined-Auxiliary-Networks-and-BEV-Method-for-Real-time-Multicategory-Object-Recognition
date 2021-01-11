import tensorflow as tf
import numpy as np
import math
import core.common as common
from utils.utils_track import *
from core.config import cfg
from utils.colorize import colorize

def cal_anchors_numpy():
    w_value=cfg.W_VALUE
    l_value=cfg.L_VALUE
    h_value=cfg.H_VALUE
    anchors_all=[]
    x = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.FEATURE_WIDTH)
    y = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.FEATURE_HEIGHT)
    cx, cy = np.meshgrid(x, y)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx) * -1.78
    for i in range(cfg.NUM_CLASS):
        w = np.ones_like(cx) * w_value[i]
        l = np.ones_like(cx) * l_value[i]
        h = np.ones_like(cx) * h_value[i]
        r = np.ones_like(cx)
        r[..., 0] = 0  # 0
        r[..., 1] = 90 / 180 * np.pi  
        # 7*(w,l,2) -> (w, l, 2, 7)
        anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)
        anchors_all.append(anchors)
    # 4*(w, l, 2, 7) -> ( w, l, 8, 7)
    anchors_all = np.concatenate(anchors_all,axis=2)
    return  anchors_all #H,W,8,7

anchors_numpy = cal_anchors_numpy()
anchors = anchors_numpy.reshape([cfg.FEATURE_HEIGHT,cfg.FEATURE_WIDTH,8,7])
    
def cspdarknet53(input_data):
    input_data = common.convolutional(input_data, (3, 3,  3,  32), activate_type="leaky")
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="leaky")

    route = input_data
    route = common.convolutional(route, (1, 1, 64, 64), activate_type="leaky")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="leaky")
    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64, activate_type="leaky")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="leaky")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="leaky")
    input_data = common.convolutional(input_data, (3, 3, 64, 128), activate_type="leaky")
    route = input_data
    route = common.convolutional(route, (1, 1, 128, 64), activate_type="leaky")
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="leaky")
    for i in range(2):
        input_data = common.residual_block(input_data, 64,  64, 64, activate_type="leaky")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="leaky")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="leaky")
    input_data = common.convolutional(input_data, (3, 3, 128, 256), activate_type="leaky")
    route = input_data
    route = common.convolutional(route, (1, 1, 256, 128), activate_type="leaky")
    input_data = common.convolutional(input_data, (1, 1, 256, 128), activate_type="leaky")
    for i in range(8):
        input_data = common.residual_block(input_data, 128, 128, 128, activate_type="leaky")
    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="leaky")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="leaky")
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="leaky")
    route = input_data
    route = common.convolutional(route, (1, 1, 512, 256), activate_type="leaky")
    input_data = common.convolutional(input_data, (1, 1, 512, 256), activate_type="leaky")
    for i in range(8):
        input_data = common.residual_block(input_data, 256, 256, 256, activate_type="leaky")
    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="leaky")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="leaky")
    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="leaky")
    route = input_data
    route = common.convolutional(route, (1, 1, 1024, 512), activate_type="leaky")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512), activate_type="leaky")
    for i in range(4):
        input_data = common.residual_block(input_data, 512, 512, 512, activate_type="leaky")
    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="leaky")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 1024, 1024), activate_type="leaky")
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

    conv = common.convolutional(conv, (3, 3, 1024, 1024))
    r_map = common.convolutional(conv, (1, 1, 1024, 12 ), activate=False, bn=False)
    p_map = common.convolutional(conv, (1, 1, 1024, 2 ), activate=False, bn=False)
    c_map = common.convolutional(conv, (1, 1, 1024, cfg.NUM_CLASS ), activate=False, bn=False)
    angle_map = common.convolutional(conv, (1, 1, 1024, cfg.ANGLE_DIVIDE_NUM ), activate=False, bn=False)
    conv_bbox = [r_map, p_map, c_map, angle_map]

    return conv_bbox

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
    #conv_lobj_branch = common.Deconv2D(conv_lobj_branch, 512, 3, 2)
    conv_lobj_branch = common.upsample(conv_lobj_branch)   
    conv = common.convolutional(conv, (1, 1, 256, 256))
    #conv = common.Deconv2D(conv, 256, 3, 2)
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)
    conv_mobj_branch = common.convolutional(conv, (3, 3, 768, 512))
    conv_branch=tf.concat([conv_lobj_branch, conv_mobj_branch], axis=-1)    
    #conv_bbox = common.convolutional(conv_branch, (1, 1, 256, 14 + cfg.NUM_CLASS + cfg.ANGLE_DIVIDE_NUM), activate=False, bn=False)
    conv_bbox = common.convolutional(conv_branch, (1, 1, 512, 8+8*7), activate=False, bn=False)
    #p_map = common.convolutional(conv_branch, (1, 1, 1024, 8 ), activate=False, bn=False)
    #c_map = common.convolutional(conv_branch, (1, 1, 1024, cfg.NUM_CLASS ), activate=False, bn=False)
    #angle_map = common.convolutional(conv_branch, (1, 1, 1024, cfg.ANGLE_DIVIDE_NUM ), activate=False, bn=False)
    #conv_bbox = [r_map, p_map, c_map, angle_map]

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

def compute_loss_and_result(conv_bbox,point_cls,lable,train=True):
    targets,pos_equal_one,neg_equal_one,point_cls_gt_placeholder=lable
    p_map, r_map=tf.split(conv_bbox,[8,8*7],axis=-1)
    p_pos = tf.sigmoid(p_map)
    r_map = tf.reshape(r_map, [-1,r_map.shape[1],r_map.shape[2],8,7])
    
    pos_equal_one_sum = tf.reduce_sum(pos_equal_one)+1
    neg_equal_one_sum = tf.reduce_sum(neg_equal_one)+1
            #loss
    small_addon_for_BCE=1e-10
    point_cls_loss = -(1-point_cls)**2*point_cls_gt_placeholder*tf.math.log(tf.clip_by_value(point_cls, small_addon_for_BCE,1.0))
    point_cls_loss = tf.reduce_mean(tf.reduce_sum(point_cls_loss,axis=-1))
    conf_pos_loss = 1.5*(1-p_pos)**2*(-pos_equal_one * tf.math.log( tf.clip_by_value(p_pos , small_addon_for_BCE,1.0))) / pos_equal_one_sum
    conf_neg_loss = p_pos**2*(-neg_equal_one * tf.math.log(tf.clip_by_value(1-p_pos , small_addon_for_BCE,1.0))) / neg_equal_one_sum
    conf_loss = tf.reduce_sum( conf_pos_loss +  conf_neg_loss)


    reg_loss = pos_equal_one*tf.reduce_sum(smooth_l1(r_map , targets , 3),axis=-1) / pos_equal_one_sum
    reg_loss = 2*tf.reduce_sum(reg_loss)
    delta_output = r_map
    prob_output = p_pos

    loss = conf_loss + reg_loss + point_cls_loss
    result=[prob_output,delta_output]
    child_loss = [conf_loss,conf_pos_loss,conf_neg_loss,reg_loss,point_cls_loss]
    train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', loss),
            tf.summary.scalar('train/reg_loss', reg_loss),
            tf.summary.scalar('train/conf_loss', conf_loss),
            tf.summary.scalar('train/point_cls_loss', point_cls_loss)
        ])
    valid_summary = tf.summary.merge([
            tf.summary.scalar('valid/loss', loss),
            tf.summary.scalar('valid/reg_loss', reg_loss),
            tf.summary.scalar('valid/conf_loss', conf_loss),
            tf.summary.scalar('valid/point_cls_loss', point_cls_loss)
        ])
    if train:
        return loss,result,child_loss,train_summary
    else:
        return loss,result,child_loss,valid_summary

def caculate_nms(boxes2d,boxes2d_scores):
    box2d_ind_after_nms = tf.image.non_max_suppression(
                boxes2d, boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)
    return box2d_ind_after_nms


def bbox_to_corner_tensor(bbox_gt):
    x_tensor,y_tensor,w_tensor,l_tensor,r_tensor = tf.split(bbox_gt,[1,1,1,1,1],axis=-1)
    x0_tensor = -tf.math.cos(r_tensor)*l_tensor/2+tf.math.sin(r_tensor)*w_tensor/2+x_tensor
    y0_tensor = tf.math.sin(r_tensor)*l_tensor/2+tf.math.cos(r_tensor)*w_tensor/2+y_tensor
    x1_tensor = -tf.math.cos(r_tensor)*l_tensor/2-tf.math.sin(r_tensor)*w_tensor/2+x_tensor
    y1_tensor = tf.math.sin(r_tensor)*l_tensor/2-tf.math.cos(r_tensor)*w_tensor/2+y_tensor
    x2_tensor = tf.math.cos(r_tensor)*l_tensor/2-tf.math.sin(r_tensor)*w_tensor/2+x_tensor
    y2_tensor = -tf.math.sin(r_tensor)*l_tensor/2-tf.math.cos(r_tensor)*w_tensor/2+y_tensor
    x3_tensor = tf.math.cos(r_tensor)*l_tensor/2+tf.math.sin(r_tensor)*w_tensor/2+x_tensor
    y3_tensor = -tf.math.sin(r_tensor)*l_tensor/2+tf.math.cos(r_tensor)*w_tensor/2+y_tensor
    corner=tf.concat([x0_tensor,y0_tensor,
                      x1_tensor,y1_tensor,
                      x2_tensor,y2_tensor,
                      x3_tensor,y3_tensor],axis=-1)
    return corner

def box_fill_grid_map_tensor(onebox):
    factor = 2
    x0,y0,x1,y1,x2,y2,x3,y3=tf.split(onebox,[1,1,1,1,1,1,1,1],axis=-1)
    x_index_tmp0 = tf.tile(tf.ones_like(x0)[:,tf.newaxis],[1,int(cfg.INPUT_HEIGHT/factor),int(cfg.INPUT_WIDTH/factor)])
    x_index_tmp1 = tf.tile(tf.range(cfg.X_MIN,cfg.X_MAX,cfg.VOXEL_X_SIZE*factor)[tf.newaxis,:],[int(cfg.INPUT_HEIGHT/factor),1])
    x_index = x_index_tmp0*x_index_tmp1
    y_index_tmp0 = tf.tile(tf.ones_like(y0)[:,tf.newaxis],[1,int(cfg.INPUT_HEIGHT/factor),int(cfg.INPUT_WIDTH/factor)])
    y_index_tmp1 = tf.tile(tf.range(cfg.Y_MIN,cfg.Y_MAX,cfg.VOXEL_Y_SIZE*2)[:,tf.newaxis],[1,int(cfg.INPUT_WIDTH/factor)])
    y_index = y_index_tmp0*y_index_tmp1
    x0 = x0[...,tf.newaxis]
    x1 = x1[...,tf.newaxis]
    x2 = x2[...,tf.newaxis]
    x3 = x3[...,tf.newaxis]
    y0 = y0[...,tf.newaxis]
    y1 = y1[...,tf.newaxis]
    y2 = y2[...,tf.newaxis]
    y3 = y3[...,tf.newaxis]
    a = tf.cast((x1-x0)*(y_index-y0)-(y1-y0)*(x_index-x0)>0,tf.int8)
    a_neg =  tf.cast((x1-x0)*(y_index-y0)-(y1-y0)*(x_index-x0)<0,tf.int8)
    b =  tf.cast((x2-x1)*(y_index-y1)-(y2-y1)*(x_index-x1)>0,tf.int8)
    b_neg =  tf.cast((x2-x1)*(y_index-y1)-(y2-y1)*(x_index-x1)<0,tf.int8)
    c = tf.cast( (x3-x2)*(y_index-y2)-(y3-y2)*(x_index-x2)>0,tf.int8)
    c_neg = tf.cast( (x3-x2)*(y_index-y2)-(y3-y2)*(x_index-x2)<0,tf.int8)
    d =  tf.cast((x0-x3)*(y_index-y3)-(y0-y3)*(x_index-x3)>0,tf.int8)
    d_neg =  tf.cast((x0-x3)*(y_index-y3)-(y0-y3)*(x_index-x3)<0,tf.int8)
    result1=a*b*c*d
    result2=a_neg*b_neg*c_neg*d_neg
    result = tf.cast(result1,tf.int8)+tf.cast(result2,tf.int8)
    return result

def iou_tensor(box, score,threshold=0.1):
    boxes_corner = bbox_to_corner_tensor(box)
    result = box_fill_grid_map_tensor(boxes_corner)
    ind = tf.py_func(_iou_tensor, [result,score,threshold], tf.int32)
    return ind
 
def _iou_tensor(result,score,threshold):
    index = score.argsort()[::-1]
    result = np.array(result[index,:,:])
    N = result.shape[0]
    result_expand = np.tile(result[:,np.newaxis,:,:],[1,N,1,1])
    share = (result+result_expand)==2
    indiv = np.abs(result-result_expand)==1
    share_sum=np.sum(share,axis=(2,3))
    div_sum = np.sum(indiv,axis=(2,3))
    share_sum = np.float32(share_sum)
    div_sum = np.float32(share_sum+div_sum) +1e-6
    iou = share_sum/div_sum
    iou_value_triu = np.triu(iou)
    iou_value_triu_mask = iou_value_triu>threshold
    iou_value_sum = np.sum(iou_value_triu_mask,axis=0)
    ind = np.where(iou_value_sum<=1)
    index_pick = index[ind]    
    return np.int32(index_pick)


boxes2d_scores = tf.placeholder(dtype=tf.float32,shape=[None])
boxes2d_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,4])
box2d_ind_after_nms = caculate_nms(boxes2d_placeholder,boxes2d_scores)
#boxes2d_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,5])
#box2d_ind_after_nms =iou_tensor(boxes2d_placeholder, boxes2d_scores,cfg.RPN_NMS_THRESH)


#模型输出到预测

def predict(sess,probs,deltas,bbox_pad_output, bbox_output,conf_output,class_output,gtbox3d_with_ID,raw_lidar,bev_f,threshold=cfg.RPN_SCORE_THRESH,evaluation=True,vis=True):
    use_tensor = False
    gt_class = gtbox3d_with_ID[0,:,1]
    gtbox3d = gtbox3d_with_ID[0,:,2::]
    if use_tensor:
        tmp_boxes3d = bbox_output
        tmp_boxes2d = bbox_pad_output
        tmp_scores = conf_output
        tmp_cls = class_output
        if tmp_boxes3d.shape[0]!=0:
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
    else:
        cls_max_arg=np.argmax(probs,axis=-1)
        cls_max =np.max(probs,axis=-1)
        deltas = deltas[np.zeros_like(cls_max_arg), 
            np.tile(np.arange(cfg.FEATURE_HEIGHT )[:,np.newaxis],[1,cfg.FEATURE_WIDTH]),
            np.tile(np.arange(cfg.FEATURE_WIDTH)[np.newaxis,:],[cfg.FEATURE_HEIGHT,1]),
            cls_max_arg,:]
        anchors_pick=anchors[np.tile(np.arange(cfg.FEATURE_HEIGHT )[:,np.newaxis],[1,cfg.FEATURE_WIDTH]),
            np.tile(np.arange(cfg.FEATURE_WIDTH)[np.newaxis,:],[cfg.FEATURE_HEIGHT,1]),
            cls_max_arg,:]

        batch_size = probs.shape[0]
        batch_probs = cls_max.reshape([batch_size, -1])
        batch_boxes3d = delta_to_boxes3d(deltas, anchors_pick)
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        ind_b , ind = np.where(batch_probs >= threshold)
        if len(ind) !=0:
            tmp_boxes3d = batch_boxes3d[0, ind, ...]
            tmp_boxes2d = batch_boxes2d[0, ind, ...]
            tmp_scores = batch_probs[0, ind]
            tmp_cls=cls_max_arg.reshape(-1)[ind]

            boxes3d_corner = box3Dcorner_from_gtbox(tmp_boxes3d)
            boxes2d_standard = corner_to_standup_box2d(np.array(boxes3d_corner))

            ind = sess.run(box2d_ind_after_nms, {
                        boxes2d_placeholder: boxes2d_standard,
                        boxes2d_scores: tmp_scores})

            tmp_boxes3d = np.array(tmp_boxes3d[ind, ...])
            tmp_scores =  np.array(tmp_scores[ind])
            tmp_cls= np.int32(np.floor(np.array(tmp_cls[ind])/2))
            boxes2d_standard =  np.array(boxes2d_standard[ind,...])
        else:
            tmp_boxes3d=np.array([])
            tmp_scores=np.array([])
            tmp_cls=np.array([])
            boxes2d_standard=np.array([])
    
    #regress_angle = regression_angle(tmp_boxes3d,bev_f)
    #tmp_boxes3d[:,6] = regress_angle
    #boxes3d_corner = box3Dcorner_from_gtbox(tmp_boxes3d)
    #boxes2d_standard = corner_to_standup_box2d(np.array(boxes3d_corner))
    
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
        bird_view = lidar_to_bird_view_img(raw_lidar)
        bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d, ret_score, ret_cls, gtbox3d,factor=4)
        heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)
        vis_set = [bird_view,heatmap]
    else:
        vis_set = []
    return ret_box3d_score,criteria_set,vis_set
