#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from utils.utils_track import *
from utils.preprocess import makeBVFeature_new,makeBVFeature_addpoint,get_points_gt

anchors = cal_anchors()
class Dataset(object):
    def __init__(self, batch_size, shuffle=True, aug=True):
        self.batch_size = batch_size
        self.num_classes = 4
        self.aug = aug
        f_lidar_set=[]
        f_labels_set=[]
        f_lidar_path=[]
        f_lidar1 = glob.glob(os.path.join('E:\\ApolloDataset\\training\\tracking_train_pcd_1\\','*_frame'))
        f_lidar_path.extend(f_lidar1)
        f_lidar2 = glob.glob(os.path.join('E:\\ApolloDataset\\training\\tracking_train_pcd_2\\','*_frame'))
        f_lidar_path.extend(f_lidar2)
        f_lidar3 = glob.glob(os.path.join('E:\\ApolloDataset\\training\\tracking_train_pcd_3\\','*_frame'))
        f_lidar_path.extend(f_lidar3)
        f_label = os.listdir('E:\\ApolloDataset\\training\\tracking_train_label\\')
        f_lidar_path.sort()
        f_label.sort()
        for f_lidar_child in f_lidar_path:
            f_lidars = glob.glob(os.path.join(f_lidar_child,'*.bin'))
            f_lidars.sort()
            f_lidar_set.extend(f_lidars)

        for f_label_child in f_label:
            f_label_path = os.path.join('E:\\ApolloDataset\\training\\tracking_train_label\\',f_label_child)
            f_labels = glob.glob(os.path.join(f_label_path,'*.txt'))
            f_labels.sort()
            f_labels_set.extend(f_labels)

        assert len(f_lidar_set) != 0, "dataset folder is not correct"
        assert len(f_lidar_set) ==len(f_labels_set), "dataset folder is not correct"
        self.num_samples = len(f_lidar_set)
        self.num_batchs = np.ceil(self.num_samples / self.batch_size).astype(np.int32)
        self.indices = list(range(self.num_samples))
        if shuffle:
            np.random.shuffle(self.indices)
        self.lidar_path = f_lidar_set
        self.label_path = f_labels_set
        self.data_tag = [name.split('\\')[-2]+name.split('\\')[-1].split('.')[-2] for name in f_lidar_set]
        self.batch_count = 0
        self.max_bbox_per_scale = 150
        self.index = 0
        
    def __iter__(self):
        return self
    def __next__(self):
        with tf.device('/CPU:0'):
            feature_map_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]
            batch_bev = np.zeros((self.batch_size, cfg.INPUT_HEIGHT,cfg.INPUT_WIDTH, 3))
            targets_batch = np.zeros((self.batch_size, *feature_map_shape, 12))
            angle_batch = np.zeros((self.batch_size, *feature_map_shape, cfg.ANGLE_DIVIDE_NUM))
            pos_equal_one_batch = np.zeros((self.batch_size, *feature_map_shape, 2))
            neg_equal_one_batch = np.zeros((self.batch_size, *feature_map_shape, 2))
            cls_onehot_batch = np.zeros((self.batch_size, *feature_map_shape, cfg.NUM_CLASS))
            pos_equal_one_for_reg_batch = np.zeros((self.batch_size, *feature_map_shape, 12))
            pos_equal_one_sum_batch = np.zeros((self.batch_size, 1, 1, 1))
            neg_equal_one_sum_batch = np.zeros((self.batch_size, 1, 1, 1))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    if self.index >= self.num_samples:
                        self.index -= self.num_samples    
                    load_index = self.indices[self.index]
                    lidar_path = self.lidar_path[load_index]
                    label_path = self.label_path[load_index]
                    data_tag = self.data_tag[load_index]
                    raw_lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
                    labels = [line for line in open(label_path, 'r').readlines()]
                    gtbox3d_with_ID = gtbox3D_from_label(labels)
                    try:
                        gtbox3d = gtbox3d_with_ID[:,2::]
                    except:
                        print('gtbox3d_with_ID_shap={}'.format(gtbox3d_with_ID.shape))
                        print(label_path)
                        self.index += 1
                        continue
                    self.index += 1
                    tag = data_tag
                    if self.aug:
                        ret = aug_data(tag,raw_lidar,gtbox3d_with_ID)
                        point_suffle,point_cls_suffle,point_reg_suffle,point_inbox_mask_suffle = ret[4],ret[5],ret[6],ret[7]
                        bev_f = ret[2]
                    else:
                        bev_f,point_keep=makeBVFeature_addpoint(raw_lidar)
                        point_suffle,point_cls_suffle,point_reg_suffle,point_inbox_mask_suffle = get_points_gt(point_keep,np.array(gtbox3d_with_ID))
                        ret = [tag, np.array(raw_lidar), np.array(bev_f), np.array(gtbox3d_with_ID)]
                    
                    pos_equal_one, neg_equal_one, targets_all, cls_onehot, angle_onehot =cal_rpn_target(ret,anchors)
                    
                    pos_equal_one_for_reg = np.concatenate(
                        [np.tile(pos_equal_one[..., [0]], 6), np.tile(pos_equal_one[..., [1]], 6)], axis=-1)
                    pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(
                        1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
                    neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(
                        1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

                    targets_batch[num, :, :, :] = targets_all
                    angle_batch[num, :, :, :] = angle_onehot
                    pos_equal_one_batch[num, :, :, :] = pos_equal_one
                    neg_equal_one_batch[num, :, :, :] = neg_equal_one
                    cls_onehot_batch[num, :, :, :] = cls_onehot
                    batch_bev[num, :, :, :] = bev_f
                    pos_equal_one_for_reg_batch[num,:,:,:] = pos_equal_one_for_reg
                    pos_equal_one_sum_batch[num,:,:,:] = pos_equal_one_sum
                    neg_equal_one_sum_batch[num,:,:,:] = neg_equal_one_sum
                    num += 1
                self.batch_count += 1
                label_set = (
                    targets_batch,
                    pos_equal_one_batch,
                    pos_equal_one_sum_batch,
                    pos_equal_one_for_reg_batch,
                    neg_equal_one_batch,
                    neg_equal_one_sum_batch,
                    cls_onehot_batch,
                    angle_batch,
                    point_suffle,
                    point_cls_suffle,
                    point_reg_suffle,
                    point_inbox_mask_suffle)
                return (batch_bev,label_set)
            else:
                self.batch_count = 0
                self.index = 0
                np.random.shuffle(self.indices)
                raise StopIteration
    
    def __len__(self):
        return self.num_batchs
        
def cal_rpn_target(data,anchors):
    tag = data[0]
    raw_lidar = data[1]
    bev_f = data[2]
    gtbox3d_with_ID = data[3]    
    cls_num = cfg.NUM_CLASS   
    feature_map_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]
    anchors_reshaped = anchors.reshape(-1, 7)
    anchors_d = np.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)
    anchors_d = anchors_d.reshape([cls_num, -1 ])
    pos_equal_one = np.zeros((1, *feature_map_shape, 2))
    neg_equal_one = np.zeros((1, cls_num, *feature_map_shape, 2))
    targets_all = np.zeros((1, *feature_map_shape, 12))
    cls_onehot = np.zeros((1, *feature_map_shape, cls_num))
    angle_onehot = np.zeros((1, *feature_map_shape, cfg.ANGLE_DIVIDE_NUM))

    anchors_standup_2d = anchor_to_standup_box2d(anchors_reshaped[:, [0, 1, 4, 5]])
    anchors_standup_2d_reshape=anchors_standup_2d.reshape([cls_num, -1, 4])
    anchors_reshaped=anchors_reshaped.reshape([cls_num, -1, 7])
    targets_set=[]
    iou_record = np.zeros((cls_num,*feature_map_shape))
    gtboxwithclass = gtbox3dwithID_to_gtboxwithclass(gtbox3d_with_ID)
    for i in range(cls_num):
        targets = np.zeros((*feature_map_shape, 12+cfg.ANGLE_DIVIDE_NUM))
        anchors_standup_2d=anchors_standup_2d_reshape[i]#anchor已经是该类别
        gtboxwithclass_corner =  box3Dcorner_from_gtbox(gtboxwithclass[i])
        gtboxwithclass_corner = np.array(gtboxwithclass_corner)        
        box_num = gtboxwithclass_corner.shape[0]
        if box_num==0:
            neg_equal_one[0,i,...] = 1
            targets_set.append(targets)
            continue
        gt_standup_2d = gtbox3d_to_anchor_box2d(gtboxwithclass[i])
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )#h*w*2,gt
        iou_reshape=iou.reshape([cfg.FEATURE_HEIGHT,cfg.FEATURE_WIDTH,-1])#每个位置2*gt_box个iou,200*176
        iou_cls_max=np.max(iou_reshape,axis=-1)
    # find anchor with highest iou(iou should also > 0)
        id_highest = np.argmax(iou, axis=0)
        id_highest_gt = np.arange(iou.shape[1])
        mask = iou[id_highest, id_highest_gt] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > cfg.RPN_POS_IOU)

        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < cfg.RPN_NEG_IOU,
                                 axis=1) == iou.shape[1])[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

        # TODO: uniquify the array in a more scientific way
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, 2))
        pos_equal_one[0,index_x, index_y, index_z] = 1

        h_value=cfg.H_VALUE
        # ATTENTION: index_z should be np.array
        targets[index_x, index_y, np.array(index_z) * 6] = (
            gtboxwithclass[i][id_pos_gt, 0] - anchors_reshaped[i][id_pos, 0]) / anchors_d[i][id_pos]
        targets[index_x, index_y, np.array(index_z) * 6 + 1] = (
            gtboxwithclass[i][id_pos_gt, 1] - anchors_reshaped[i][id_pos, 1]) / anchors_d[i][id_pos]
        targets[index_x, index_y, np.array(index_z) * 6 + 2] = (gtboxwithclass[i][id_pos_gt, 2] - anchors_reshaped[i][id_pos, 2]) / h_value[i]
        targets[index_x, index_y, np.array(index_z) * 6 + 3] = np.log(gtboxwithclass[i][id_pos_gt, 3] / anchors_reshaped[i][id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 6 + 4] = np.log(
            gtboxwithclass[i][id_pos_gt, 4] / anchors_reshaped[i][id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 6 + 5] = np.log(
            gtboxwithclass[i][id_pos_gt, 5] / anchors_reshaped[i][id_pos, 5])
        targets[index_x, index_y, 12::] = angle_to_onehot(gtboxwithclass[i][id_pos_gt, 6] )

        iou_record[i, ...]=iou_cls_max

        targets_set.append(targets)

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, 2))
        neg_equal_one[0, i, index_x, index_y, index_z] = 1
        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, 2))
        neg_equal_one[0, i, index_x, index_y, index_z] = 0

    iou_max_arg = np.argmax(iou_record,axis=0)
    targets_onebatch=np.array(targets_set)[iou_max_arg,
                                       np.tile(np.arange(cfg.FEATURE_HEIGHT)[:,np.newaxis],[1,cfg.FEATURE_WIDTH]),
                                       np.tile(np.arange(cfg.FEATURE_WIDTH)[np.newaxis:],[cfg.FEATURE_HEIGHT,1]),:]
    deta = 0.01
    cls_onehot[0,np.tile(np.arange(
        cfg.FEATURE_HEIGHT)[:,np.newaxis],[1,cfg.FEATURE_WIDTH]),
               np.tile(np.arange(cfg.FEATURE_WIDTH)[np.newaxis:],[cfg.FEATURE_HEIGHT,1]),iou_max_arg]=1
    mask = np.where(cls_onehot != 1)
    cls_onehot = cls_onehot*(1 - deta)+ deta *(1.0/cfg.NUM_CLASS)
    cls_onehot[mask] =  deta * (1.0/cfg.NUM_CLASS)
    targets_all[0,...]=targets_onebatch[:,:,0:12]
    angle_onehot[0,...] = targets_onebatch[:,:,12::]
    neg_equal_one=np.min(neg_equal_one, axis=1)
    return pos_equal_one, neg_equal_one, targets_all, cls_onehot, angle_onehot

class Dataset_predict(object):
    def __init__(self, batch_size, shuffle=True, aug=True):
        self.batch_size = batch_size
        self.num_classes = 4
        self.aug = aug
        self.shuffle = shuffle
        f_lidar_set=[]
        f_labels_set=[]
        f_lidar_path=[]
        f_lidar1 = glob.glob(os.path.join('E:\\ApolloDataset\\validation\\tracking_train_pcd_1\\','*_frame'))
        f_lidar_path.extend(f_lidar1)
        f_lidar2 = glob.glob(os.path.join('E:\\ApolloDataset\\validation\\tracking_train_pcd_2\\','*_frame'))
        f_lidar_path.extend(f_lidar2)
        f_lidar3 = glob.glob(os.path.join('E:\\ApolloDataset\\validation\\tracking_train_pcd_3\\','*_frame'))
        f_lidar_path.extend(f_lidar3)
        f_label = os.listdir('E:\\ApolloDataset\\validation\\tracking_train_label\\')
        f_lidar_path.sort()
        f_label.sort()
        for f_lidar_child in f_lidar_path:
            f_lidars = glob.glob(os.path.join(f_lidar_child,'*.bin'))
            f_lidars.sort()
            f_lidar_set.extend(f_lidars)

        for f_label_child in f_label:
            f_label_path = os.path.join('E:\\ApolloDataset\\validation\\tracking_train_label\\',f_label_child)
            f_labels = glob.glob(os.path.join(f_label_path,'*.txt'))
            f_labels.sort()
            f_labels_set.extend(f_labels)

        assert len(f_lidar_set) != 0, "dataset folder is not correct"
        assert len(f_lidar_set) ==len(f_labels_set), "dataset folder is not correct"
        self.num_samples = len(f_lidar_set)
        self.num_batchs = np.ceil(self.num_samples / self.batch_size).astype(np.int32)
        self.indices = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.lidar_path = f_lidar_set
        self.label_path = f_labels_set
        self.data_tag = [name.split('\\')[-2]+name.split('\\')[-1].split('.')[-2] for name in f_lidar_set]
        self.batch_count = 0
        self.max_bbox_per_scale = 150
        self.index = 0
        
    def __iter__(self):
        return self
    def __next__(self):
        with tf.device('/CPU:0'):
            feature_map_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]
            batch_bev = np.zeros((self.batch_size, cfg.INPUT_HEIGHT,cfg.INPUT_WIDTH, 3))
            
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                
                    if self.index >= self.num_samples:
                        self.index -= self.num_samples    
                    load_index = self.indices[self.index]
                    lidar_path = self.lidar_path[load_index]
                    label_path = self.label_path[load_index]
                    data_tag = self.data_tag[load_index]
                    raw_lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
                    labels = [line for line in open(label_path, 'r').readlines()]
                    gtbox3d_with_ID = gtbox3D_from_label(labels)
                    try:
                        gtbox3d = gtbox3d_with_ID[:,2::]
                    except:
                        print('gtbox3d_with_ID_shap={}'.format(gtbox3d_with_ID.shape))
                        print(label_path)
                        self.index += 1
                        pass
                    self.index += 1
                    tag = data_tag
                    if self.aug:
                        ret = aug_data(tag,raw_lidar,gtbox3d_with_ID)
                        ret = ret[0:4]
                    else:
                        bev_f,point_keep=makeBVFeature_addpoint(raw_lidar)
                        ret = [tag, np.array(raw_lidar), np.array(bev_f), np.array(gtbox3d_with_ID)]
                    batch_bev[0, :, :, :] = bev_f
                    num += 1
                self.batch_count += 1
                return ret
            else:
                self.batch_count = 0
                self.index = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)
                raise StopIteration
    
    def __len__(self):
        return self.num_batchs
        
class Dataset_valid(object):
    def __init__(self, batch_size, shuffle=True, aug=True):
        self.batch_size = batch_size
        self.num_classes = 4
        self.aug = aug
        self.shuffle = shuffle
        f_lidar_set=[]
        f_labels_set=[]
        f_lidar_path=[]
        f_lidar1 = glob.glob(os.path.join('E:\\ApolloDataset\\validation\\tracking_train_pcd_1\\','*_frame'))
        f_lidar_path.extend(f_lidar1)
        f_lidar2 = glob.glob(os.path.join('E:\\ApolloDataset\\validation\\tracking_train_pcd_2\\','*_frame'))
        f_lidar_path.extend(f_lidar2)
        f_lidar3 = glob.glob(os.path.join('E:\\ApolloDataset\\validation\\tracking_train_pcd_3\\','*_frame'))
        f_lidar_path.extend(f_lidar3)
        f_label = os.listdir('E:\\ApolloDataset\\validation\\tracking_train_label\\')
        f_lidar_path.sort()
        f_label.sort()
        for f_lidar_child in f_lidar_path:
            f_lidars = glob.glob(os.path.join(f_lidar_child,'*.bin'))
            f_lidars.sort()
            f_lidar_set.extend(f_lidars)

        for f_label_child in f_label:
            f_label_path = os.path.join('E:\\ApolloDataset\\validation\\tracking_train_label\\',f_label_child)
            f_labels = glob.glob(os.path.join(f_label_path,'*.txt'))
            f_labels.sort()
            f_labels_set.extend(f_labels)

        assert len(f_lidar_set) != 0, "dataset folder is not correct"
        assert len(f_lidar_set) ==len(f_labels_set), "dataset folder is not correct"
        self.num_samples = len(f_lidar_set)
        self.num_batchs = np.ceil(self.num_samples / self.batch_size).astype(np.int32)
        self.indices = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.lidar_path = f_lidar_set
        self.label_path = f_labels_set
        self.data_tag = [name.split('\\')[-2]+name.split('\\')[-1].split('.')[-2] for name in f_lidar_set]
        self.batch_count = 0
        self.max_bbox_per_scale = 150
        self.index = 0
        
    def shuffle_indices(self):
        np.random.shuffle(self.indices)
    def get_data(self):
        with tf.device('/CPU:0'):
            feature_map_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]
            batch_bev = np.zeros((self.batch_size, cfg.INPUT_HEIGHT,cfg.INPUT_WIDTH, 3))
            targets_batch = np.zeros((self.batch_size, *feature_map_shape, 12))
            angle_batch = np.zeros((self.batch_size, *feature_map_shape, cfg.ANGLE_DIVIDE_NUM))
            pos_equal_one_batch = np.zeros((self.batch_size, *feature_map_shape, 2))
            neg_equal_one_batch = np.zeros((self.batch_size, *feature_map_shape, 2))
            cls_onehot_batch = np.zeros((self.batch_size, *feature_map_shape, cfg.NUM_CLASS))
            pos_equal_one_for_reg_batch = np.zeros((self.batch_size, *feature_map_shape, 12))
            pos_equal_one_sum_batch = np.zeros((self.batch_size, 1, 1, 1))
            neg_equal_one_sum_batch = np.zeros((self.batch_size, 1, 1, 1))
            num = 0
            while num < self.batch_size:
                load_index = self.indices[0]
                lidar_path = self.lidar_path[load_index]
                label_path = self.label_path[load_index]
                data_tag = self.data_tag[load_index]
                raw_lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
                labels = [line for line in open(label_path, 'r').readlines()]
                gtbox3d_with_ID = gtbox3D_from_label(labels)
                try:
                    gtbox3d = gtbox3d_with_ID[:,2::]
                except:
                    print('gtbox3d_with_ID_shap={}'.format(gtbox3d_with_ID.shape))
                    print(label_path)
                    np.random.shuffle(self.indices)
                    continue
                tag = data_tag
                if self.aug:
                    ret = aug_data(tag,raw_lidar,gtbox3d_with_ID)
                    point_suffle,point_cls_suffle,point_reg_suffle,point_inbox_mask_suffle = ret[4],ret[5],ret[6],ret[7]
                    bev_f = ret[2]
                else:
                    bev_f,point_keep=makeBVFeature_addpoint(raw_lidar)
                    point_suffle,point_cls_suffle,point_reg_suffle,point_inbox_mask_suffle = get_points_gt(point_keep,np.array(gtbox3d_with_ID))
                    ret = [tag, np.array(raw_lidar), np.array(bev_f), np.array(gtbox3d_with_ID)]
                
                pos_equal_one, neg_equal_one, targets_all, cls_onehot, angle_onehot =cal_rpn_target(ret,anchors)
                
                pos_equal_one_for_reg = np.concatenate(
                    [np.tile(pos_equal_one[..., [0]], 6), np.tile(pos_equal_one[..., [1]], 6)], axis=-1)
                pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(
                    1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
                neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(
                    1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

                targets_batch[num, :, :, :] = targets_all
                angle_batch[num, :, :, :] = angle_onehot
                pos_equal_one_batch[num, :, :, :] = pos_equal_one
                neg_equal_one_batch[num, :, :, :] = neg_equal_one
                cls_onehot_batch[num, :, :, :] = cls_onehot
                batch_bev[num, :, :, :] = bev_f
                pos_equal_one_for_reg_batch[num,:,:,:] = pos_equal_one_for_reg
                pos_equal_one_sum_batch[num,:,:,:] = pos_equal_one_sum
                neg_equal_one_sum_batch[num,:,:,:] = neg_equal_one_sum
                num += 1
            label_set = (
                targets_batch,
                pos_equal_one_batch,
                pos_equal_one_sum_batch,
                pos_equal_one_for_reg_batch,
                neg_equal_one_batch,
                neg_equal_one_sum_batch,
                cls_onehot_batch,
                angle_batch,
                point_suffle,
                point_cls_suffle,
                point_reg_suffle,
                point_inbox_mask_suffle)
            np.random.shuffle(self.indices)
            return (batch_bev,label_set)
