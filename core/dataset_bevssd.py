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

#anchors = cal_anchors()
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
    # 4*(w, l, 2, 7) -> (w, l, 8, 7)
    anchors_all = np.concatenate(anchors_all,axis=2)
    return  anchors_all #H,W,8,7,8=2+2+2+2

anchors_numpy = cal_anchors_numpy()

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
            batch_bev = np.zeros((self.batch_size, cfg.INPUT_HEIGHT,cfg.INPUT_WIDTH, 4))
            targets_batch = np.zeros((self.batch_size, *feature_map_shape, 8,7))
            angle_batch = np.zeros((self.batch_size, *feature_map_shape, cfg.ANGLE_DIVIDE_NUM))
            pos_equal_one_batch = np.zeros((self.batch_size, *feature_map_shape, 8))
            neg_equal_one_batch = np.zeros((self.batch_size, *feature_map_shape, 8))
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
                    
                    pos_equal_one, neg_equal_one, targets_all =cal_rpn_target(ret)
                    
                   

                    targets_batch[num, :, :, :,:] = targets_all
                   
                    pos_equal_one_batch[num, :, :, :] = pos_equal_one
                    neg_equal_one_batch[num, :, :, :] = neg_equal_one
                    
                    batch_bev[num, :, :, :] = bev_f
                    
                    num += 1
                self.batch_count += 1
                label_set = (
                    targets_batch,
                    pos_equal_one_batch,
                    neg_equal_one_batch,
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
        
def cal_rpn_target(data):   
    tag = data[0]
    raw_lidar = data[1]
    bev_f = data[2]
    gtbox3d_with_ID = data[3]
    gtboxwithclass = gtbox3dwithID_to_gtboxwithclass(gtbox3d_with_ID)
    
    cls_num = cfg.NUM_CLASS
    feature_h = cfg.FEATURE_HEIGHT
    feature_w = cfg.FEATURE_WIDTH
    
    anchors_reshaped = anchors_numpy.reshape(-1, 7)
    anchors_standup_2d = anchor_to_standup_box2d(anchors_reshaped[:, [0, 1, 4, 5]])
    anchors_standup_2d_reshape=anchors_standup_2d.reshape([-1, 8, 4])
    anchors_map=np.reshape(anchors_reshaped,[-1,8,7])
    
    w_value=cfg.W_VALUE
    l_value=cfg.L_VALUE
    h_value=cfg.H_VALUE
    anchors_w = np.array(w_value)
    anchors_l = np.array(l_value)
    anchors_h = np.array(h_value)
    anchors_d = np.sqrt(anchors_w**2 +anchors_l**2)
    
    pos_equal_one = np.zeros((1,feature_h,feature_w, 8))
    neg_equal_one = np.zeros((1,feature_h,feature_w, 8))
    targets_all = np.zeros((1,feature_h,feature_w,8, 7))

    for i in range(8):
        j=np.floor(i/2).astype(np.int32)
        anchors_standup_2d=anchors_standup_2d_reshape[:,i,:]
        anchors_pick = anchors_map[:,i,:]
        box_num = gtboxwithclass[j].shape[0]
        if box_num==0:
            neg_equal_one[0,:,:,2*j] = 1
            neg_equal_one[0,:,:,2*j+1] = 1
            continue
        gt_standup_2d = gtbox3d_to_anchor_box2d(gtboxwithclass[j])
        iou = bbox_overlaps(
                    np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
                    np.ascontiguousarray(gt_standup_2d).astype(np.float32),
                )
        
        #iou_reshape=iou.reshape([feature_h,feature_w,-1])
        id_highest = np.argmax(iou, axis=0)
        id_highest_gt = np.arange(iou.shape[1])
        mask = iou[id_highest,id_highest_gt] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]
        id_pos, id_pos_gt = np.where(iou > cfg.RPN_POS_IOU)
        
        iou_max = np.max(iou,axis=-1)
        id_neg = np.where(iou_max < cfg.RPN_NEG_IOU)[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        index_x, index_y = np.unravel_index(id_pos, (feature_h,feature_w))
        pos_equal_one[0, index_x, index_y, i] = 1
        targets_all[0,index_x, index_y,i, 0] = (gtboxwithclass[j][id_pos_gt, 0] -anchors_pick[id_pos,0]) / anchors_d[j]
        targets_all[0,index_x, index_y,i, 1] = (gtboxwithclass[j][id_pos_gt, 1] -anchors_pick[id_pos,1]) / anchors_d[j]
        targets_all[0,index_x, index_y,i, 2] = (gtboxwithclass[j][id_pos_gt, 2] -anchors_pick[id_pos,2]) / anchors_h[j]
        targets_all[0,index_x, index_y,i, 3] = np.log(gtboxwithclass[j][id_pos_gt, 3] / anchors_h[j])
        targets_all[0,index_x, index_y,i, 4] = np.log(gtboxwithclass[j][id_pos_gt, 4] / anchors_w[j])
        targets_all[0,index_x, index_y,i, 5] = np.log(gtboxwithclass[j][id_pos_gt, 5] / anchors_l[j])
        targets_all[0,index_x, index_y,i, 6] = gtboxwithclass[j][id_pos_gt, 6] - anchors_pick[id_pos,6]
        
        index_x_neg, index_y_neg = np.unravel_index(id_neg, (feature_h,feature_w))
        neg_equal_one[0, index_x_neg, index_y_neg, i] = 1
        index_x, index_y = np.unravel_index(id_highest, (feature_h,feature_w))
        neg_equal_one[0, index_x, index_y, i] = 0
    return pos_equal_one, neg_equal_one, targets_all

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
            batch_bev = np.zeros((self.batch_size, cfg.INPUT_HEIGHT,cfg.INPUT_WIDTH, 4))
            
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
            batch_bev = np.zeros((self.batch_size, cfg.INPUT_HEIGHT,cfg.INPUT_WIDTH, 4))
            targets_batch = np.zeros((self.batch_size, *feature_map_shape, 8,7))
            
            pos_equal_one_batch = np.zeros((self.batch_size, *feature_map_shape, 8))
            neg_equal_one_batch = np.zeros((self.batch_size, *feature_map_shape, 8))

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
                
                pos_equal_one, neg_equal_one, targets_all  =cal_rpn_target(ret)
                
                targets_batch[num, :, :, :] = targets_all
                
                pos_equal_one_batch[num, :, :, :] = pos_equal_one
                neg_equal_one_batch[num, :, :, :] = neg_equal_one
                
                batch_bev[num, :, :, :] = bev_f
                
                
                num += 1
            label_set = (
                targets_batch,
                pos_equal_one_batch,
                
                neg_equal_one_batch,

                point_suffle,
                point_cls_suffle,
                point_reg_suffle,
                point_inbox_mask_suffle)
            np.random.shuffle(self.indices)
            return (batch_bev,label_set)
