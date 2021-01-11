import os
import numpy as np
from utils.preprocess_kitti import makeBVFeature_new,makeBVFeature_addpoint,get_points_gtbox
from core.config import cfg
import glob
import math
import cv2
from utils.box_overlaps import *

def cal_anchors():
    #计算标准anchors
    #Output:
    #anchors: (cls_num, w, l, 2, 7), [x,y,z,h,w,l,r][1.31,1.59,3.33],[2.54,2.28,6.81],[1.19,0.51,0.45],[1.22,0.64,1.46]
    w_value= cfg.W_VALUE   #[1.63, 2.54, 0.66, 0.60]
    l_value= cfg.L_VALUE   #[3.88, 16.09, 0.84, 1.76]
    h_value= cfg.H_VALUE    #[1.53, 3.53, 1.76, 1.73]
    anchors_all=[]
    x = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.FEATURE_WIDTH)
    y = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.FEATURE_HEIGHT)
    cx, cy = np.meshgrid(x, y)
    # all is (w, l, 2)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx) * (cfg.Z_GROUND)
    for i in range(len(w_value)):
        w = np.ones_like(cx) * w_value[i]
        l = np.ones_like(cx) * l_value[i]
        h = np.ones_like(cx) * h_value[i]
        r = np.ones_like(cx)
        r[..., 0] = 0  # 0
        r[..., 1] = 90 / 180 * np.pi  # 90
        # 7*(w,l,2) -> (w, l, 2, 7)
        anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)
        anchors_all.append(anchors)
    # 4*(w, l, 2, 7) -> (4, w, l, 2, 7)
    anchors_all=np.array(anchors_all).astype(np.float32)
    return anchors_all

    
#标签转数据部分
def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0): 
    #点的坐标转换
    # Input:
    #   points: (N, 3) 
    # Output:
    #   points: (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])
    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)
    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)
    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)
    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)
    return points[:, 0:3]

def camera_to_lidar(x, y, z, T_VELO_2_CAM=None, R_RECT_0=None):
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(cfg.MATRIX_T_VELO_2_CAM)
    
    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(cfg.MATRIX_R_RECT_0)

    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(R_RECT_0), p)
    p = np.matmul(np.linalg.inv(T_VELO_2_CAM), p)
    p = p[0:3]
    return tuple(p)

def camera_to_lidar_box(boxes, T_VELO_2_CAM=None, R_RECT_0=None,withcls=False):
    # (N, 9) -> (N, 9) id,cls,x,y,z,h,w,l,r
    ret = []    
    for box in boxes:
        _,cls,x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(
            x, y, z, T_VELO_2_CAM, R_RECT_0), h, w, l, -ry - np.pi / 2
        rz = angle_in_limit(rz)
        ret.append([0,cls,x, y, z, h, w, l, rz])
    return np.array(ret).reshape(-1, 9)

def label_to_gt_box3d_kitti(labels, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar'
    # Output:
    #   (N, N', 7)
    acc_cls = ['Car','Van','Truck','Pedestrian', 'Cyclist']
    class_to_value = {'Car':0,'Van':1, 'Truck':1,'Pedestrian':2, 'Cyclist':3}
    boxes3d_a_label = []
    for line in labels:
        ret = line.split()
        if ret[0] in acc_cls:
            h, w, l, x, y, z, r = [float(i) for i in ret[-7:]]
            oc=float(ret[2])
            cls = class_to_value[ret[0]]
            box3d = np.array([oc,cls,x, y, z, h, w, l, r])
            boxes3d_a_label.append(box3d)
    boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label), T_VELO_2_CAM, R_RECT_0)
    boxes3d_a_label = (np.array(boxes3d_a_label).reshape(-1, 9))
    boxes3d_a_label = box_limit_in_range(boxes3d_a_label)
    return boxes3d_a_label

def encode_gtbox_kitti(gtbox3d):
    anchors_h = np.array(cfg.H_VALUE)#kitti
    anchors_w = np.array(cfg.W_VALUE)
    anchors_l = np.array(cfg.L_VALUE)
    anchors_d = np.sqrt(anchors_w**2 +anchors_l**2)
    box_num = gtbox3d.shape[0]
    encode_value = np.zeros([box_num,8])
    for i in range(box_num):
        _,cls,x,y,z,h,w,l,r = gtbox3d[i,:]
        cls_int = int(cls)
        x_int = int((x-cfg.X_MIN)/cfg.VOXEL_X_SIZE)
        x_offset = (x-cfg.X_MIN-x_int*cfg.VOXEL_X_SIZE)/anchors_d[cls_int]
        y_int = int((y-cfg.Y_MIN)/cfg.VOXEL_Y_SIZE)
        y_offset = (y-cfg.Y_MIN-y_int*cfg.VOXEL_Y_SIZE)/anchors_d[cls_int]
        z_offset = (z-cfg.Z_GROUND)/anchors_h[cls_int]
        h_offset = np.log(h/anchors_h[cls_int])
        w_offset = np.log(w/anchors_w[cls_int])
        l_offset = np.log(l/anchors_l[cls_int])
        r_sin_offset = np.sin(r)
        r_cos_offset = np.cos(r)
        encode_value[i,:]=x_offset,y_offset,z_offset,h_offset,w_offset,l_offset,r_sin_offset,r_cos_offset
    return encode_value
    
def label_to_gt_box3d_kitti_evaluate(labels, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar'
    # Output:
    #   (N, N', 7)
    acc_cls = ['Car','Van', 'Truck','Pedestrian', 'Cyclist', 'Person_sitting', 'Tram', 'Misc' , 'DontCare']
    class_to_value = {'Car':0,'Van':1, 'Truck':1,'Pedestrian':2, 'Cyclist':3, 'Person_sitting':4, 'Tram':4,'Misc':4 ,'DontCare':4}
    boxes3d_a_label = []
    boxes3d_a_label_notcare = []
    for line in labels:
        ret = line.split()
        if ret[0] in acc_cls:
            h, w, l, x, y, z, r = [float(i) for i in ret[-7:]]
            cls = class_to_value[ret[0]]
            oc=float(ret[1])
            box_height=float(ret[5])-float(ret[3])
            if oc>0.5 or box_height<25:
                cls = 4
            box3d = np.array([oc,cls,x, y, z, h, w, l, r])
            if cls <4:
                boxes3d_a_label.append(box3d)
            else:
                boxes3d_a_label_notcare.append(box3d)
    boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label), T_VELO_2_CAM, R_RECT_0)
    boxes3d_a_label = (np.array(boxes3d_a_label).reshape(-1, 9))
    boxes3d_a_label_notcare = camera_to_lidar_box(np.array(boxes3d_a_label_notcare), T_VELO_2_CAM, R_RECT_0)
    boxes3d_a_label_notcare = (np.array(boxes3d_a_label_notcare).reshape(-1, 9))
    
    boxes3d_a_label = box_limit_in_range(boxes3d_a_label)
    boxes3d_a_label_notcare = box_limit_in_range(boxes3d_a_label_notcare)
    return boxes3d_a_label,boxes3d_a_label_notcare

def box_limit_in_range(boxes3d_a_label):
    box_num = boxes3d_a_label.shape[0]
    box_keep = np.zeros_like(boxes3d_a_label)
    box_keep_num = 0
    for i in range(box_num):
        ID,cls,x, y, z, h, w, l, r = boxes3d_a_label[i,:]
        if x<cfg.X_MAX and x>cfg.X_MIN and y<cfg.Y_MAX and y>cfg.Y_MIN and z>cfg.Z_MIN and z<cfg.Z_MAX:
            box_keep[box_keep_num,:]=ID,cls,x, y, z, h, w, l, r
            box_keep_num +=1
    return box_keep[0:box_keep_num,:]
    
def gtbox3D_from_label(labels): 
    #标签转gtbox3d
    #Input:
    #label: [id,c,x,y,z,h,w,l,r], N*9
    #Output:
    #boxes: array,[id,class,x,y,z,h,w,l,r], N*9
    boxes=[]
    for label in labels:
        ret = label.split()
        object_id = int(ret[0])
        object_class = int(ret[1])-1
        if 0 <= object_class <= 3: 
            location_x = float(ret[2])
            location_y = float(ret[3])
            location_z = float(ret[4])
            l = float(ret[5])
            w = float(ret[6])
            h = float(ret[7])
            object_head = float(ret[8])
            if cfg.X_MIN <= location_x <= cfg.X_MAX and cfg.Y_MIN <= location_y <= cfg.Y_MAX and cfg.Z_MIN <= location_z <= cfg.Z_MAX:
                boxes.append([object_id,object_class,location_x,location_y,location_z,h,w,l,object_head])
    return np.array(boxes)

def box3Dcorner_from_label(labels): 
    #标签转八顶点坐标形式 N*[8*3]
    #Input:
    #label: [id,c,x,y,z,h,w,l,r], N*9
    #Output:
    #boxes: N list, each list: 8*3,[x,y,z]
    boxes=[]
    for label in labels:
        ret = label.split()
        object_id = int(ret[0])
        object_class = int(ret[1])-1
        if 0 <= object_class <= 3:
            location_x = float(ret[2])
            location_y = float(ret[3])
            location_z = float(ret[4])
            l = float(ret[5])
            w = float(ret[6])
            h = float(ret[7])
            object_head = float(ret[8])
            rotMat = np.array([
                    [np.cos(object_head), -np.sin(object_head), 0.0],
                    [np.sin(object_head), np.cos(object_head), 0.0],
                    [0.0, 0.0, 1.0]])
            trackletBox = np.array([  
                    [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                    [0, 0, 0, 0, h, h, h, h]])
            translation = np.array([location_x,location_y,location_z])
            box = np.dot(rotMat, trackletBox) + \
                    np.tile(translation, (8, 1)).T
            box=np.transpose(box,[1,0])
            
            boxes.append(box)
    return boxes

def box3Dcorner_from_gtbox(gtbox): 
    #gtbox转八顶点坐标形式
    #Input:
    #gtbox, array,[x,y,z,h,w,l,r], N*7
    #Output:
    #boxes: N list, each list: 8*3,[x,y,z]
    boxes=[]
    box_num = gtbox.shape[0]
    for i in range(box_num):
        location_x = gtbox[i,0]
        location_y = gtbox[i,1]
        location_z = gtbox[i,2]
        h = gtbox[i,3]
        w = gtbox[i,4]
        l = gtbox[i,5]
        object_head = gtbox[i,6]
        rotMat = np.array([
                [np.cos(object_head), -np.sin(object_head), 0.0],
                [np.sin(object_head), np.cos(object_head), 0.0],
                [0.0, 0.0, 1.0]])
        trackletBox = np.array([  
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                [0, 0, 0, 0, h, h, h, h]])
        translation = np.array([location_x,location_y,location_z])
        box = np.dot(rotMat, trackletBox) + \
                np.tile(translation, (8, 1)).T
        box=np.transpose(box,[1,0]) 
        boxes.append(box)
    return boxes

def box_transform(gtbox3d,tx, ty, tz, r):
    #gtbox3d转旋转平移变化后的八顶点坐标形式
    #Input:
    #gtbox3d, array,[x,y,z,h,w,l,r], N*7
    #Output:
    #boxes_corner: N list, each list: 8*3,[x,y,z]
    boxes_corner = box3Dcorner_from_gtbox(gtbox3d)
    for i in range(len(boxes_corner)):
        boxes_corner[i] = point_transform(boxes_corner[i], tx, ty, tz, rz=r)
    return boxes_corner

def angle_to_onehot(angle):
    angle_num = angle.shape[0]
    angle100 = (angle+np.pi)*100
    resolution  = 5 #cfg.ANGLE_REVOLUTION
    resolution_eq = np.floor(resolution*np.pi*100/180).astype(np.int32)#cfg.ANGLE_REVOLUTION_EQ
    angle_divide_num = (2*np.pi*100//resolution_eq).astype(np.int32)#cfg.ANGLE_DIVIDE_NUM
    angle_onehot = np.zeros([angle_num,angle_divide_num])
    onehot_num = (angle100//resolution_eq).astype(np.int32)

    for i in range(3):
        k=1-0.3*i
        onehot_num_real1 = np.mod(onehot_num+i,angle_divide_num)
        onehot_num_real2 = np.mod(onehot_num-i,angle_divide_num)
        angle_onehot[np.arange(angle_num),onehot_num_real1] = k
        angle_onehot[np.arange(angle_num),onehot_num_real2] = k
    deta = 0.01
    mask = np.where(angle_onehot == 0)
    angle_onehot = angle_onehot*(1 - deta)+ deta *(1.0/angle_divide_num)
    angle_onehot[mask] =  deta * (1.0/angle_divide_num)
    return angle_onehot


def angle_in_limit(angle):
    #限定angle在[-pi, pi], 忽略微小旋转角度
    limit_degree = 5
    while angle >= np.pi :
        angle -= 2*np.pi
    while angle < -np.pi :
        angle += 2*np.pi
    if abs(angle ) < limit_degree / 180 * np.pi:
        angle = 0
    return angle
    
def angle_in_limit_array(angle):
    #限定angle在[-pi, pi], 忽略微小旋转角度
    angle = np.array(angle)
    mask = np.where(angle>np.pi)
    angle[mask] = angle[mask]- 2*np.pi
    mask = np.where(angle<-np.pi)
    angle[mask] = angle[mask]+ 2*np.pi
    return angle

def regression_angle(boxes3d,bev_f):
    bev_value =bev_f[:,:,0]+bev_f[:,:,1]
    x=np.linspace(cfg.X_MIN,cfg.X_MAX,bev_f.shape[1])
    y=np.linspace(cfg.Y_MIN,cfg.Y_MAX,bev_f.shape[0])
    cx, cy = np.meshgrid(x, y)

    tmp_boxes3d = np.copy(boxes3d)
    angle_origin = tmp_boxes3d[:,6]
    tmp_boxes3d[:,4]=1.1*tmp_boxes3d[:,4]
    tmp_boxes3d[:,5]=1.1*tmp_boxes3d[:,5]
    box_num = tmp_boxes3d.shape[0]
    tmp2=np.array(gtbox3d_to_anchor_box2d(tmp_boxes3d))
    tmp2_reshape = tmp2.reshape([-1,4])
    cx_mask1 = cx[...,np.newaxis]>=tmp2_reshape[:,0]
    cx_mask2 = cx[...,np.newaxis]<=tmp2_reshape[:,2]
    cy_mask1 = cy[...,np.newaxis]>=tmp2_reshape[:,1]
    cy_mask2 = cy[...,np.newaxis]<=tmp2_reshape[:,3]
    mask = cx_mask1*cx_mask2*cy_mask1*cy_mask2
    mask = np.transpose(mask,[2,0,1])
    p_value = mask*bev_value
    p_value_sum = np.sum(p_value,axis=(1,2))

    tmp_boxes3d_2 = np.copy(boxes3d)
    tmp_boxes3d_2[:,4]=1.1*tmp_boxes3d_2[:,4]
    tmp_boxes3d_2[:,5]=1.1*tmp_boxes3d_2[:,5]
    tmp_boxes3d_2[:,6] = tmp_boxes3d_2[:,6]+np.pi/2
    tmp2=np.array(gtbox3d_to_anchor_box2d(tmp_boxes3d_2))
    tmp2_reshape = tmp2.reshape([-1,4])
    cx_mask1 = cx[...,np.newaxis]>=tmp2_reshape[:,0]
    cx_mask2 = cx[...,np.newaxis]<=tmp2_reshape[:,2]
    cy_mask1 = cy[...,np.newaxis]>=tmp2_reshape[:,1]
    cy_mask2 = cy[...,np.newaxis]<=tmp2_reshape[:,3]
    mask = cx_mask1*cx_mask2*cy_mask1*cy_mask2
    mask = np.transpose(mask,[2,0,1])
    p_value_2 = mask*bev_value
    p_value_sum_2 = np.sum(p_value_2,axis=(1,2))

    y_map = np.tile(np.linspace(cfg.Y_MIN,cfg.Y_MAX,bev_f.shape[0])[...,np.newaxis],[1,bev_f.shape[1]])
    x_map = np.tile(np.linspace(cfg.X_MIN,cfg.X_MAX,bev_f.shape[1])[np.newaxis,...],[bev_f.shape[0],1])
    angle_set=np.zeros(box_num)
    for i in range(box_num):
        if p_value_sum_2[i]>p_value_sum[i]:
            p_value[i] = p_value_2[i]
            angle_set[i] += np.pi/2
        ind_x = np.where(p_value[i]!=0)
        probs_map = p_value[i][ind_x].reshape(1,-1)
        x_map_pick = np.array(x_map[ind_x]).reshape(1,-1)
        y_map_pick = np.array(y_map[ind_x]).reshape(1,-1)
        x_center = tmp_boxes3d[i,0]
        y_center = tmp_boxes3d[i,1]
        a= np.arange(-np.pi,np.pi,0.02)
        A= np.tan(a)[np.newaxis,...] #1,315
        B= -1
        C= y_center[...,np.newaxis]-A * x_center[...,np.newaxis]#5,315
        d_map = np.abs(np.dot(x_map_pick[...,np.newaxis],A)-y_map_pick[...,np.newaxis]+C[:,np.newaxis,:])/np.sqrt(A**2+1)[:,np.newaxis,:]*probs_map[...,np.newaxis]
        try:
            d_map_max =  np.max(d_map[0],axis=0)
            arg_d= np.argmin(d_map_max,axis=-1)
            angle = arg_d*0.02-np.pi
            angle_set[i] += angle
        except:
            angle_set[i] += angle_origin[i]
    return angle_set
    
def corner_to_center_box3d(boxes_corner):
    #八顶点形式还原成gtbox3d
    #Input:
    #boxes_corner: N list, each list: 8*3,[x,y,z]
    #Output:
    #gtbox3d, array,[x,y,z,h,w,l,r], N*7
    ret = []
    for roi in boxes_corner:     
        roi = np.array(roi)
        h = abs(np.sum(roi[:4, 2] - roi[4:, 2]) / 4)
        w = np.sum(
            np.sqrt(np.sum((roi[0, [0, 1]] - roi[3, [0, 1]])**2)) +
            np.sqrt(np.sum((roi[1, [0, 1]] - roi[2, [0, 1]])**2)) +
            np.sqrt(np.sum((roi[4, [0, 1]] - roi[7, [0, 1]])**2)) +
            np.sqrt(np.sum((roi[5, [0, 1]] - roi[6, [0, 1]])**2))
        ) / 4
        l = np.sum(
            np.sqrt(np.sum((roi[0, [0, 1]] - roi[1, [0, 1]])**2)) +
            np.sqrt(np.sum((roi[2, [0, 1]] - roi[3, [0, 1]])**2)) +
            np.sqrt(np.sum((roi[4, [0, 1]] - roi[5, [0, 1]])**2)) +
            np.sqrt(np.sum((roi[6, [0, 1]] - roi[7, [0, 1]])**2))
        ) / 4
        x = np.sum(roi[:, 0], axis=0)/ 8
        y = np.sum(roi[:, 1], axis=0)/ 8
        z = np.sum(roi[0:4, 2], axis=0)/ 4
        ry = np.sum(
            math.atan2(roi[2, 0] - roi[1, 0], roi[2, 1] - roi[1, 1]) +
            math.atan2(roi[6, 0] - roi[5, 0], roi[6, 1] - roi[5, 1]) +
            math.atan2(roi[3, 0] - roi[0, 0], roi[3, 1] - roi[0, 1]) +
            math.atan2(roi[7, 0] - roi[4, 0], roi[7, 1] - roi[4, 1]) +
            math.atan2(roi[0, 1] - roi[1, 1], roi[1, 0] - roi[0, 0]) +
            math.atan2(roi[4, 1] - roi[5, 1], roi[5, 0] - roi[4, 0]) +
            math.atan2(roi[3, 1] - roi[2, 1], roi[2, 0] - roi[3, 0]) +
            math.atan2(roi[7, 1] - roi[6, 1], roi[6, 0] - roi[7, 0])
        ) / 8
        ry = angle_in_limit(ry)
        # if w > l:
            # w, l = l, w
            # ry = angle_in_limit(ry + np.pi / 2)
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret)

#数据增强部分
def aug_data(tag,lidar,gtbox3d_with_ID):
    #数据增强及体素化
    #Input:
    #tag:str, lidar:K*4, gtbox3d_with_ID:N*9
    #Output:
    #array, str, K*4, 400*352*3, N*9
    gtbox3d = gtbox3d_with_ID[:,2::]
    box3d_corner = box3Dcorner_from_gtbox(gtbox3d)
    choice = np.random.randint(0, 10)
    if choice >= 7 and gtbox3d.shape[0] < 20:
        for idx in range(gtbox3d.shape[0]):
            is_collision = True
            _count = 0
            while is_collision and _count < 10:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = np.random.normal()
                t_y = np.random.normal()
                t_z = np.random.normal()
                # check collision
                tmp_corner = box_transform(gtbox3d[idx][np.newaxis,...], t_x, t_y, t_z, t_rz)
                tmp_center = corner_to_center_box3d(tmp_corner)            
                is_collision = False
                for idy in range(idx):
                    x1, y1, w1, l1, r1 = tmp_center[0][[0, 1, 4, 5, 6]]
                    x2, y2, w2, l2, r2 = gtbox3d[idy][[0, 1, 4, 5, 6]]
                    iou = cal_iou2d(np.array([x1, y1, w1, l1, r1], dtype=np.float32),
                                    np.array([x2, y2, w2, l2, r2], dtype=np.float32))
                    if iou > 0:
                        is_collision = True
                        _count += 1
                        break
            if not is_collision:
                box_corner = box3d_corner[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(
                    lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(
                    lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(
                    lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(
                    np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
                tmp_corner = box_transform(gtbox3d[idx][np.newaxis,...], t_x, t_y, t_z, t_rz)
                gtbox3d[idx] = corner_to_center_box3d(tmp_corner)
        newtag = 'aug_{}_1_{}'.format(tag, np.random.randint(1, 1024))
    elif choice < 7 and choice >= 4:
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)   
        tmp_corner = box_transform(gtbox3d, 0, 0, 0, angle)
        gtbox3d = corner_to_center_box3d(tmp_corner)
        newtag = 'aug_{}_2_{:.4f}'.format(tag, angle).replace('.', '_')
    else:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        gtbox3d[:, 0:6] = gtbox3d[:, 0:6] * factor
        newtag = 'aug_{}_3_{:.4f}'.format(tag, factor).replace('.', '_')

    gtbox3d_with_ID[:,2::] = gtbox3d
    bev_f,point_keep=makeBVFeature_addpoint(lidar)
    point_suffle,point_cls_suffle,point_in_box_gt_value, point_in_box_weight_gt_value = get_points_gtbox(point_keep,np.array(gtbox3d_with_ID))
    return newtag, np.array(lidar), np.array(bev_f), np.array(gtbox3d_with_ID),point_suffle,point_cls_suffle,point_in_box_gt_value, point_in_box_weight_gt_value

def cal_iou2d(box1, box2):
    #离散统计的方式计算两个盒体的iou
    #Input:
    #box1/2: array, [x, y, w, l, r]
    #Output:
    #iou: float
    buf1 = np.zeros((cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 3))
    buf2 = np.zeros((cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 3))        
    tmp_con = np.concatenate([box1[np.newaxis,:],box2[np.newaxis,:]],axis=0)
    tmp = box2Dcorner_from_gtbox(tmp_con)
    box1_corner = batch_lidar_to_bird_view(tmp[0]).astype(np.int32)
    box2_corner = batch_lidar_to_bird_view(tmp[1]).astype(np.int32)
    buf1 = cv2.fillConvexPoly(buf1, box1_corner, color=(1,1,1))[..., 0]
    buf2 = cv2.fillConvexPoly(buf2, box2_corner, color=(1,1,1))[..., 0]
    indiv = np.sum(np.absolute(buf1-buf2))
    share = np.sum((buf1 + buf2) == 2)
    if indiv == 0:
        return 0.0 # when target is out of bound
    return share / (indiv + share)

def batch_lidar_to_bird_view(points, factor=1):
    #原始点云转成图像坐标系下的鸟瞰点云图
    #Input:
    #points (N, >=2)
    #Output:
    #points (N, 2)
    a = (points[:, 0] - cfg.X_MIN) / cfg.VOXEL_X_SIZE * factor
    b = (points[:, 1] - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor
    a = np.clip(a, a_max=(cfg.X_MAX - cfg.X_MIN) / cfg.VOXEL_X_SIZE * factor, a_min=0)
    b = np.clip(b, a_max=(cfg.Y_MAX - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor, a_min=0)
    return np.concatenate([a[:, np.newaxis], b[:, np.newaxis]], axis=-1)

def box2Dcorner_from_gtbox(gtbox): # 转4*[x,y,z] N*[4*2]
    #gtbox转成平面二维盒体4顶点形式
    #Input:
    #gtbox, array, (N, 5), [x,y,w,l,r] 
    #Output:
    #boxes: list, N*(4, 2), [x,y]
    boxes=[]
    box_num = gtbox.shape[0]
    for i in range(box_num):
        location_x = gtbox[i,0]
        location_y = gtbox[i,1]
        w = gtbox[i,2]
        l = gtbox[i,3]
        object_head = gtbox[i,4]
        rotMat = np.array([
                [np.cos(object_head), -np.sin(object_head)],
                [np.sin(object_head), np.cos(object_head)]])
        trackletBox = np.array([  
                [-l / 2, -l / 2, l / 2, l / 2], \
                [w / 2, -w / 2, -w / 2, w / 2]])
        translation = np.array([location_x,location_y])
        box = np.dot(rotMat, trackletBox) + \
                np.tile(translation, (4, 1)).T
        box=np.transpose(box,[1,0]) 
        boxes.append(box)
    return boxes



def anchor_to_standup_box2d(anchors):
    #将标准anchors转换成平面四顶点形式,标准anchors需要提前reshape
    #Input:
    #anchors: (N, 4), [x,y,w,l]
    #Output:
    #anchor_standup: (N, 4), [x1,y1,x2,y2]
    anchor_standup = np.zeros_like(anchors,dtype=np.float32)
    # r == 0
    anchor_standup[::2, 0] = anchors[::2, 0] - anchors[::2, 3] / 2
    anchor_standup[::2, 1] = anchors[::2, 1] - anchors[::2, 2] / 2
    anchor_standup[::2, 2] = anchors[::2, 0] + anchors[::2, 3] / 2
    anchor_standup[::2, 3] = anchors[::2, 1] + anchors[::2, 2] / 2
    # r == pi/2
    anchor_standup[1::2, 0] = anchors[1::2, 0] - anchors[1::2, 2] / 2
    anchor_standup[1::2, 1] = anchors[1::2, 1] - anchors[1::2, 3] / 2
    anchor_standup[1::2, 2] = anchors[1::2, 0] + anchors[1::2, 2] / 2
    anchor_standup[1::2, 3] = anchors[1::2, 1] + anchors[1::2, 3] / 2

    return anchor_standup

    
    
def gtbox3dwithID_to_gtboxwithclass(gtbox3d_with_ID):
    #将gtbox3d_withID转换成class区分的gtbox3d形式
    #Input:
    #gtbox3d_with_ID: array, (N, 9), [ID,cls,x,y,z,h,w,l,r]
    #Output:
    #boxes3d: list, cls_num*(N, 7), [x,y,z,h,w,l,r]
    cls_num = cfg.NUM_CLASS
    boxes3d = []
    for j in range(cls_num):
        boxes3d.append([])
    box_num = gtbox3d_with_ID.shape[0]
    for i in range(box_num):
        cls,x,y,z,h,w,l,r= gtbox3d_with_ID[i][1::]
        cls = int(cls)
        if  0<= cls < cls_num:
            boxes3d[cls].append(np.array([x,y,z,h,w,l,r]))    
    for j in range(cls_num):
        boxes3d[j]=np.array(boxes3d[j])               
    return boxes3d

def gtbox3d_to_anchor_box2d(gtbox3d):
    #将gtbox3d转换成平面正置二维box
    #Input:
    #gtbox3d: array, (N, 7), [x,y,z,h,w,l,r],x与l对应；y与w对应
    #Output:
    #standup_boxes2d: array, (N, 4), [x1,y1,x2,y2]
    N = gtbox3d.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = gtbox3d[:, 0]- gtbox3d[:, 5]/2#x_min
    standup_boxes2d[:, 1] = gtbox3d[:, 1]- gtbox3d[:, 4]/2#y_min
    standup_boxes2d[:, 2] = gtbox3d[:, 0]+ gtbox3d[:, 5]/2#x_max
    standup_boxes2d[:, 3] = gtbox3d[:, 1]+ gtbox3d[:, 4]/2#y_max

    return standup_boxes2d
    
def corner_to_standup_box2d(boxes_corner):
    #将gtbox3d把顶点格式转换成平面等效二维box
    #Input:
    #boxes_corner: array, (N, 8, 3), [x,y,z]
    #Output:
    #standup_boxes2d: array, (N, 4), [x1,y1,x2,y2]
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)#x_min
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)#y_min
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)#x_max
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)#y_max

    return standup_boxes2d
    
#预测用的函数
def delta_to_boxes3d(deltas, anchors):
    #模型输出转成boxes3d
    #Input:
    #deltas: (N, h, w, 7)
    #anchors: (h, w, 7) anchor是对应cls下的anchor
    #Ouput:
    #boxes3d: (N, w*l*2, 7)
    anchors_reshaped = anchors.reshape(-1, 7)#h*w,7
    batch_size = deltas.shape[0]
    deltas = deltas.reshape(batch_size, -1, 7)#b,h*w,7
    anchors_d = np.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)#w*l*2
    boxes3d = np.zeros_like(deltas)#b,w*l*2,7
    boxes3d[..., [0, 1]] = deltas[..., [0, 1]] * \
        anchors_d[:, np.newaxis] + anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = deltas[..., [2]] * \
        anchors_reshaped[..., [3]] + anchors_reshaped[..., [2]]
    boxes3d[..., [3, 4, 5]] = np.exp(np.clip(
        deltas[..., [3, 4, 5]],a_min=-5,a_max=5)) * anchors_reshaped[..., [3, 4, 5]]
    boxes3d[..., [6]] = deltas[..., [6]]+anchors_reshaped[..., [6]]
    return boxes3d

def delta_to_boxes3d_for_tan_angle(deltas, anchors):
    #模型输出转成boxes3d
    #Input:
    #deltas: (N, h, w, 7)
    #anchors: (h, w, 7) anchor是对应cls下的anchor
    #Ouput:
    #boxes3d: (N, w*l*2, 7)
    anchors_reshaped = anchors.reshape(-1, 7)#h*w,7
    batch_size = deltas.shape[0]
    deltas = deltas.reshape(batch_size, -1, 8)#b,h*w,7
    anchors_d = np.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)#w*l*2
    boxes3d = np.zeros_like(deltas)#b,w*l*2,7
    boxes3d[..., [0, 1]] = deltas[..., [0, 1]] * \
        anchors_d[:, np.newaxis] + anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = deltas[..., [2]] * \
        anchors_reshaped[..., [3]] + anchors_reshaped[..., [2]]
    boxes3d[..., [3, 4, 5]] = np.exp(np.clip(
        deltas[..., [3, 4, 5]],a_min=-5,a_max=5)) * anchors_reshaped[..., [3, 4, 5]]
    boxes3d[..., [6]] = np.arctan2(np.sin(deltas[..., [6]]),np.cos(deltas[..., [6]]))
    return boxes3d[...,0:7]
    
def lidar_to_bird_view_img(lidar, factor=1):
    #原始点云到鸟瞰点云图
    #Input:
    #lidar: (N', 4)
    #Output:
    #birdview: (w, l, 3)
    birdview = np.zeros(
        (cfg.INPUT_HEIGHT * factor, cfg.INPUT_WIDTH * factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if cfg.X_MIN < x < cfg.X_MAX and cfg.Y_MIN < y < cfg.Y_MAX:
            x, y = int((x - cfg.X_MIN) / cfg.VOXEL_X_SIZE *
                       factor), int((y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) *
                       5 * factor, a_min=0, a_max=255)
    birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview

def lidar_to_bird_view(x, y, factor=1):
    # using the cfg.INPUT_XXX
    a = (x - cfg.X_MIN) / cfg.VOXEL_X_SIZE * factor
    b = (y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor
    a = np.clip(a, a_max=(cfg.X_MAX - cfg.X_MIN) / cfg.VOXEL_X_SIZE * factor, a_min=0)
    b = np.clip(b, a_max=(cfg.Y_MAX - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor, a_min=0)
    return a, b

def draw_lidar_box3d_on_birdview(birdview, boxes3d, scores, cls,gt_boxes3d=np.array([]),gt_boxes3d_notcare=np.array([]),
                                 color=(0, 255, 255), gt_color=(255, 0, 255), gt_color_notcare=(100, 50, 100), thickness=1, factor=1):
    #在鸟瞰图上绘制边框
    #Input:
    #birdview: (h, w, 3)
    #boxes3d (N, 7) [x, y, z, h, w, l, r]
    #scores (N,1)
    #cls(N,1)
    #gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    shape_x,shape_y = birdview.shape[0:2]
    img = cv2.resize(birdview,(shape_x*factor,shape_y*factor))
    value_to_class = ['VEH','TRUCK', 'PED','CYC']
    corner_boxes3d = box3Dcorner_from_gtbox(boxes3d)
    corner_gt_boxes3d = box3Dcorner_from_gtbox(gt_boxes3d)
    corner_gt_boxes3d_notcare = box3Dcorner_from_gtbox(gt_boxes3d_notcare)
    # draw gt
    for box in corner_gt_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 gt_color, thickness, cv2.LINE_AA)
                 
    for box in corner_gt_boxes3d_notcare:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 gt_color_notcare, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 gt_color_notcare, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 gt_color_notcare, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 gt_color_notcare, thickness, cv2.LINE_AA)
    # draw detections
    box_index = 0
    for box in corner_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)
        
        color = (int(cls[box_index]*40),255,255)
        score_tmp=scores[box_index]
        color=(int(cls[box_index]*40),255,255)
        lable_txt=value_to_class[int(cls[box_index])]+':'+str(round(score_tmp,3))
        cv2.putText(img, lable_txt, (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 color, thickness, cv2.LINE_AA)
        box_index += 1
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)



#gtbox3d_with_ID到yolov4的label接口
def gtbox3d_2_lable(gtbox3d_with_ID):
    strides = np.array([2,4],dtype=np.int32)
    ANCHORS = np.array([[1.31,1.59,3.33],[2.54,2.28,6.81],[1.19,0.51,0.45],[1.22,0.64,1.46]],dtype=np.float32)
    train_output_sizes = cfg.TRAIN.INPUT_SIZE//strides
    label = [np.zeros(
        (
            train_output_sizes[i],
            train_output_sizes[i],
            cfg.NUM_CLASS,
            8 + cfg.NUM_CLASS,
        ),dtype=np.float32) for i in range(2)]

    bboxes_xyzhwlr = [np.zeros((150, 7),dtype=np.float32) for _ in range(2)]
    bbox_count = np.zeros((2,),dtype=np.int32)
    box_num = gtbox3d_with_ID.shape[0]
    for i in range(box_num):
        box_id,box_class,x,y,z,h,w,l,r = gtbox3d_with_ID[i,:]
        onehot = np.zeros(cfg.NUM_CLASS, dtype=np.float32)
        onehot[np.array([box_class]).astype(np.int32)[0]] = 1.0
        uniform_distribution = np.full(cfg.NUM_CLASS, 1.0 / cfg.NUM_CLASS)
        deta = 0.01
        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
        bbox_xyzhwl = gtbox3d_with_ID[i,2:8]
        bbox_xyzhwlr = gtbox3d_with_ID[i,2:9]
        bbox_xyzhwl_scaled = 1.0 * bbox_xyzhwl[np.newaxis, :] / strides[:, np.newaxis]
        bbox_xyzhwl_scaled[:,3:6] = bbox_xyzhwl[3:6]

        r_scaled = np.tile(np.array([r])[np.newaxis,:],[2,1])
        bbox_xyzhwlr_scaled = np.concatenate([bbox_xyzhwl_scaled,r_scaled],axis=-1)

        iou = []
        exist_positive = False
        for j in range(2):
            anchors_xyzhwl = np.zeros((cfg.NUM_CLASS, 7))
            anchors_xyzhwl[:, 3:6] = ANCHORS[:,0:3]
            anchors_xyzhwl[:, 0:3] = (np.floor(bbox_xyzhwlr_scaled[j, 0:3]) + 0.5)    
            iou_scale = bbox_iou_from_pre_numpy(bbox_xyzhwlr_scaled[j,:][np.newaxis,:],anchors_xyzhwl,mode='iou')
            iou.append(iou_scale)
            iou_mask = iou_scale > 0.3 
            if np.any(iou_mask):
                xind, yind = np.floor(bbox_xyzhwlr_scaled[j, 0:2]).astype(np.int32)
                label[j][yind, xind, iou_mask, :] = 0
                label[j][yind, xind, iou_mask, 0:7] = bbox_xyzhwlr
                label[j][yind, xind, iou_mask, 7:8] = 1.0
                label[j][yind, xind, iou_mask, 8:] = smooth_onehot

                bbox_ind = (bbox_count[j] % 150).astype(np.int32)
                bboxes_xyzhwlr[j][bbox_ind, 0:7] = bbox_xyzhwlr
                bbox_count[j] += 1
                exist_positive = True
        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = (best_anchor_ind / cfg.NUM_CLASS).astype(np.int32)
            best_anchor = (best_anchor_ind % cfg.NUM_CLASS).astype(np.int32)
            xind, yind = np.floor(bbox_xyzhwlr_scaled[best_detect, 0:2]).astype(np.int32)
            label[best_detect][yind, xind, best_anchor, :] = 0
            label[best_detect][yind, xind, best_anchor, 0:7] = bbox_xyzhwlr
            label[best_detect][yind, xind, best_anchor, 7:8] = 1.0
            label[best_detect][yind, xind, best_anchor, 8:] = smooth_onehot

            bbox_ind = (bbox_count[best_detect] % 150).astype(np.int32)
            bboxes_xyzhwlr[best_detect][bbox_ind, 0:7] = bbox_xyzhwlr
            bbox_count[best_detect] += 1
        label_mbbox, label_lbbox = label
        mbboxes, lbboxes = bboxes_xyzhwlr
    return label_mbbox[np.newaxis,...],label_lbbox[np.newaxis,...],mbboxes[np.newaxis,...],lbboxes[np.newaxis,...]

#pre（xyzhwlr）计算等效iou
def bbox_iou_from_pre(pre,label,mode='iou'):
    pre_xmin = tf.minimum(pre[..., 0:1] - pre[..., 5:6] * 0.5*np.cos(pre[..., 6:7])-pre[..., 4:5] * 0.5*np.sin(pre[..., 6:7]),
              pre[..., 0:1] + pre[..., 5:6] * 0.5*np.cos(pre[..., 6:7])+pre[..., 4:5] * 0.5*np.sin(pre[..., 6:7]))
    pre_ymin = tf.minimum( pre[..., 1:2] - pre[..., 5:6] * 0.5*np.sin(pre[..., 6:7])-pre[..., 4:5] * 0.5*np.cos(pre[..., 6:7]),
              pre[..., 1:2] + pre[..., 5:6] * 0.5*np.sin(pre[..., 6:7])+pre[..., 4:5] * 0.5*np.cos(pre[..., 6:7]))
    pre_zmin = pre[..., 2:3]
    
    pre_xmax = tf.maximum(pre[..., 0:1] - pre[..., 5:6] * 0.5*np.cos(pre[..., 6:7])-pre[..., 4:5] * 0.5*np.sin(pre[..., 6:7]),
              pre[..., 0:1] + pre[..., 5:6] * 0.5*np.cos(pre[..., 6:7])+pre[..., 4:5] * 0.5*np.sin(pre[..., 6:7]))
    pre_ymax = tf.maximum( pre[..., 1:2] - pre[..., 5:6] * 0.5*np.sin(pre[..., 6:7])-pre[..., 4:5] * 0.5*np.cos(pre[..., 6:7]),
              pre[..., 1:2] + pre[..., 5:6] * 0.5*np.sin(pre[..., 6:7])+pre[..., 4:5] * 0.5*np.cos(pre[..., 6:7]))
    pre_zmax = pre[..., 2:3] + pre[..., 3:4]

    label_xmin = tf.minimum(label[..., 0:1] - label[..., 5:6] * 0.5*np.cos(label[..., 6:7])-label[..., 4:5] * 0.5*np.sin(label[..., 6:7]),
              label[..., 0:1] + label[..., 5:6] * 0.5*np.cos(label[..., 6:7])+label[..., 4:5] * 0.5*np.sin(label[..., 6:7]))
    label_ymin = tf.minimum(label[..., 1:2] - label[..., 5:6] * 0.5*np.sin(label[..., 6:7])-label[..., 4:5] * 0.5*np.cos(label[..., 6:7]),
              label[..., 1:2] + label[..., 5:6] * 0.5*np.sin(label[..., 6:7])+label[..., 4:5] * 0.5*np.cos(label[..., 6:7]))
    label_zmin = label[..., 2:3]
    label_xmax = tf.maximum(label[..., 0:1] - label[..., 5:6] * 0.5*np.cos(label[..., 6:7])-label[..., 4:5] * 0.5*np.sin(label[..., 6:7]),
              label[..., 0:1] + label[..., 5:6] * 0.5*np.cos(label[..., 6:7])+label[..., 4:5] * 0.5*np.sin(label[..., 6:7]))
    label_ymax = tf.maximum(label[..., 1:2] - label[..., 5:6] * 0.5*np.sin(label[..., 6:7])-label[..., 4:5] * 0.5*np.cos(label[..., 6:7]),
              label[..., 1:2] + label[..., 5:6] * 0.5*np.sin(label[..., 6:7])+label[..., 4:5] * 0.5*np.cos(label[..., 6:7]))
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

def safe_divide(a,b):
    c = np.zeros_like(b)
    mask = (b !=0)
    c[mask]=a[mask]/b[mask]
    return c

def bbox_iou_from_pre_numpy(pre,label,mode='iou'):
    pre_xmin = np.minimum(pre[..., 0:1] - pre[..., 5:6] * 0.5*np.cos(pre[..., 6:7])-pre[..., 4:5] * 0.5*np.sin(pre[..., 6:7]),
              pre[..., 0:1] + pre[..., 5:6] * 0.5*np.cos(pre[..., 6:7])+pre[..., 4:5] * 0.5*np.sin(pre[..., 6:7]))
    pre_ymin = np.minimum( pre[..., 1:2] - pre[..., 5:6] * 0.5*np.sin(pre[..., 6:7])-pre[..., 4:5] * 0.5*np.cos(pre[..., 6:7]),
              pre[..., 1:2] + pre[..., 5:6] * 0.5*np.sin(pre[..., 6:7])+pre[..., 4:5] * 0.5*np.cos(pre[..., 6:7]))
    pre_zmin = pre[..., 2:3]
    
    pre_xmax = np.maximum(pre[..., 0:1] - pre[..., 5:6] * 0.5*np.cos(pre[..., 6:7])-pre[..., 4:5] * 0.5*np.sin(pre[..., 6:7]),
              pre[..., 0:1] + pre[..., 5:6] * 0.5*np.cos(pre[..., 6:7])+pre[..., 4:5] * 0.5*np.sin(pre[..., 6:7]))
    pre_ymax = np.maximum( pre[..., 1:2] - pre[..., 5:6] * 0.5*np.sin(pre[..., 6:7])-pre[..., 4:5] * 0.5*np.cos(pre[..., 6:7]),
              pre[..., 1:2] + pre[..., 5:6] * 0.5*np.sin(pre[..., 6:7])+pre[..., 4:5] * 0.5*np.cos(pre[..., 6:7]))
    pre_zmax = pre[..., 2:3] + pre[..., 3:4]

    label_xmin = np.minimum(label[..., 0:1] - label[..., 5:6] * 0.5*np.cos(label[..., 6:7])-label[..., 4:5] * 0.5*np.sin(label[..., 6:7]),
              label[..., 0:1] + label[..., 5:6] * 0.5*np.cos(label[..., 6:7])+label[..., 4:5] * 0.5*np.sin(label[..., 6:7]))
    label_ymin = np.minimum(label[..., 1:2] - label[..., 5:6] * 0.5*np.sin(label[..., 6:7])-label[..., 4:5] * 0.5*np.cos(label[..., 6:7]),
              label[..., 1:2] + label[..., 5:6] * 0.5*np.sin(label[..., 6:7])+label[..., 4:5] * 0.5*np.cos(label[..., 6:7]))
    label_zmin = label[..., 2:3]
    label_xmax = np.maximum(label[..., 0:1] - label[..., 5:6] * 0.5*np.cos(label[..., 6:7])-label[..., 4:5] * 0.5*np.sin(label[..., 6:7]),
              label[..., 0:1] + label[..., 5:6] * 0.5*np.cos(label[..., 6:7])+label[..., 4:5] * 0.5*np.sin(label[..., 6:7]))
    label_ymax = np.maximum(label[..., 1:2] - label[..., 5:6] * 0.5*np.sin(label[..., 6:7])-label[..., 4:5] * 0.5*np.cos(label[..., 6:7]),
              label[..., 1:2] + label[..., 5:6] * 0.5*np.sin(label[..., 6:7])+label[..., 4:5] * 0.5*np.cos(label[..., 6:7]))
    label_zmax = label[..., 2:3] + label[..., 3:4]
    
    bboxes1_area = pre[..., 4] * pre[..., 5]
    bboxes1_volume = pre[..., 4] * pre[..., 5] * pre[..., 3]
    bboxes2_area = label[..., 4] * label[..., 5]
    bboxes2_volume = label[..., 4] * label[..., 5] * label[..., 3]

    pre_min_xyz = np.concatenate([pre_xmin,pre_ymin,pre_zmin],axis=-1)
    pre_max_xyz = np.concatenate([pre_xmax,pre_ymax,pre_zmax],axis=-1)
    label_min_xyz = np.concatenate([label_xmin,label_ymin,label_zmin],axis=-1)
    label_max_xyz = np.concatenate([label_xmax,label_ymax,label_zmax],axis=-1)

    left_up = np.maximum(pre_min_xyz, label_min_xyz)
    right_down = np.minimum(pre_max_xyz, label_max_xyz)
    inter_section = np.maximum(right_down - left_up, 0.0)
    
    
    if mode=='iou':
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = bboxes1_area + bboxes2_area - inter_area
        iou = safe_divide(inter_area, union_area)
        return iou    
    if mode=='giou':
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = bboxes1_area + bboxes2_area - inter_area
        iou = safe_divide(inter_area, union_area)
        enclose_left_up = np.minimum(pre_min_xyz, label_min_xyz)
        enclose_right_down = np.maximum(pre_max_xyz, label_max_xyz)
        enclose_section = enclose_right_down - enclose_left_up
        enclose_area = enclose_section[..., 0] * enclose_section[..., 1]
        giou = iou - safe_divide(enclose_area - union_area, enclose_area)
        return giou
    if mode =='giou3d':
        inter_volume = inter_section[..., 0] * inter_section[..., 1] * inter_section[..., 2]
        union_volume = bboxes1_volume + bboxes2_volume - inter_volume        
        iou = safe_divide(inter_volume, union_volume)
        enclose_left_up = np.minimum(pre_min_xyz, label_min_xyz)
        enclose_right_down = np.maximum(pre_max_xyz, label_max_xyz)
        enclose_section = enclose_right_down - enclose_left_up
        enclose_volume = enclose_section[..., 0] * enclose_section[..., 1] * enclose_section[..., 2]
        giou_3d = iou - safe_divide(enclose_volume - union_volume, enclose_volume)
        return giou_3d
    if mode =='iou3d':
        inter_volume = inter_section[..., 0] * inter_section[..., 1] * inter_section[..., 2]
        union_volume = bboxes1_volume + bboxes2_volume - inter_volume        
        iou = safe_divide(inter_volume, union_volume)
        return iou        
    return iou

def lidar_to_camera_point(points, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T
    
    
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(cfg.MATRIX_T_VELO_2_CAM)
    
    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(cfg.MATRIX_R_RECT_0)

    points = np.matmul(T_VELO_2_CAM, points)
    points = np.matmul(R_RECT_0, points).T
    points = points[:, 0:3]
    return points.reshape(-1, 3)

def lidar_box3d_to_camera_box(boxes3d, cal_projection=False, P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 4)/(N, 8, 2)  x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
    num = boxes3d.shape[0]
    boxes2d = np.zeros((num, 4), dtype=np.int32)
    projections = np.zeros((num, 8, 2), dtype=np.float32)
    lidar_boxes3d_corner = box3Dcorner_from_gtbox(boxes3d)
    if type(P2) == type(None):
        P2 = np.array(cfg.MATRIX_P2)

    for n in range(num):
        box3d = lidar_boxes3d_corner[n]
        box3d = lidar_to_camera_point(box3d, T_VELO_2_CAM, R_RECT_0)
        points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
        points = np.matmul(P2, points).T
        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]

        projections[n] = points[:, 0:2]
        minx = int(np.min(points[:, 0]))
        maxx = int(np.max(points[:, 0]))
        miny = int(np.min(points[:, 1]))
        maxy = int(np.max(points[:, 1]))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d

def load_calib(calib_dir):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    lines = [ line.split()[1:] for line in lines ][:-1]
    #
    P = np.array(lines[2]).reshape(3,4)
    P = np.concatenate( (  P, np.array( [[0,0,0,0]] )  ), 0  )
    #
    Tr_velo_to_cam = np.array(lines[5]).reshape(3,4)
    Tr_velo_to_cam = np.concatenate(  [ Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)  ]  , 0     )
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3,:3] = np.array(lines[4][:9]).reshape(3,3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect    

def draw_lidar_box3d_on_image(img, boxes3d, scores, cls,gt_boxes3d=np.array([]),gt_boxes3d_notcare=np.array([]),
                              color=(0, 255, 255), gt_color=(255, 0, 255),gt_notcare_color=(100, 50, 100),  thickness=1, P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    value_to_class = ['Car','Truck','Pedestrian', 'Cyclist']
    img = img.copy()
    projections = lidar_box3d_to_camera_box(boxes3d, cal_projection=True, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    gt_projections = lidar_box3d_to_camera_box(gt_boxes3d, cal_projection=True, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    gt_projections_notcare = lidar_box3d_to_camera_box(gt_boxes3d_notcare, cal_projection=True, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    box_index=0
    # draw projections
    for qs in projections:
        score_tmp=scores[box_index]
        #color=(int(cls[box_index]*40),255,255)
        lable_txt=value_to_class[int(cls[box_index])]+':'+str(round(score_tmp,3))
        cv2.putText(img, lable_txt, (qs[0, 0], qs[1, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, 2*thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, 2*thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, 2*thickness, cv2.LINE_AA)
        box_index += 1
    # draw gt projections
    for qs in gt_projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)
    for qs in gt_projections_notcare:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_notcare_color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_notcare_color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_notcare_color, thickness, cv2.LINE_AA)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
