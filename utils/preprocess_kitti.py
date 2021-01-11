#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : preprocess.py
# Purpose :
# Creation Date : 10-12-2017
# Last Modified : Thu 18 Jan 2018 05:34:42 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import os
import numpy as np
import sys
sys.path.append('../')
from core.config import cfg
from numba import jit

data_dir = 'velodyne'
Height = cfg.INPUT_HEIGHT
Width = cfg.INPUT_WIDTH 
d_x=np.tile(np.arange(Width)[np.newaxis,:],[Height,1])
d_y1=np.tile(np.arange(Height/2)[:,np.newaxis],[1,Width])
d_y2=np.tile((Height/2-np.arange(Height/2))[:,np.newaxis],[1,Width])
d_y=np.concatenate([d_y2,d_y1],axis=0)
d=np.sqrt(d_x**2+d_y**2)
d_k=np.sqrt(d)/10+1
PMAX_PER_BOX = cfg.PMAX_PER_BOX
HEIGHT_HALF = int(cfg.INPUT_HEIGHT/2)
@jit(nopython=True)
def makeBVFeature_addpoint_jit(points,voxel_size,coors_range,grid_size,map_rgb,points_yxz):
    max_voxels=25000
    max_points=35
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        map_rgb[coor[1], coor[0], 4] += 1
        d = (1+np.sqrt(coor[1]**2+(coor[0]-HEIGHT_HALF)**2)/10)
        map_rgb[coor[1], coor[0], 0] = np.minimum(np.log(map_rgb[coor[1], coor[0], 4]*d+1)/np.log(64),1)
        if points[i,2] > map_rgb[coor[0], coor[1], 1]:
            map_rgb[coor[1], coor[0], 1] = points[i,2]
            map_rgb[coor[1], coor[0], 3] = points[i,3]
            points_yxz[coor[1], coor[0], 0] = points[i,1]
            points_yxz[coor[1], coor[0], 1] = points[i,0]
            points_yxz[coor[1], coor[0], 2] = points[i,2]
        if points[i,2] < map_rgb[coor[0], coor[1], 2]:
            map_rgb[coor[1], coor[0], 2] = points[i,2]
            
    return map_rgb,points_yxz

voxel_size = np.array([cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE], dtype=np.float32)
coors_range = np.array([cfg.X_MIN,cfg.Y_MIN,cfg.Z_MIN,cfg.X_MAX,cfg.Y_MAX,cfg.Z_MAX], dtype=np.float32)    
coor = np.zeros(shape=(3, ), dtype=np.int32)
grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
Height = cfg.INPUT_HEIGHT
Width = cfg.INPUT_WIDTH

def makeBVFeature_addpoint(points):
    map_rgb = np.zeros([Height,Width,5],dtype=np.float32)
    map_rgb[:,:,1]=np.ones([Height,Width],dtype=np.float32)*(cfg.Z_MIN)
    map_rgb[:,:,2]=np.ones([Height,Width],dtype=np.float32)*(cfg.Z_MAX)
    points_yxz = np.zeros([Height,Width,3],dtype=np.float32)
    bef, points_keep = makeBVFeature_addpoint_jit(points,voxel_size,
                                                    coors_range,
                                                    grid_size,
                                                    map_rgb,
                                                    points_yxz)
    bef[:,:,1] = bef[:,:,1] - cfg.Z_MIN
    bef[:,:,2] = bef[:,:,2] - cfg.Z_MIN
    return bef[:,:,0:4],points_keep

@jit(nopython=True)
def makeBVFeature_addpoint_addvoxel_jit(points,voxel_size,coors_range,grid_size,map_rgb,points_yxz,
                                                    num_points_per_voxel,
                                                    coor_to_voxelidx,
                                                    voxels,
                                                    coors,
                                                    voxel_mask):
    max_voxels=25000
    max_points=35
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        map_rgb[coor[1], coor[0], 3] += 1
        d = (1+np.sqrt(coor[1]**2+(coor[0]-HEIGHT_HALF)**2)/10)
        map_rgb[coor[1], coor[0], 0] = np.log(map_rgb[coor[1], coor[0], 3]*d+1)/np.log(64)
        if points[i,2] > map_rgb[coor[0], coor[1], 1]:
            map_rgb[coor[1], coor[0], 1] = points[i,2]
            map_rgb[coor[1], coor[0], 2] = points[i,3]
            points_yxz[coor[1], coor[0], 0] = points[i,1]
            points_yxz[coor[1], coor[0], 1] = points[i,0]
            points_yxz[coor[1], coor[0], 2] = points[i,2]
        #增加体素部分
        
        voxelidx = coor_to_voxelidx[coor[1],coor[0], coor[2]]
        if voxelidx == -1:
            voxel_mask[coor[1], coor[0], coor[2]] =1
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[1], coor[0], coor[2]] = voxelidx
            coors[voxelidx,1:] = coor[1],coor[0],coor[2]
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1

    return map_rgb,points_yxz,voxel_num,coors,voxels,num_points_per_voxel,voxel_mask

voxel_size = np.array([cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE], dtype=np.float32)
coors_range = np.array([cfg.X_MIN,cfg.Y_MIN,cfg.Z_MIN,cfg.X_MAX,cfg.Y_MAX,cfg.Z_MAX], dtype=np.float32)    
coor = np.zeros(shape=(3, ), dtype=np.int32)
grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
Height = cfg.INPUT_HEIGHT
Width = cfg.INPUT_WIDTH

def makeBVFeature_addpoint_addvoexl(points,get_points=False):
    #增加体素部分
    max_voxels=25000
    max_points=35
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=grid_size, dtype=np.int32)
    voxel_mask = np.zeros(shape=grid_size, dtype=np.float32)
    voxels = np.zeros(shape=(max_voxels, max_points, points.shape[-1]), dtype=np.float32)
    coors = np.ones(shape=(max_voxels, 4), dtype=np.int32)
    
    map_rgb = np.zeros([Height,Width,4],dtype=np.float32)
    map_rgb[:,:,1]=np.ones([Height,Width],dtype=np.float32)*(cfg.Z_MIN)
    points_yxz = np.zeros([Height,Width,3],dtype=np.float32)
    bef, points_keep,voxel_num,coors,voxels,num_points_per_voxel,voxel_mask = makeBVFeature_addpoint_addvoxel_jit(points,voxel_size,
                                                    coors_range,
                                                    grid_size,
                                                    map_rgb,
                                                    points_yxz,
                                                    num_points_per_voxel,
                                                    coor_to_voxelidx,
                                                    voxels,
                                                    coors,
                                                    voxel_mask)
    coors = coors[:voxel_num,:]
    voxels = voxels[:voxel_num,:,:]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    
    voxels_mean = voxels[:, :, :4].sum(axis=1)/num_points_per_voxel.reshape(-1, 1)
    voxels_mean[:,0:3] = voxels_mean[:,0:3]-(coors[:,1:]*grid_size+coors_range[[2,1,3]])
    voxel_mask = voxel_mask[np.newaxis,...][...,np.newaxis]
    return bef[:,:,0:3],points_keep,coors,voxels_mean,voxel_mask
    
def get_points_gtbox(point_keep,gtbox3d_with_ID):
    box3d = np.float32(gtbox3d_with_ID[:,2:9])
    box_cls = np.int32(gtbox3d_with_ID[:,1])
    mask1= point_keep[:,:,0]==0
    mask2= point_keep[:,:,1]==0
    mask3= point_keep[:,:,2]==0
    mask = mask1*mask2*mask3
    point_nozeros = point_keep[np.where(mask==0)]
    box3d_num = box3d.shape[0]
    point_num_max=point_nozeros.shape[0]
    indices = list(range(point_num_max))
    np.random.shuffle(indices)
    indices=np.array(indices,dtype=np.int32)
    point = np.zeros([2001,3],dtype=np.float32)
    point_cls = np.zeros([2001,5],dtype=np.float32)
    
    pmax_per_box = PMAX_PER_BOX
    point_in_box = np.zeros([box3d_num,pmax_per_box,3],dtype=np.float32)
    point_in_box_weight = np.zeros([box3d_num,pmax_per_box,1],dtype=np.float32)
    point_in_box[:,0,:]=box3d[:,0:3]
    point_in_box_weight[:,0,0]=1
    point_index = get_points_gtbox_jit(point_nozeros,box3d,box_cls,indices,point_num_max,box3d_num,point,point_cls,point_in_box,point_in_box_weight)
    point=point[0:point_index,:]
    point_cls=point_cls[0:point_index,:]
    #打乱顺序
    point_num = point.shape[0]
    indices = list(range(point_num))
    np.random.shuffle(indices)
    point_suffle = point[indices,:]
    point_cls_suffle = point_cls[indices,:]

    
    return point_suffle,point_cls_suffle,point_in_box,point_in_box_weight

    
@jit(nopython=True)
def get_points_gtbox_jit(point_nozeros,box3d,box_cls,indices,point_num_max,box3d_num,point,point_cls,point_in_box,point_in_box_weight):
    zeros_point_num = 0
    nozeros_point_num = 0
    point_index=0
    pmax_per_box = point_in_box.shape[1]
    box_index = np.ones(box3d_num,dtype=np.int32)
    for i in range(point_num_max):
        m = indices[i]
        p_y,p_x,p_z = point_nozeros[m,:]
        asign_flag=False
        for j in range(box3d_num):
            x,y,z,h,w,l,r = box3d[j,:]
            cls_index = box_cls[j]
            x0=-np.cos(r)*l/2+np.sin(r)*w/2+x
            y0= np.sin(r)*l/2+np.cos(r)*w/2+y
            x1=-np.cos(r)*l/2-np.sin(r)*w/2+x
            y1= np.sin(r)*l/2-np.cos(r)*w/2+y
            x2= np.cos(r)*l/2-np.sin(r)*w/2+x
            y2=-np.sin(r)*l/2-np.cos(r)*w/2+y
            x3= np.cos(r)*l/2+np.sin(r)*w/2+x
            y3=-np.sin(r)*l/2+np.cos(r)*w/2+y
            a = (x1-x0)*(p_y-y0)-(y1-y0)*(p_x-x0)
            b = (x2-x1)*(p_y-y1)-(y2-y1)*(p_x-x1)
            c = (x3-x2)*(p_y-y2)-(y3-y2)*(p_x-x2)
            d = (x0-x3)*(p_y-y3)-(y0-y3)*(p_x-x3)
            if (a>0 and b>0 and c>0 and d>0 )or(a<0 and b<0 and c<0 and d<0 ):
                if p_z>z and p_z<z+h:
                    if box_index[j]<pmax_per_box:
                        point_in_box[j,box_index[j],:] = p_y,p_x,p_z
                        point_in_box_weight[j,box_index[j],0] = 1/(1+np.sqrt((p_x-x)**2+(p_y-y)**2))
                        box_index[j] +=1
                    if nozeros_point_num<1000:
                        point[point_index,:]=p_y,p_x,p_z
                        point_cls[point_index,cls_index]=1
                        point_index+=1
                        nozeros_point_num+=1
                        asign_flag = True
                        break
            elif asign_flag == False:
                if zeros_point_num < 250:
                    point[point_index,:]=p_y,p_x,p_z
                    point_cls[point_index,4]=1
                    point_index+=1
                    zeros_point_num+=1
                    asign_flag = True
    return point_index
    
    
def get_points_gt(point_keep,gtbox3d_with_ID):
    box3d = np.float32(gtbox3d_with_ID[:,2:9])
    box_cls = np.int32(gtbox3d_with_ID[:,1])
    mask1= point_keep[:,:,0]==0
    mask2= point_keep[:,:,1]==0
    mask3= point_keep[:,:,2]==0
    mask = mask1*mask2*mask3
    point_nozeros = point_keep[np.where(mask==0)]
    box3d_num = box3d.shape[0]
    point_num_max=point_nozeros.shape[0]
    indices = list(range(point_num_max))
    np.random.shuffle(indices)
    indices=np.array(indices,dtype=np.int32)
    point = np.zeros([1251,3],dtype=np.float32)
    point_cls = np.zeros([1251,5],dtype=np.float32)
    point_reg = np.zeros([1251,5],dtype=np.float32)
    point_inbox_mask = np.zeros([1251],dtype=np.float32)
    point_index = get_points_gt_jit(point_nozeros,box3d,box_cls,indices,point_num_max,box3d_num,point,point_cls,point_reg,point_inbox_mask)
    point=point[0:point_index,:]
    point_cls=point_cls[0:point_index,:]
    point_reg=point_reg[0:point_index,:]
    point_inbox_mask=point_inbox_mask[0:point_index]
    #打乱顺序
    point_num = point.shape[0]
    indices = list(range(point_num))
    np.random.shuffle(indices)
    point_suffle = point[indices,:]
    point_cls_suffle = point_cls[indices,:]
    point_reg_suffle = point_reg[indices,:]
    point_inbox_mask_suffle = point_inbox_mask[indices]
    
    return point_suffle,point_cls_suffle,point_reg_suffle,point_inbox_mask_suffle

    
@jit(nopython=True)
def get_points_gt_jit(point_nozeros,box3d,box_cls,indices,point_num_max,box3d_num,point,point_cls,point_reg,point_inbox_mask):
    anchors_h = np.array([1.52,2.64,1.75,1.73] ,dtype=np.float32)#kitti
    anchors_w = np.array([1.64,2.25,0.67,0.58],dtype=np.float32)
    anchors_l = np.array([3.86,8.08,0.86,1.78],dtype=np.float32)
    anchors_d = np.sqrt(anchors_w**2 +anchors_l**2)
    zeros_point_num = 0
    nozeros_point_num = 0
    point_index=0
    for i in range(point_num_max):
        m = indices[i]
        p_y,p_x,p_z = point_nozeros[m,:]
        asign_flag=False
        for j in range(box3d_num):
            x,y,z,h,w,l,r = box3d[j,:]
            cls_index = box_cls[j]
            x0=-np.cos(r)*l/2+np.sin(r)*w/2+x
            y0= np.sin(r)*l/2+np.cos(r)*w/2+y
            x1=-np.cos(r)*l/2-np.sin(r)*w/2+x
            y1= np.sin(r)*l/2-np.cos(r)*w/2+y
            x2= np.cos(r)*l/2-np.sin(r)*w/2+x
            y2=-np.sin(r)*l/2-np.cos(r)*w/2+y
            x3= np.cos(r)*l/2+np.sin(r)*w/2+x
            y3=-np.sin(r)*l/2+np.cos(r)*w/2+y
            a = (x1-x0)*(p_y-y0)-(y1-y0)*(p_x-x0)
            b = (x2-x1)*(p_y-y1)-(y2-y1)*(p_x-x1)
            c = (x3-x2)*(p_y-y2)-(y3-y2)*(p_x-x2)
            d = (x0-x3)*(p_y-y3)-(y0-y3)*(p_x-x3)
            if (a>0 and b>0 and c>0 and d>0 )or(a<0 and b<0 and c<0 and d<0 ):
                if nozeros_point_num<1000 and p_z>z and p_z<z+h:
                    point[point_index,:]=p_y,p_x,p_z
                    point_cls[point_index,cls_index]=1
                    point_reg[point_index,:]=(p_x-x)/anchors_d[cls_index],(p_y-y)/anchors_d[cls_index],np.log(w/anchors_w[cls_index]),np.log(l/anchors_l[cls_index]),r
                    point_inbox_mask[point_index]=1#/((np.sqrt((p_x-x)**2+(p_y-y)**2)+0.05)/anchors_d[cls_index])
                    point_index+=1
                    nozeros_point_num+=1
                    asign_flag = True
                    break
            elif asign_flag == False:
                if zeros_point_num < 1000:
                    point[point_index,:]=p_y,p_x,p_z
                    point_cls[point_index,4]=1
                    point_reg[point_index,:]=1,1,0,0,0
                    point_inbox_mask[point_index]=0
                    point_index+=1
                    zeros_point_num+=1
                    asign_flag = True
    return point_index

def process_pointcloud(point_cloud, cls=cfg.DETECT_OBJ):
    # Input:
    #   (N, 4)
    # Output:
    #   voxel_dict
    if cls == 'Car':
        scene_size = np.array([4, 80, 70.4], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 400, 352], dtype=np.int64)
        lidar_coord = np.array([0, 40, 3], dtype=np.float32)
        max_point_number = 35
    else:
        scene_size = np.array([4, 40, 48], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 200, 240], dtype=np.int64)
        lidar_coord = np.array([0, 20, 3], dtype=np.float32)
        max_point_number = 45

        np.random.shuffle(point_cloud)

    shifted_coord = point_cloud[:, :3] + lidar_coord
    # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
    voxel_index = np.floor(
        shifted_coord[:, ::-1] / voxel_size).astype(np.int)

    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    point_cloud = point_cloud[bound_box]
    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)

    K = len(coordinate_buffer)
    T = max_point_number

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64)

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 7), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis=1, keepdims=True)/number_buffer.reshape(K, 1, 1)

    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}
    return voxel_dict

def voxelize(point_cloud):
    voxel_size = np.array([0.2, 0.2, 0.4], dtype=point_cloud.dtype)
    coors_range = np.array([0,-40,-3,70.4,40,1], dtype=point_cloud.dtype)
    
    
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    max_voxels = 20000
    max_points =cfg.VOXEL_POINT_COUNT
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(shape=(max_voxels, max_points, point_cloud.shape[-1]), dtype=point_cloud.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    
    voxel_num,coors,voxels,num_points_per_voxel = _points_to_voxel_reverse_kernel(
            point_cloud, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)
    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    
    voxels_mean =  voxels[:, :, :3] - \
        voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    
    feature_buffer=np.concatenate([voxels,voxels_mean],axis=-1)
    
    voxel_dict = {'feature_buffer': feature_buffer,
                      'coordinate_buffer': coors,
                      'number_buffer': num_points_per_voxel}
    return voxel_dict

@jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num,coors,voxels,num_points_per_voxel

@jit(nopython=True)
def makeBVFeature_jit(points,voxel_size,coors_range,grid_size,map_rgb,max_voxels = 25000):       
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        map_rgb[coor[1], coor[0], 0] += 1
        if points[i,2] > map_rgb[coor[0], coor[1], 1]:
            map_rgb[coor[1], coor[0], 1] = points[i,2]
            map_rgb[coor[1], coor[0], 2] = points[i,3]            
    return map_rgb


def makeBVFeature_new(points):
    voxel_size = np.array([cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE], dtype=np.float32)
    coors_range = np.array([cfg.X_MIN,cfg.Y_MIN,cfg.Z_MIN,cfg.X_MAX,cfg.Y_MAX,cfg.Z_MAX], dtype=np.float32)    
    coor = np.zeros(shape=(3, ), dtype=np.int32)    
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    Height = cfg.INPUT_HEIGHT
    Width = cfg.INPUT_WIDTH  
    map_rgb = np.zeros([Height,Width,3],dtype=np.float32)
    map_rgb[:,:,1]=np.ones([Height,Width],dtype=np.float32)*cfg.Z_MIN
    bef=makeBVFeature_jit(points,voxel_size,coors_range,grid_size,map_rgb,20000)
    bef[:,:,0] = np.minimum(1.0, np.log(bef[:,:,0]*d_k + 1) / np.log(64))
    return bef


def makeBVFeature(raw_lidar):
    minX = cfg.X_MIN
    maxX = cfg.X_MAX
    minY = cfg.Y_MIN 
    maxY = cfg.Y_MAX
    minZ = cfg.Z_MIN
    maxZ = cfg.Z_MAX
    Discretization_x = cfg.VOXEL_X_SIZE
    Discretization_y = cfg.VOXEL_X_SIZE
    PointCloud_mask= raw_lidar
    mask = np.where(
            (PointCloud_mask[:, 0] > minX) & 
            (PointCloud_mask[:, 0] < maxX) & 
            (PointCloud_mask[:, 1] > minY) & 
            (PointCloud_mask[:, 1] < maxY) & 
            (PointCloud_mask[:, 2] > minZ) & 
            (PointCloud_mask[:, 2] < maxZ)
            )
    PointCloud_mask = PointCloud_mask[mask]

    Height = int((maxY - minY) / Discretization_y) 
    Width = int((maxX - minX) / Discretization_x) 

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_mask)
    PointCloud[:,0] = np.int_(np.floor(PointCloud[:,0] / Discretization_x))
    PointCloud[:,1] = np.int_(np.floor(PointCloud[:,1] / Discretization_y) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:,2], PointCloud[:,1], PointCloud[:,0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:2], axis = 0, return_index = True, return_counts = True)
    PointCloud_frac = PointCloud[indices]

    # Some important problem is image coordinate is (y,x), not (x,y)
    heightMap[np.int_(PointCloud_frac[:, 1]), np.int_(PointCloud_frac[:, 0])] = PointCloud_frac[:, 2]#得到z最高值

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    PointCloud_top = PointCloud_frac
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 1]), np.int_(PointCloud_top[:, 0])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 1]), np.int_(PointCloud_top[:, 0])] = normalizedCounts
    RGB_Map = np.zeros((Height,Width, 3))

    # RGB channels respectively
    RGB_Map[:,:,0] = densityMap
    RGB_Map[:,:,1] = heightMap
    RGB_Map[:,:,2] = intensityMap
    save = np.zeros((Height, Width, 3))
    save = RGB_Map[0:Height, 0:Width, :]
    return save
