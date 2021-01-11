import tensorflow as tf
from core.config import cfg
import numpy as np

def tensor2points(tensor, offset=(-20.8, 0), voxel_size=(0.2, 0.2)):
    offset = tf.constant(np.array(offset),dtype=tf.float32)
    voxel_size = tf.constant(np.array(voxel_size),dtype=tf.float32)
    _,H,W,C = tensor.shape
    H_indices = tf.tile(tf.range(H)[:,tf.newaxis],[1,W])
    W_indices = tf.tile(tf.range(W)[tf.newaxis,:],[H,1])

    HW_indices = tf.stack([H_indices,W_indices],axis=-1)
    HW_indices = tf.cast(HW_indices,tf.float32)
    HW_indices = HW_indices*voxel_size+offset+0.5*voxel_size
    tensor_reshape = tf.reshape(tensor,[-1,C])
    indices = tf.reshape(HW_indices,[-1,2])
    return tensor_reshape,indices

def get_nearest_feature(points_mean,indices,tensor_reshape):
    points_mean,_ = tf.split(points_mean,[2,1],axis=1)  #取yx
    points_mean_expand = tf.tile(points_mean[:,tf.newaxis,:],[1,indices.shape[0],1])
    dist_sqr = tf.reduce_sum(tf.pow(points_mean_expand - indices,2),axis=-1)
    min_arg = tf.argmin(dist_sqr,axis=-1)
    extract = tf.gather(tensor_reshape, min_arg,axis=0)
    return extract

def AuxNetwork(middle, point_keep, point_in_box, point_in_box_weight, cls_num=cfg.NUM_CLASS):
    vx_f,vx_nxy=tensor2points(middle, offset=(cfg.Y_MIN,cfg.X_MIN), voxel_size=(2*cfg.VOXEL_Y_SIZE, 2*cfg.VOXEL_X_SIZE))
    p = get_nearest_feature(point_keep,vx_nxy,vx_f)
    
    point_in_box_reshape = tf.reshape(point_in_box,[-1,3])
    p2 = get_nearest_feature(point_in_box_reshape,vx_nxy,vx_f)
    
    
    #p = tf.concat([p0,p1],axis=-1)
    pointwise = tf.layers.dense(p,256,activation=None,use_bias=False,trainable=True,reuse=tf.AUTO_REUSE,name='dense0_0')
    pointwise = tf.keras.layers.BatchNormalization()(pointwise)
    pointwise = tf.nn.relu(pointwise)
    point_cls = tf.layers.dense(pointwise,cls_num+1,activation=None,use_bias=False,trainable=True,reuse=tf.AUTO_REUSE,name ='dense1_0')
    point_cls = tf.nn.softmax(point_cls,axis=-1)
    
    boxwise = tf.reshape(p2,[-1,cfg.PMAX_PER_BOX,p2.shape[-1]])
    boxwise = tf.layers.dense(boxwise,256,activation=None,use_bias=False,trainable=True,reuse=tf.AUTO_REUSE,name='dense2_0')
    boxwise = tf.keras.layers.BatchNormalization()(boxwise)
    boxwise = tf.layers.dense(boxwise,128,activation=None,use_bias=False,trainable=True,reuse=tf.AUTO_REUSE,name='dense3_0')
    boxwise = tf.keras.layers.BatchNormalization()(boxwise)
    boxwise_max = tf.reduce_max(boxwise,axis=1)
    boxwise = boxwise*point_in_box_weight
    boxwise = tf.reduce_sum(boxwise,axis=1)/tf.reduce_sum(point_in_box_weight,axis=1)
    boxwise = tf.concat([boxwise,boxwise_max],axis=-1)
    boxwise = tf.layers.dense(boxwise,256,activation=None,use_bias=False,trainable=True,reuse=tf.AUTO_REUSE,name='dense4_0')
    boxwise = tf.keras.layers.BatchNormalization()(boxwise)
    boxwise = tf.nn.relu(boxwise)
    boxreg = tf.layers.dense(boxwise,8,activation=None,use_bias=False,trainable=True,reuse=tf.AUTO_REUSE,name ='dense5_0')
    
    return point_cls,boxreg