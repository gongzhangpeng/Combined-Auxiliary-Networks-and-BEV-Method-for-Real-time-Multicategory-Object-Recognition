import tensorflow as tf
import numpy as np
from core.config import cfg
import core.common as common

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    
def voxel_to_bev(coordinate,voxelwise,voxelmask,bev):
    input_voxel = tf.scatter_nd(
                coordinate, voxelwise, [1, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH,cfg.INPUT_DEEP, 4])
    input_voxel0,voxelmask0 = common.sparse_conv3d(input_voxel,voxelmask,32,3,(1,1,1))
    input_voxel0= tf.contrib.layers.group_norm(input_voxel0,groups=input_voxel0.shape[-1], channels_axis=-1, reduction_axes=[-4,-3, -2])
    input_voxel0 = tf.nn.relu(input_voxel0)
    input_voxel0,voxelmask0 = common.sparse_conv3d(input_voxel0,voxelmask0,32,3,(2,2,2))
    input_voxel0= tf.contrib.layers.group_norm(input_voxel0,groups=input_voxel0.shape[-1], channels_axis=-1, reduction_axes=[-4,-3, -2])
    input_voxel0 = tf.nn.relu(input_voxel0)
    input_voxel0,voxelmask0 = common.sparse_conv3d(input_voxel0,voxelmask0,64,3,(1,1,1))
    input_voxel0= tf.contrib.layers.group_norm(input_voxel0,groups=input_voxel0.shape[-1], channels_axis=-1, reduction_axes=[-4,-3, -2])
    input_voxel0 = tf.nn.relu(input_voxel0)
    input_voxel0,voxelmask0 = common.sparse_conv3d(input_voxel0,voxelmask0,64,3,(1,1,2))
    input_voxel0= tf.contrib.layers.group_norm(input_voxel0,groups=input_voxel0.shape[-1], channels_axis=-1, reduction_axes=[-4,-3, -2])
    input_voxel0 = tf.nn.relu(input_voxel0)
    input_voxel0,voxelmask0 = common.sparse_conv3d(input_voxel0,voxelmask0,64,3,(1,1,1))
    input_voxel0= tf.contrib.layers.group_norm(input_voxel0,groups=input_voxel0.shape[-1], channels_axis=-1, reduction_axes=[-4,-3, -2])
    input_voxel0 = tf.nn.relu(input_voxel0)
    input_voxel0,voxelmask0 = common.sparse_conv3d(input_voxel0,voxelmask0,64,3,(1,1,2))
    input_voxel0= tf.contrib.layers.group_norm(input_voxel0,groups=input_voxel0.shape[-1], channels_axis=-1, reduction_axes=[-4,-3, -2])
    input_voxel0 = tf.nn.relu(input_voxel0)
    input_voxel0,voxelmask0 = common.sparse_conv3d(input_voxel0,voxelmask0,128,3,(1,1,1))
    input_voxel0= tf.contrib.layers.group_norm(input_voxel0,groups=input_voxel0.shape[-1], channels_axis=-1, reduction_axes=[-4,-3, -2])
    input_voxel0 = tf.nn.relu(input_voxel0)
    input_voxel0,voxelmask0 = common.sparse_conv3d(input_voxel0,voxelmask0,128,3,(1,1,2))
    input_voxel1 = tf.squeeze(input_voxel0,axis=3)
    voxelmask1 = tf.squeeze(voxelmask0,axis=3)
    input_voxel1= tf.contrib.layers.group_norm(input_voxel1,groups=input_voxel1.shape[-1], channels_axis=-1, reduction_axes=[-3, -2])
    input_voxel1 = tf.nn.relu(input_voxel1)

    input_voxel1 = tf.concat([input_voxel1,bev],axis=-1)
    input_voxel1 = common.convolutional(input_voxel1,(3, 3, 256, 256))
    #input_voxel1,voxelmask1 = common.sparse_conv2d(input_voxel1,voxelmask1,256,3,(1,1))
    #input_voxel1= tf.contrib.layers.group_norm(input_voxel1,groups=input_voxel1.shape[-1], channels_axis=-1, reduction_axes=[-3, -2])
    #input_voxel1 = mish(input_voxel1)
    input_voxel1 = common.convolutional(input_voxel1,(3, 3, 256, 256))
    input_voxel2 = common.convolutional(input_voxel1,(3, 3, 256, 512))
    input_voxel2 = common.convolutional(input_voxel2, (1, 1, 512, 16 + cfg.NUM_CLASS ), activate=False, bn=False)
    middle=[bev,input_voxel1]
    
    return input_voxel2,middle