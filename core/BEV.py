import tensorflow as tf
import numpy as np
import math
#公用网络
def conv3d(input,cout,k,s,p= 'same',training=True,activation=True,name='conv3d'):
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.conv3d(
                input, cout, k, strides=s, padding = p, reuse=tf.AUTO_REUSE, name=scope)
        temp_conv = tf.layers.batch_normalization(
            temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        if activation:
            return tf.nn.relu(temp_conv)
        else:
            return temp_conv

def conv2d(input,cout,k,s,p= 'same',training=True,activation=True,name='conv2d'):
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.conv2d(
                input, cout, k, strides=s, padding = p, reuse=tf.AUTO_REUSE, name=scope)
        temp_conv = tf.layers.batch_normalization(
            temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        if activation:
            return tf.nn.relu(temp_conv)
        else:
            return temp_conv

def dense(input,cout,training=True,activation=True,name='dense'):
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.dense(input,cout,activation=None,use_bias=True,trainable=training,reuse=tf.AUTO_REUSE)
        if activation:
            return tf.nn.relu(temp_conv)
        else:
            return temp_conv

def tensor2points(tensor, offset=(0., -40., -3.), voxel_size=(0.05, 0.05, 0.1)):
    offset = tf.constant(np.array(offset),dtype=tf.float32)
    voxel_size = tf.constant(np.array(voxel_size),dtype=tf.float32)

    N,D,H,W,C = tensor.shape
    N_indices = tf.tile(tf.range(N)[:,tf.newaxis,tf.newaxis,tf.newaxis],[1,D,H,W])
    D_indices = tf.tile(tf.range(D)[tf.newaxis,:,tf.newaxis,tf.newaxis],[N,1,H,W])
    H_indices = tf.tile(tf.range(H)[tf.newaxis,tf.newaxis,:,tf.newaxis],[N,D,1,W])
    W_indices = tf.tile(tf.range(W)[tf.newaxis,tf.newaxis,tf.newaxis,:],[N,D,H,1])

    N_indices = tf.cast(N_indices,tf.float32)
    WHD_indices = tf.stack([W_indices,H_indices,D_indices],axis=-1)
    WHD_indices = tf.cast(WHD_indices,tf.float32)
    WHD_indices = WHD_indices*voxel_size+offset+0.5*voxel_size
    indices = tf.concat([N_indices[...,tf.newaxis],WHD_indices],axis=-1)
    tensor_reshape = tf.reshape(tensor,[-1,C])
    indices = tf.reshape(indices,[-1,4])
    return tensor_reshape,indices

def get_nearest_feature(points_mean,indices,tensor_reshape):
    points_mean_expand = tf.tile(points_mean[:,tf.newaxis,:],[1,indices.shape[0],1])
    dist_sqr = tf.reduce_sum(tf.pow(points_mean_expand - indices,2),axis=-1)
    min_arg = tf.argmin(dist_sqr,axis=-1)
    extract = tf.gather(tensor_reshape, min_arg,axis=0)
    return extract
#骨干网络
def BEV_backbone(input,):
    
    middle = list()
    temp_conv = conv2d(input,16, 3, (1, 1),'same', name='conv0_0')
    temp_conv = conv2d(temp_conv,16, 3, (1, 1),'same', name='conv0_1')
    
    temp_conv = conv2d(temp_conv,32, 3, (2, 2),'same', name='conv1_0')
    temp_conv = conv2d(temp_conv,32, 3, (1, 1),'same', name='conv1_1')
    temp_conv = conv2d(temp_conv,32, 3, (1, 1),'same', name='conv1_2')
    middle.append(temp_conv)
    temp_conv = conv2d(temp_conv,64, 3, (2, 2),'same', name='conv2_0')
    temp_conv = conv2d(temp_conv,64, 3, (1, 1),'same', name='conv2_1')
    temp_conv = conv2d(temp_conv,64, 3, (1, 1),'same', name='conv2_2')
    temp_conv = conv2d(temp_conv,64, 3, (1, 1),'same', name='conv2_3')
    middle.append(temp_conv)
    temp_conv = conv2d(temp_conv,64, 3, (2, 2),'same', name='conv3_0')
    temp_conv = conv2d(temp_conv,64, 3, (1, 1),'same', name='conv3_1')
    temp_conv = conv2d(temp_conv,64, 3, (1, 1),'same', name='conv3_2')
    temp_conv = conv2d(temp_conv,64, 3, (1, 1),'same', name='conv3_3')
    middle.append(temp_conv)
    temp_conv = conv2d(temp_conv,64, 1, (1, 1),'same', name='conv4_0')
    
    #BEV
    temp_conv = conv2d(temp_conv,256, 3, (1, 1),'same', name='conv5_0')
    temp_conv = conv2d(temp_conv,256, 3, (1, 1),'same', name='conv5_1')
    temp_conv = conv2d(temp_conv,256, 3, (1, 1),'same', name='conv5_2')
    temp_conv = conv2d(temp_conv,256, 3, (1, 1),'same', name='conv5_3')
    temp_conv = conv2d(temp_conv,256, 3, (1, 1),'same', name='conv5_4')
    temp_conv = conv2d(temp_conv,256, 3, (1, 1),'same', name='conv5_5')
    temp_conv = conv2d(temp_conv,256, 3, (1, 1),'same', name='conv5_6')
    conv6 = temp_conv
    temp_conv = conv2d(temp_conv,256, 1, (1, 1),'same', name='conv6_0')
    
    return temp_conv, conv6, middle

def BEV_head(temp_conv,num_class = 4):
    cls_preds = conv2d(temp_conv,2*num_class,1,1,'same',name = 'cls')
    box_preds = conv2d(temp_conv,2*6,1,1,'same',name = 'box') #x,y,h,w,l,r
    conv_dir_cls = conv2d(temp_conv,2*num_class,1,1,'same',name = 'dir')

def aux_head(conv6, box, window_size=(4, 7), grid_offsets=(0, 0), spatial_scale=1.):
    N = box.shape[0]
    win = window_size[0] * window_size[1]
    xg, yg, wg, lg, rg = tf.split(box, [1,1,1,1,1], axis=-1)
    xg = tf.tile(xg[...,tf.newaxis],[1,window_size[0],window_size[1]])
    yg = tf.tile(yg[...,tf.newaxis],[1,window_size[0],window_size[1]])
    rg = tf.tile(rg[...,tf.newaxis],[1,window_size[0],window_size[1]])
    cosTheta = tf.math.cos(rg)
    sinTheta = tf.math.sin(rg)
    xx = tf.linspace(-0.5,0.5,window_size[0])*wg
    xx = tf.tile(xx[:,:,tf.newaxis],[1,1,window_size[1]])
    yy = tf.linspace(-0.5,0.5,window_size[1])*lg
    yy = tf.tile(yy[:,tf.newaxis,:],[1,window_size[0],1])
    x=(xx * cosTheta + yy * sinTheta + xg)
    y=(yy * cosTheta - xx * sinTheta + yg)
    x = (x+grid_offsets[0])*spatial_scale
    y = (y+grid_offsets[1])*spatial_scale
    x = tf.reshape(x,[-1,win])
    y = tf.reshape(y,[-1,win])
    im = tf.squeeze(conv6,axis=0)
    samples_xy = tf.cast(tf.stack([x,y],axis=-1),dtype=tf.int32)#可能是yx
    pick_pixel = tf.gather_nd(im,samples_xy)
    conf_mean = tf.reduce_mean(pick_pixel,axis=1)
       
