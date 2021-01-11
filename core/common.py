#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
# import tensorflow_addons as tfa
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def sparse_conv3d(tensor,binary_mask = None,filters=32,kernel_size=3,strides=1,padding="same"):
    if binary_mask == None: 
        b,h,w,d,c = tensor.get_shape()
        channels=tf.split(tensor,c,axis=-1)
        binary_mask = tf.where(tf.equal(channels[0], 0), tf.zeros_like(channels[0]), tf.ones_like(channels[0]))
    features = tf.multiply(tensor,binary_mask)  
    features = tf.layers.conv3d(features, filters=filters, kernel_size=kernel_size, strides=strides, trainable=True, use_bias=False, padding=padding)  #对应与权重的卷积>>b
 
    norm = tf.layers.conv3d(binary_mask, filters=filters,kernel_size=kernel_size,strides=strides,kernel_initializer=tf.ones_initializer(),trainable=False,use_bias=False,padding=padding)
    norm = tf.where(tf.equal(norm,0),tf.zeros_like(norm),tf.reciprocal(norm)) 
    _,_,_,_,bias_size = norm.get_shape()

    b = tf.Variable(tf.constant(0.0, shape=[bias_size]),trainable=True)   
    feature = tf.multiply(features,norm)+b  
    mask = tf.layers.max_pooling3d(binary_mask,strides = strides,pool_size=kernel_size,padding=padding) 
    return feature,mask

def sparse_conv2d(tensor,binary_mask = None,filters=32,kernel_size=3,strides=1,padding="same"):
    if binary_mask == None: 
        b,h,w,c = tensor.get_shape()
        channels=tf.split(tensor,c,axis=-1)
        binary_mask = tf.where(tf.equal(channels[0], 0), tf.zeros_like(channels[0]), tf.ones_like(channels[0]))
    features = tf.multiply(tensor,binary_mask)  
    features = tf.layers.conv2d(features, filters=filters, kernel_size=kernel_size, strides=strides, trainable=True, use_bias=False, padding=padding)  #对应与权重的卷积>>b
 
    norm = tf.layers.conv2d(binary_mask, filters=filters,kernel_size=kernel_size,strides=strides,kernel_initializer=tf.ones_initializer(),trainable=False,use_bias=False,padding=padding)
    norm = tf.where(tf.equal(norm,0),tf.zeros_like(norm),tf.reciprocal(norm))  
    _,_,_,bias_size = norm.get_shape()

    b = tf.Variable(tf.constant(0.0, shape=[bias_size]),trainable=True)   
    feature = tf.multiply(features,norm)+b  
    mask = tf.layers.max_pooling2d(binary_mask,strides = strides,pool_size=kernel_size,padding=padding) 
    return feature,mask
        
        
def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='relu'):#leaky
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: #conv = BatchNormalization()(conv)
        conv = tf.contrib.layers.group_norm(conv,groups=filters_shape[-1], channels_axis=-1, reduction_axes=[-3, -2])
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
        else:
            conv = tf.nn.relu(conv)
    return conv

def Deconv2D(input_layer, f, k, s, activate=True, bn=True, activate_type='mish'):
    conv = tf.keras.layers.Conv2DTranspose(
            filters=f, kernel_size= k, strides=s, padding='same',
             dilation_rate=(1, 1), activation=None, use_bias=False,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(0.0005)
        )(input_layer)
    if bn: 
        conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

# def block_tiny(input_layer, input_channel, filter_num1, activate_type='leaky'):
#     conv = convolutional(input_layer, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#     short_cut = input_layer
#     conv = convolutional(conv, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#
#     input_data = tf.concat([conv, short_cut], axis=-1)
#     return residual_output

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')

