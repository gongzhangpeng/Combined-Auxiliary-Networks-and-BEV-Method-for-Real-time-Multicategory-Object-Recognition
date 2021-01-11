import numpy as np
import tensorflow as tf
from core.config import cfg

def cal_anchors_tensor():
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
    anchors_all = tf.constant(anchors_all,dtype=tf.float32)
    return anchors_all

anchors = cal_anchors_tensor()

def generate_box_tough(conf_preds, box_preds,threshold):
    N,H,W,C = conf_preds.shape
    conf_max_arg = tf.argmax(conf_preds,axis=-1)
    conf_max = tf.reduce_max(conf_preds,axis=-1)
    
    conf_max_reshape = tf.squeeze(conf_max,axis=0)
    conf_max_arg_reshape = tf.squeeze(conf_max_arg,axis=0)

    box_preds_reshape = tf.squeeze(box_preds,axis=0)
    h_index= tf.tile(tf.range(H)[:,tf.newaxis],[1,W])
    w_index= tf.tile(tf.range(W)[tf.newaxis,:],[H,1])

    index_anchor = tf.stack([h_index,w_index,conf_max_arg_reshape],axis=-1)
    anchors_pick = tf.gather_nd(anchors,index_anchor)
    box_preds_pick = tf.gather_nd(box_preds_reshape,index_box)
    x,y,z,h,w,l,r =  tf.split(box_preds_pick,[1,1,1,1,1,1,1],axis=-1)
    #encode
    x_a,y_a,z_a,h_a,w_a,l_a,r_a = tf.split(anchors_pick,[1,1,1,1,1,1,1],axis=-1)
    d_a = tf.math.sqrt(tf.pow(w_a,2)+tf.pow(l_a,2))
    x_value = x_a+x*d_a
    y_value = y_a+y*d_a
    z_value = z_a+z*h_a
    h_value = tf.math.exp(h)*h_a
    w_value = tf.math.exp(w)*w_a*1.05
    l_value = tf.math.exp(l)*l_a*1.05
    r_value = r_a+r
    bbox_value = tf.concat([x_value,
                            y_value,
                            z_value,
                            h_value,
                            w_value,
                            l_value,
                            r_value],axis=-1)
    bbox_pad_value = tf.concat([x_value,
                                y_value,
                                w_value,
                                l_value,
                                r_value],axis=-1)
    mask = tf.where(conf_max_reshape>threshold)
    bbox_value_pick = tf.gather_nd(bbox_value,mask)
    bbox_pad_value_pick = tf.gather_nd(bbox_pad_value,mask)
    conf_max_pick = tf.gather_nd(conf_max_reshape,mask)
    class_index = tf.gather_nd(class_index,mask)
    w_a_pick = tf.gather_nd(w_a,mask)
    l_a_pick = tf.gather_nd(l_a,mask)
    #生成box区域score
    window_size=(4, 8)
    grid_offsets=(cfg.X_MIN, cfg.Y_MIN)
    win = window_size[0] * window_size[1]
    spatial_scale=2

    xg, yg, wg, lg, rg = tf.split(bbox_pad_value_pick, [1,1,1,1,1], axis=-1)
    xg = tf.tile(xg[...,tf.newaxis],[1,window_size[0],window_size[1]])
    yg = tf.tile(yg[...,tf.newaxis],[1,window_size[0],window_size[1]])
    rg = tf.tile(rg[...,tf.newaxis],[1,window_size[0],window_size[1]])
    cosTheta = tf.math.cos(rg)
    sinTheta = tf.math.sin(rg)
    area = tf.squeeze(wg*lg,axis=-1)#tf.squeeze((wg/w_a_pick)*(lg/l_a_pick),axis=-1)
    xx = tf.linspace(-0.8,0.8,window_size[0])*wg
    xx = tf.tile(xx[:,:,tf.newaxis],[1,1,window_size[1]])
    yy = tf.linspace(-0.8,0.8,window_size[1])*lg
    yy = tf.tile(yy[:,tf.newaxis,:],[1,window_size[0],1])
    x=(xx * cosTheta + yy * sinTheta + xg)
    y=(yy * cosTheta - xx * sinTheta + yg)
    x = (x-grid_offsets[0])/(cfg.VOXEL_X_SIZE*spatial_scale)
    y = (y-grid_offsets[1])/(cfg.VOXEL_Y_SIZE*spatial_scale)
    x = tf.reshape(x,[-1,win])
    y = tf.reshape(y,[-1,win])
    samples_yx = tf.cast(tf.stack([y,x],axis=-1),dtype=tf.int32)#可能是yx
    pick_pixel = tf.gather_nd(conf_max_reshape,samples_yx)
    conf_mean = tf.reduce_mean(pick_pixel,axis=1)*area+conf_max_pick
    return bbox_pad_value_pick, bbox_value_pick,conf_mean,class_index#conf_max_pick换conf_mean

