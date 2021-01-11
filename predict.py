import os
import shutil
import tensorflow as tf
from core.yolov4_new import *
from core.auxiliary_new import AuxNetwork
from core.config import cfg
import numpy as np
import math
from utils.preprocess_kitti import makeBVFeature_new
from utils.utils_track import *
from core.dataset_kitti import Dataset,Dataset_predict
import core.common as common
import time
import xlwt
#######构建网络######
cal_ap = True
evaluation = True
vis = False
if cal_ap:
    epoch_num = 37
else:
    epoch_num = 1
input_layer = tf.placeholder(dtype=tf.float32,shape=[None,cfg.INPUT_WIDTH,cfg.INPUT_HEIGHT,4])
targets = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 16])
pos_equal_one = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2])
pos_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])
pos_equal_one_for_reg = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 16])
neg_equal_one = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2])
neg_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])
cls_onehot=tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, cfg.NUM_CLASS])

point_gt_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,3],name='point_gt_placeholder')
point_cls_gt_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,cfg.NUM_CLASS+1],name='point_cls_gt_placeholder')
point_in_box_gt_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,cfg.PMAX_PER_BOX,3],name='point_in_box_gt_placeholder')
point_in_box_weight_gt_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,cfg.PMAX_PER_BOX,1],name='point_in_box_weight_gt_placeholder')
box_gt_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,8],name='box_gt_placeholder')

lable=[targets,pos_equal_one,pos_equal_one_sum,pos_equal_one_for_reg,neg_equal_one,neg_equal_one_sum,cls_onehot,
        point_cls_gt_placeholder,box_gt_placeholder]

boxes2d = tf.placeholder(tf.float32, [None, 4])
boxes2d_scores = tf.placeholder(tf.float32, [None])

conv_bbox,conv_branch =YOLOv4_tiny(input_layer)
point_cls,box_reg = AuxNetwork(conv_branch,point_gt_placeholder,point_in_box_gt_placeholder,point_in_box_weight_gt_placeholder,cls_num=cfg.NUM_CLASS)

loss, result, child_loss, train_summary = compute_loss_and_result(conv_bbox,point_cls,box_reg,lable)
loss_valid, result_valid, child_loss_valid, valid_summary = compute_loss_and_result(conv_bbox,point_cls,box_reg,lable,False)
box2d_ind_after_nms = caculate_nms(boxes2d,boxes2d_scores)
learning_rate = tf.Variable(0.001, trainable=False, dtype=tf.float32)
epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
epoch_add_op = epoch.assign(epoch + 1)

boundaries = [75, 90]
values = [ learning_rate, learning_rate * 0.1, learning_rate * 0.01 ]
lr = tf.train.piecewise_constant(epoch, boundaries, values)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#使用tf.layers.batch_normalization必须加
with tf.control_dependencies(update_ops):
    opt = tf.train.AdamOptimizer(lr).minimize(loss)
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
        max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
batch_size = 1
save_model_dir = os.path.join('./save_model', 'default')
output_path='./prediction'
workbook=xlwt.Workbook(encoding='utf-8')
booksheet=workbook.add_sheet('Sheet',cell_overwrite_ok=True)

predict_dataset = Dataset_predict(batch_size, shuffle=False, aug=False,vis=vis)
steps_per_epoch = len(predict_dataset)
first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS 
second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS 
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64) 
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch 
total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1,allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True,)
sess= tf.Session(config=config)

if tf.train.get_checkpoint_state(save_model_dir):
    print("Reading model parameters from %s" % save_model_dir)
    saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
    start_epoch = sess.run(epoch) + 1

else:
    print("Created model with fresh parameters.")
    sess.run(tf.global_variables_initializer())
    
epoch_value = 0

time_preprocess_start = time.time()
for epoch in range(epoch_num):
    reg_TP_all=np.zeros(cfg.NUM_CLASS)
    reg_TR_all=np.zeros(cfg.NUM_CLASS)
    reg_TP_strict_all=np.zeros(cfg.NUM_CLASS)
    reg_TR_strict_all=np.zeros(cfg.NUM_CLASS)
    cls_TP_all=np.zeros(cfg.NUM_CLASS)
    cls_TR_all=np.zeros(cfg.NUM_CLASS)
    pre_box_num_all=np.zeros(cfg.NUM_CLASS)
    gt_box_num_all=np.zeros(cfg.NUM_CLASS)
    preprocess_time_set = []
    forward_time_set = []
    train_num=0
    for ret in predict_dataset:
        if cal_ap:
            if epoch <6:
                rpn_thresh = 0.1*epoch+0.1
            elif epoch <24:
                rpn_thresh = 0.01*(epoch-6)+0.8
            else:
                rpn_thresh = 0.99+0.001*(epoch-26)
        else:
            rpn_thresh = cfg.RPN_SCORE_THRESH
        (tag, 
        raw_lidar, 
        bev_f,
        gtbox3d_with_ID,
        gtbox3d_with_ID_notcare,
        img) = ret
        time_preprocess_end = time.time()
        time_preprocess_dur = time_preprocess_end - time_preprocess_start
        
        if gtbox3d_with_ID.shape[0] ==0:
            continue
        time_forward_start = time.time()
        input_dict = {input_layer:bev_f[np.newaxis,...]}        
        result_value = sess.run(result,input_dict)
        [delta_output,prob_output,cls_output] = result_value
        ret_box3d_score,criteria_set,vis_set = predict(sess,prob_output,delta_output,cls_output,gtbox3d_with_ID[np.newaxis,...],gtbox3d_with_ID_notcare[np.newaxis,...],raw_lidar,bev_f,img,rpn_threshold=rpn_thresh,evaluation=evaluation,vis=vis)
        time_forward_end = time.time()
        time_forward_dur = time_forward_end - time_forward_start
        if evaluation:
            (reg_TP,
            reg_TR,
            reg_TP_strict,
            reg_TR_strict,
            cls_TP,
            cls_TR,
            pre_box_num,
            gt_box_num) = criteria_set
            reg_TP_all += reg_TP
            reg_TR_all += reg_TR
            reg_TP_strict_all += reg_TP_strict
            reg_TR_strict_all += reg_TR_strict
            cls_TP_all += cls_TP
            cls_TR_all += cls_TR
            pre_box_num_all += pre_box_num
            gt_box_num_all += gt_box_num
            if train_num>10:
                preprocess_time_set.append(time_preprocess_dur)
                forward_time_set.append(time_forward_dur)
        if vis:
            bird_view, heatmap,front_img = vis_set
            bird_view_path = os.path.join( output_path, 'vis', tag + '_bv.jpg'  )
            heatmap_path = os.path.join( output_path, 'vis', tag + '_heatmap.jpg'  )
            front_img_path = os.path.join( output_path, 'vis', tag + '_front.jpg'  )
            cv2.imwrite( bird_view_path, bird_view )
            cv2.imwrite( heatmap_path, heatmap )
            cv2.imwrite( front_img_path, front_img )
        print('=>PREDICT_STEP {}/{} epoch={}'.format(train_num,steps_per_epoch,epoch))
        train_num += 1
        time_preprocess_start = time.time()

    reg_precision = reg_TP_all/np.maximum(1,pre_box_num_all)
    reg_recall = reg_TR_all/np.maximum(1,gt_box_num_all)
    reg_strict_precision = reg_TP_strict_all/np.maximum(1,pre_box_num_all)
    reg_strict_recall = reg_TR_strict_all/np.maximum(1,gt_box_num_all)
    cls_precision = cls_TP_all/np.maximum(1,pre_box_num_all)
    cls_recall = cls_TR_all/np.maximum(1,gt_box_num_all)
    preprocess_time_mean = np.mean(preprocess_time_set)
    forward_time_mean = np.mean(forward_time_set)
    for i in range(cfg.NUM_CLASS):
        print('cls = {}; reg_precision = {}; reg_recall = {}; reg_strict_precision = {}; reg_strict_recall = {}; cls_precision = {}; cls_recall = {}'.format(
        i, reg_precision[i], reg_recall[i], reg_strict_precision[i], reg_strict_recall[i], cls_precision[i], cls_recall[i]))
        print('cls = {}; pre_box_num_all = {}; gt_box_num_all = {}'.format(i,pre_box_num_all[i], gt_box_num_all[i]))

    reg_TP_sum = np.sum(reg_TP_all)
    reg_TR_sum = np.sum(reg_TR_all)
    reg_TP_strict_sum = np.sum(reg_TP_strict_all)
    reg_TR_strict_sum = np.sum(reg_TR_strict_all)
    cls_TP_sum = np.sum(cls_TP_all)
    cls_TR_sum = np.sum(cls_TR_all)
    pre_box_num_sum = np.sum(pre_box_num_all)
    gt_box_num_sum = np.sum(gt_box_num_all)
    reg_precision_final =  reg_TP_sum/np.maximum(1,pre_box_num_sum)
    reg_recall_final = reg_TR_sum/np.maximum(1,gt_box_num_sum)
    reg_strict_precision_final = reg_TP_strict_sum/np.maximum(1,pre_box_num_sum)
    reg_strict_recall_final = reg_TR_strict_sum/np.maximum(1,gt_box_num_sum)
    cls_precision_final = cls_TP_sum/np.maximum(1,pre_box_num_sum)
    cls_recall_final = cls_TR_sum/np.maximum(1,gt_box_num_sum)
    print('reg_precision_final = {}; reg_recall_final = {}; reg_strict_precision_final = {}; reg_strict_recall_final = {}; cls_precision_final = {}; cls_recall_final = {}'.format(
        reg_precision_final, reg_recall_final, reg_strict_precision_final, reg_strict_recall_final, cls_precision_final, cls_recall_final))
    print('pre_box_num_final = {}; gt_box_num_final = {}'.format(pre_box_num_sum, gt_box_num_sum))
    print('preprocess_time = {}, forwar_time = {}, total_time = {}'.format(preprocess_time_mean, forward_time_mean, preprocess_time_mean + forward_time_mean))
    if cal_ap:
        for i in range(4):
                booksheet.write(epoch,i*6,float(reg_precision[i]))
                booksheet.write(epoch,i*6+1,float(reg_recall[i]))
                booksheet.write(epoch,i*6+2,float(reg_strict_precision[i]))
                booksheet.write(epoch,i*6+3,float(reg_strict_recall[i]))
                booksheet.write(epoch,i*6+4,float(cls_precision[i]))
                booksheet.write(epoch,i*6+5,float(cls_recall[i]))
        reg_precision_totall = np.sum(reg_TP_all)/np.maximum(1,np.sum(pre_box_num_all))
        reg_recall_totall = np.sum(reg_TR_all)/np.maximum(1,np.sum(gt_box_num_all))
        reg_strict_precision_final = np.sum(reg_TP_strict_all)/np.maximum(1,np.sum(pre_box_num_all))
        reg_strict_recall_final = np.sum(reg_TR_strict_all)/np.maximum(1,np.sum(gt_box_num_all))
        cls_precision_totall = np.sum(cls_TP_all)/np.maximum(1,np.sum(pre_box_num_all))
        cls_recall_totall = np.sum(cls_TR_all)/np.maximum(1,np.sum(gt_box_num_all))
        pre_box_num_sum = np.sum(pre_box_num_all)
        gt_box_num_sum = np.sum(gt_box_num_all)
        booksheet.write(epoch,24,float(reg_precision_totall))
        booksheet.write(epoch,25,float(reg_recall_totall))
        booksheet.write(epoch,26,float(reg_strict_precision_final))
        booksheet.write(epoch,27,float(reg_strict_recall_final))
        booksheet.write(epoch,28,float(cls_precision_totall))
        booksheet.write(epoch,29,float(cls_recall_totall))
        booksheet.write(epoch,30,float(pre_box_num_sum))
        booksheet.write(epoch,31,float(gt_box_num_sum))
        booksheet.write(epoch,32,float(rpn_thresh))
        workbook.save('./log/evaluate.xls')
