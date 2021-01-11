import os
import shutil
import tensorflow as tf
from core.yolov4_new import *
from core.auxiliary_new import AuxNetwork
from core.config import cfg
import numpy as np
import math
from utils.utils_track import *
from core.dataset_kitti import Dataset,Dataset_valid, Dataset_predict
import core.common as common
import xlwt
#######构建网络######
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
#######构建网络######
#######参数设置######
batch_size = 1
save_model_dir = os.path.join('./save_model', 'default')
log_dir = os.path.join('./log', 'default')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)
#训练集，预测集，验证集定义
train_dataset = Dataset(batch_size, shuffle=True, aug=True)
predict_dataset = Dataset_predict(batch_size, shuffle=False, aug=False,vis=False)
valid_dataset = Dataset_valid(batch_size, shuffle=False, aug=False)

steps_per_predict = len(predict_dataset)
steps_per_epoch = len(train_dataset)
first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS #2
second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS #10
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64) #1
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch #1个epoch
total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1,allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True,)
sess= tf.Session(config=config)

if tf.train.get_checkpoint_state(save_model_dir):
    print("Reading model parameters from %s" % save_model_dir)
    saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
    epoch_value = sess.run(epoch) + 1
else:
    print("Created model with fresh parameters.")
    sess.run(tf.global_variables_initializer())
    epoch_value = 0

global_counter = epoch_value * steps_per_epoch
workbook=xlwt.Workbook(encoding='utf-8')
booksheet=workbook.add_sheet('Sheet',cell_overwrite_ok=True)

summary_interval = 5
summary_val_interval = 10
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
#######参数设置######
#######开始训练######
for i in range(100):
    train_num=0
    for image_data, target in train_dataset:
        bev_f = image_data
        (targets_value,
         pos_equal_one_value,
         pos_equal_one_sum_value,
         pos_equal_one_for_reg_value,
         neg_equal_one_value,
         neg_equal_one_sum_value,
         cls_onehot_value,
         point_value,
         point_cls_value,
         point_in_box_gt_value,
         point_in_box_weight_gt_value,
         box_gt_value) = target
        input_dict = {
                      targets:targets_value,
                      input_layer:bev_f,
                      pos_equal_one:pos_equal_one_value,
                      pos_equal_one_sum:pos_equal_one_sum_value,
                      pos_equal_one_for_reg:pos_equal_one_for_reg_value,
                      neg_equal_one:neg_equal_one_value,
                      neg_equal_one_sum:neg_equal_one_sum_value,
                      cls_onehot:cls_onehot_value,
                      point_gt_placeholder:point_value,
                      point_cls_gt_placeholder:point_cls_value,
                      point_in_box_gt_placeholder:point_in_box_gt_value,
                      point_in_box_weight_gt_placeholder:point_in_box_weight_gt_value,
                      box_gt_placeholder:box_gt_value
                     }
        if global_counter % 100 ==0:
            valid_flag = True
        else:
            valid_flag = False
        if global_counter % 100 ==0:
            summary_flag = True
        else:
            summary_flag = False
        if summary_flag:
            _,loss_value,child_loss_value,train_summary_value=sess.run([opt,loss,child_loss,train_summary],input_dict)
            summary_writer.add_summary(train_summary_value, global_counter)
            print('train_sumary at {} global_counter'.format(global_counter))
        else:
            _,loss_value,child_loss_value=sess.run([opt,loss,child_loss],input_dict)
        if valid_flag:
            image_data, target = valid_dataset.get_data()
            bev_f = image_data
            (targets_value,
             pos_equal_one_value,
             pos_equal_one_sum_value,
             pos_equal_one_for_reg_value,
             neg_equal_one_value,
             neg_equal_one_sum_value,
             cls_onehot_value,
             point_value,
             point_cls_value,
             point_in_box_gt_value,
             point_in_box_weight_gt_value,
             box_gt_value) = target
            input_dict = {
                          targets:targets_value,
                          input_layer:bev_f,
                          pos_equal_one:pos_equal_one_value,
                          pos_equal_one_sum:pos_equal_one_sum_value,
                          pos_equal_one_for_reg:pos_equal_one_for_reg_value,
                          neg_equal_one:neg_equal_one_value,
                          neg_equal_one_sum:neg_equal_one_sum_value,
                          cls_onehot:cls_onehot_value,
                          point_gt_placeholder:point_value,
                          point_cls_gt_placeholder:point_cls_value,
                          point_in_box_gt_placeholder:point_in_box_gt_value,
                          point_in_box_weight_gt_placeholder:point_in_box_weight_gt_value,
                          box_gt_placeholder:box_gt_value
                         }
            valid_summary_value=sess.run(valid_summary,input_dict)
            summary_writer.add_summary(valid_summary_value, global_counter)
            print('valid_sumary at {} global_counter'.format(global_counter))
        print('=>STEP {}/{}, epoch={}, loss={}'.format(
        train_num,steps_per_epoch,epoch_value,loss_value))
        train_num += 1
        global_counter += 1
    _,epoch_value = sess.run([epoch_add_op,epoch])
    saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=epoch_value)
    print('save model at {}'.format(epoch_value))

    #验证集评估#######
    predict_num = 0 
    reg_TP_all=np.zeros(cfg.NUM_CLASS)
    reg_TR_all=np.zeros(cfg.NUM_CLASS)
    reg_TP_strict_all=np.zeros(cfg.NUM_CLASS)
    reg_TR_strict_all=np.zeros(cfg.NUM_CLASS)
    cls_TP_all=np.zeros(cfg.NUM_CLASS)
    cls_TR_all=np.zeros(cfg.NUM_CLASS)
    pre_box_num_all=np.zeros(cfg.NUM_CLASS)
    gt_box_num_all=np.zeros(cfg.NUM_CLASS)
    for ret in predict_dataset:
        (tag, 
        raw_lidar, 
        bev_f,
        gtbox3d_with_ID,
        gtbox3d_with_ID_notcare,
        img) = ret
        if gtbox3d_with_ID.shape[0] ==0:
            continue
        input_dict = {input_layer:bev_f[np.newaxis,...]}        
        result_value = sess.run(result,input_dict)
        [delta_output,prob_output,cls_output] = result_value
        ret_box3d_score,criteria_set,vis_set = predict(sess,prob_output,delta_output,cls_output,gtbox3d_with_ID[np.newaxis,...],gtbox3d_with_ID_notcare[np.newaxis,...],raw_lidar,bev_f,img,evaluation=True,vis=False)
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
        print('=>PREDICT_STEP {}/{}'.format(predict_num,steps_per_predict))
        predict_num += 1
    reg_precision = reg_TP_all/np.maximum(1,pre_box_num_all)
    reg_recall = reg_TR_all/np.maximum(1,gt_box_num_all)
    reg_strict_precision = reg_TP_strict_all/np.maximum(1,pre_box_num_all)
    reg_strict_recall = reg_TR_strict_all/np.maximum(1,gt_box_num_all)
    cls_precision = cls_TP_all/np.maximum(1,pre_box_num_all)
    cls_recall = cls_TR_all/np.maximum(1,gt_box_num_all)
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
    epoch_value = int(epoch_value)
    booksheet.write(epoch_value,0,float(reg_precision_final))
    booksheet.write(epoch_value,1,float(reg_recall_final))
    booksheet.write(epoch_value,2,float(reg_strict_precision_final))
    booksheet.write(epoch_value,3,float(reg_strict_recall_final))
    booksheet.write(epoch_value,4,float(cls_precision_final))
    booksheet.write(epoch_value,5,float(cls_recall_final)) 
    booksheet.write(epoch_value,6,float(pre_box_num_sum))
    booksheet.write(epoch_value,7,float(gt_box_num_sum))
    workbook.save('./log/pre.xls')
    print('评估完成，epoch={}'.format(epoch_value))
    #验证集评估结束#######
    
    

#######开始训练######