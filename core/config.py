#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict

use_kitti=True
__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

######voxelnet
__C.NUM_CLASS = 4
__C.DETECT_OBJ = 'Car'  # Pedestrian/Cyclist
__C.Y_MIN = -20.8
__C.Y_MAX = 20.8
__C.X_MIN = 0
__C.X_MAX = 41.6
__C.Z_MIN = -2
__C.Z_MAX = 2
__C.VOXEL_X_SIZE = 0.2
__C.VOXEL_Y_SIZE = 0.2
__C.VOXEL_Z_SIZE = 4
__C.Z_GROUND = -1.78
if use_kitti:
    __C.H_VALUE = [1.52,2.64,1.75,1.73]    #
    __C.W_VALUE = [1.64,2.25,0.67,0.58]   #
    __C.L_VALUE = [3.86,8.08,0.86,1.78] #
else:
    __C.H_VALUE = [1.53, 3.53, 1.76, 1.73]#[1.40,2.50,1.24,1.22]    #
    __C.W_VALUE = [1.63, 2.54, 0.66, 0.60]#[1.62,2.18,0.43,0.64]   #
    __C.L_VALUE = [3.52,6.93,0.48,1.52]#[3.71,7.61,0.61,1.55] #

__C.ANGLE_REVOLUTION = 5 #单位°
__C.ANGLE_REVOLUTION_EQ = int(__C.ANGLE_REVOLUTION*3.1415926*100/180)
__C.ANGLE_DIVIDE_NUM = int(2*3.1415926*100/__C.ANGLE_REVOLUTION_EQ)


__C.VOXEL_POINT_COUNT = 60
__C.INPUT_WIDTH = int((__C.X_MAX - __C.X_MIN) / __C.VOXEL_X_SIZE)  #352
__C.INPUT_HEIGHT = int((__C.Y_MAX - __C.Y_MIN) / __C.VOXEL_Y_SIZE)  #400
__C.INPUT_DEEP = int((__C.Z_MAX - __C.Z_MIN) / __C.VOXEL_Z_SIZE)  #10
__C.FEATURE_RATIO = 2
__C.FEATURE_WIDTH = int(__C.INPUT_WIDTH / __C.FEATURE_RATIO)  #176
__C.FEATURE_HEIGHT = int(__C.INPUT_HEIGHT / __C.FEATURE_RATIO)  #200

__C.RPN_NMS_POST_TOPK = 20
__C.RPN_NMS_THRESH = 0.1
__C.RPN_SCORE_THRESH = 0.998
__C.RPN_POS_IOU = 0.6
__C.RPN_NEG_IOU = 0.3
__C.BV_LOG_FACTOR = 4
######voxelnet_end
__C.YOLO.CLASSES              = "./data/classes/coco.names"
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/val2017.txt"
__C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 0#2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 0
__C.TRAIN.SECOND_STAGE_EPOCHS   = 5



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/val2017.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5

__C.MATRIX_P2 = ([[719.787081,    0.,            608.463003, 44.9538775],
                      [0.,            719.787081,    174.545111, 0.1066855],
                      [0.,            0.,            1.,         3.0106472e-03],
                      [0.,            0.,            0.,         0]])

# cal mean from train set
__C.MATRIX_T_VELO_2_CAM = ([
    [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
    [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
    [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
    [0, 0, 0, 1]
])
# cal mean from train set
__C.MATRIX_R_RECT_0 = ([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]
])
__C.IMAGE_WIDTH = 1242#1920#
__C.IMAGE_HEIGHT = 375#1280#
__C.PMAX_PER_BOX = 50