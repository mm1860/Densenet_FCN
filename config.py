import os
import os.path as osp
from easydict import EasyDict as edict
import platform
from utils.logger import create_logger

__C = edict()

cfg = __C

__C.RNG_SEED = 3

# source code directory
__C.SRC_DIR = osp.join(osp.dirname(__file__))

# default output directory
__C.OUTPUT_DIR = "output"

# default tensorboard directory
__C.TB_DIR = "tensorboard"

# default prediction directory
__C.PRED_DIR = "prediction"

# default logging directory
__C.LOG_DIR = "logs"

# tag of model directory
__C.TAG = "default"

# model prefix
__C.PREFIX = "default"

# tag of prediction directory
__C.PRED_TAG = "default"

# we use only one logger in the whole project
__C.LOGGER = "MainLogger"

# network to use
__C.BACKBONE = "FC-Densenet"

# /////////////////////////////////////////////////////////////////
__C.IMG = edict()

# image width and height
# __C.IMG.WIDTH = 512
# __C.IMG.HEIGHT = 512

# image channels
__C.IMG.CHANNEL = 1

# medical image bit depth
__C.IMG.BIT = 16

# medical image window width and level
__C.IMG.W_WIDTH = 250
__C.IMG.W_LEVEL = 55

# /////////////////////////////////////////////////////////////////
__C.DATA = edict()

# platform compatible
if "Windows" in platform.system():
    __C.DATA.ROOT_DIR = "D:\\DataSet\\LiverQL"
elif "Linux" in platform.system():
    __C.DATA.ROOT_DIR = "/home/jarvis/DataSet/LiverQL"
else:
    raise SystemError("Not supported operating system!")

# specify default training dataset
__C.DATA.TRAINSET = "Liver_2016_train+Liver_2017_train"

# specify default validation dataset
__C.DATA.VALSET = "Liver_2017_test"

# specify default testing dataset
__C.DATA.TESTSET = "Liver_2017_test"

# specify default 3D testing dataset
__C.DATA.TESTSET_3D = "Liver_2017_test_3D"

# /////////////////////////////////////////////////////////////////
__C.TRAIN = edict()

# batch size for training
__C.TRAIN.BS = 64

# update options
__C.TRAIN.UPDATE_OPS = False

# dispaly step
__C.TRAIN.DISPLAY = 200

# keep probability used in encoder stage
# while in decoder stage we disable dropout
__C.TRAIN.KEEP_PROB = 0.7

# learning rate
__C.TRAIN.LR = 0.01

# learning rate decay rate
__C.TRAIN.LR_DECAY = 0.1

# learning rate step point
__C.TRAIN.LR_STEP = [100000]

# use probability map or binary map to compute metrics
__C.TRAIN.MAP = "prob"

# max iterations
__C.TRAIN.MAX_ITERS = 150000

# momentum used in mementum optimizer
__C.TRAIN.MOMENTUM = 0.9

# moving average for which variables: (deprecated)
#           none: disable moving average
#            all: apply to all the trainable variables
#         weight: only apply to weights
#           loss: only apply to loss
#         metric: only apply to metrics
#   A+B or A+B+C: apply to A and B
# all is equal to weight+loss+metric
__C.TRAIN.MAVG = "loss+metric"

# moving average decay rate (deprecated)
__C.TRAIN.MAVG_RATE = 0.999

# snapshot iters
__C.TRAIN.SNAPSHOT_ITERS = 5000

# snapshot_kept
__C.TRAIN.SNAPSHOT_KEPT = 3

# summary interval(seconds)
__C.TRAIN.SUMMARY_INTERVAL = 60

# /////////////////////////////////////////////////////////////////
__C.VAL = edict()

# validate on val set during training or not
__C.VAL.NEED = False

# validation internal (seconds)
__C.VAL.STEPS = 3000

# validation number
__C.VAL.NUMBER = 1

# output prediction map type:
#   probability(prob)
#   binary(bin)
__C.VAL.MAP = "prob"

# /////////////////////////////////////////////////////////////////
__C.TEST = edict()

# batch size for test 2D image
__C.TEST.BS_2D = 64

# batch size for test 3D image
# Warning: different images have variance depths, set this value to
# 1 is the best choice
__C.TEST.BS_3D = 1

# test mode, use which iteration of models
__C.TEST.ITER = __C.TRAIN.MAX_ITERS

# /////////////////////////////////////////////////////////////////
# model settings for FC-Densenet
__C.MODEL = edict()

# activation function:
#   relu
#   prelu
#   leaky_relu
__C.MODEL.ACTIVATION = "relu"

# bias decay or not
__C.MODEL.BIAS_DECAY = False

# weight decay
__C.MODEL.WEIGHT_DECAY = 1e-3

# cross entropy
__C.MODEL.CROSS_ENTROPY = 10.0

# output channels of the first convolutional layer
__C.MODEL.INIT_CHANNELS = 24

# number of dense blocks
__C.MODEL.BLOCKS = 3

# number of layers per block
__C.MODEL.NUM_LAYERS_PER_BLOCK = [12]

# growth rat
__C.MODEL.GROWTH_RATE = 12

# compression
__C.MODEL.THETA = 0.5

# use bias or not
__C.MODEL.USE_BIAS = False

# threshold of final segmentation
__C.MODEL.THRESHOLD = 0.5

# weight initializer
# * trunc_norm
# * rand_norm
# * xavier
__C.MODEL.WEIGHT_INITIALIZER = "trunc_norm"

# //////////////////////////////////////////////////////////////////
__C.UNET = edict()

# initial channels
__C.UNET.INIT_CHANNELS = 64

# number of down samples
__C.UNET.NUM_DOWN_SAMPLE = 4

# number of conv per layer
__C.UNET.NUM_CONV_PER_LAYER = [2]

# //////////////////////////////////////////////////////////////////
__C.UDN = edict()

# compression
__C.UDN.THETA = 1.0

# use dropout or not
__C.UDN.USE_DROPOUT = False

# initial channels
__C.UDN.INIT_CHANNELS = 48

# number of blocks
__C.UDN.NUM_BLOCKS = 4

# number of layers per block
__C.UDN.NUM_LAYERS_PER_BLOCK = [4]

# growth rate
__C.UDN.GROWTH_RATE = 16

# //////////////////////////////////////////////////////////////////
__C.OPTIMIZER = edict()

# optimizer method
__C.OPTIMIZER.METHOD = "adam"

# adam parameters
__C.OPTIMIZER.ADAM = edict()
__C.OPTIMIZER.ADAM.ARGS = {"beta1": 0.9, "beta2": 0.99}

# ///////////////////////////////////////////////////////////////////

def merge_cfg(old_cfg, new_cfg):
    if type(old_cfg) is not edict:
        return

    for k, v in new_cfg.items():
        if k not in old_cfg:
            raise KeyError("{} is not a valid config key".format(k))

        # check both types
        old_type = type(old_cfg[k])
        if old_type is not type(v):
            raise ValueError("Type mismatch ({} vs. {}) for config key: {}".format(
                            type(old_cfg[k]), type(v), k))
        
        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_cfg(old_cfg[k], new_cfg[k])
            except:
                raise ValueError("Error under config key: {}".format(k))
        else:
            old_cfg[k] = v

def update_cfg(filename):
    """ Load a config file and cover default value.
    """
    if not osp.exists(filename):
        raise FileNotFoundError("Can't find file: {}".format(filename))

    import yaml
    with open(filename, "r") as f:
        file_cfg = edict(yaml.load(f))

    merge_cfg(__C, file_cfg)

def global_logger(name, propagate=False):
    logger = create_logger(file_=False, console=False, propagate=propagate, name=name)
    return logger