import os
import os.path as osp
from easydict import EasyDict as edict
import platform

__C = edict()

cfg = __C

__C.RNG_SEED = 3

# source code directory
__C.SRC_DIR = osp.join(osp.dirname(__file__))

# default output directory
__C.OUTPUT_DIR = "output"

# default tensorboard directory
__C.TB_DIR = "tensorboard"

# default logging directory
__C.LOG_DIR = "logs"

# tag of model directory
__C.TAG = "default"

# model prefix
__C.PREFIX = "default"

# /////////////////////////////////////////////////////////////////
__C.IMG = edict()

# image width and height
__C.IMG.WIDTH = 512
__C.IMG.HEIGHT = 512

# image channels
__C.IMG.CHANNEL = 1

# /////////////////////////////////////////////////////////////////
__C.DATA = edict()

# platform compatible
if "Windows" in platform.system():
    __C.DATA.ROOT_DIR = "C:\\DataSet\\LiverQL"
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

# /////////////////////////////////////////////////////////////////
__C.TRAIN = edict()

# batch size for training
__C.TRAIN.BS = 64

# learning rate
__C.TRAIN.LR = 0.01

# learning rate decay rate
__C.TRAIN.LR_DECAY = 0.1

# max iterations
__C.TRAIN.MAX_ITERS = 150000

# momentum
__C.TRAIN.MOMENTUM = 0.9

# learning rate step point
__C.TRAIN.LR_STEP = [100000]

# snapshot iters
__C.TRAIN.SNAPSHOT_ITERS = 5000

# snapshot_kept
__C.TRAIN.SNAPSHOT_KETP = 3

# dispaly step
__C.TRAIN.DISPLAY = 200

# summary interval(seconds)
__C.TRAIN.SUMMARY_INTERVAL = 60

# keep probability
__C.TRAIN.KEEP_PROB = 0.7

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
