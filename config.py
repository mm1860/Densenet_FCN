import os
import os.path as osp
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.RNG_SEED = 3

# /////////////////////////////////////////////////////////////////
__C.IMG = edict()

__C.IMG.CHANNEL = 1

# /////////////////////////////////////////////////////////////////
__C.TRAIN = edict()

# batch size for training
__C.TRAIN.BS = 1


# /////////////////////////////////////////////////////////////////
__C.MODEL = edict()

# bias decay or not
__C.MODEL.BIAS_DECAY = False

# weight decay
__C.MODEL.WEIGHT_DECAY = 0.001

# output channels of the first convolution
__C.MODEL.INIT_CHANNELS = 24

# use bias or not
__C.MODEL.USE_BIAS = False