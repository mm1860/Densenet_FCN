import os
import os.path as osp
from easydict import EasyDict as edict
import platform

__C = edict()

cfg = __C

__C.RNG_SEED = 3

__C.SRC_DIR = osp.join(osp.dirname(__file__))

# /////////////////////////////////////////////////////////////////
__C.IMG = edict()

__C.IMG.CHANNEL = 1

# /////////////////////////////////////////////////////////////////
__C.DATA = edict()

if "Windows" in platform.system():
    __C.DATA.ROOT_DIR = "C:\\DataSet\\LiverQL"
elif "Linux" in platform.system():
    __C.DATA.ROOT_DIR = "/home/jarvis/DataSet/LiverQL"
else:
    raise SystemError("Not supported operating system!")

# /////////////////////////////////////////////////////////////////
__C.TRAIN = edict()

# batch size for training
__C.TRAIN.BS = 1

# learning rate
__C.TRAIN.LR = 0.01

# learning rate decay rate
__C.TRAIN.LR_DECAY = 0.1

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
__C.MODEL = edict()

# bias decay or not
__C.MODEL.BIAS_DECAY = False

# weight decay
__C.MODEL.WEIGHT_DECAY = 0.001

# output channels of the first convolution
__C.MODEL.INIT_CHANNELS = 24

# number of dense blocks
__C.MODEL.BLOCKS = 3

# number of layers per block
__C.MODEL.NUM_LAYERS_PER_BLOCK = 12

# growth rate
__C.MODEL.GROWTH_RATE = 12

# use bias or not
__C.MODEL.USE_BIAS = False

# threshold of final segmentation
__C.MODEL.THRESHOLD = 0.5