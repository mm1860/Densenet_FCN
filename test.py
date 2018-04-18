import argparse
import os
import os.path as osp
import sys
from pprint import pprint

import tensorflow as tf

from config import cfg, update_cfg
from fcn import FCN
from solver import test_model_2D, test_model_3D
from utils.logger import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Test a FCN-DenseNet network")
    parser.add_argument("--mode", dest="mode", default="2D", type=str, 
                        choices=["2D", "3D"],
                        help="test mode (2D/3D image, default is 2D)")
    parser.add_argument("--cfg", dest="cfg_file", default=None, type=str,
                        help="extra configuration (it will cover default config in config.py)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    if args.cfg_file:
        update_cfg(args.cfg_file)
 
    logdir = osp.join(cfg.SRC_DIR, cfg.LOG_DIR)
    if not osp.exists(logdir):
        os.makedirs(logdir)
    logfile = osp.join(logdir, "test_%s_%s_iter_%d" % (cfg.TAG, cfg.PREFIX, cfg.TEST.ITER))
    logger = create_logger(log_file=logfile, withtime=True, propagate=False, name=cfg.LOGGER)

    logger.info("Configuration: ")
    for handler in logger.handlers:
        pprint(cfg, handler.stream)

    model_path = osp.join(cfg.SRC_DIR, cfg.OUTPUT_DIR, cfg.TAG)
    model_file = osp.join(model_path, "{:s}_iter_{:d}.ckpt".format(cfg.PREFIX, cfg.TEST.ITER))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # define computation graph
    main_graph = tf.Graph()

    sess = tf.Session(config=tfconfig, graph=main_graph)

    with main_graph.as_default():
        net = FCN(cfg.MODEL.INIT_CHANNELS, 
                  cfg.MODEL.BLOCKS, 
                  cfg.MODEL.NUM_LAYERS_PER_BLOCK,
                  cfg.MODEL.GROWTH_RATE, 
                  bc_mode=True, name="FCN-DenseNet")
        net.create_architecture("TEST")

        if osp.exists(model_file + ".meta"):
            logger.info("Loading checkpoint from " + model_file)
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
            logger.info("Model loaded")
        else:
            raise FileNotFoundError("Invalid model tag or iters! Model file: {:s}".format(model_file))
    
    if cfg.PRED_TAG != "":
        test_path = osp.join(cfg.SRC_DIR, cfg.PRED_DIR, cfg.PRED_TAG)
        if not osp.exists(test_path):
            os.makedirs(test_path)
    else:
        test_path = None
    
    if args.mode == "2D":
        test_model_2D(sess, net, cfg.DATA.TESTSET, test_path)
    elif args.mode == "3D":
        test_model_3D(sess, net, cfg.DATA.TESTSET_3D, test_path)
    else:
        raise ValueError("Only support 2D and 3D test routine.")

    sess.close()
