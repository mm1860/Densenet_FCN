import argparse
import os.path as osp
import sys

import tensorflow as tf

from config import cfg
from fcn import FCN
from solver import test_model_2D, test_model_3D
from utils.logger import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Test a FCN-DenseNet network")
    parser.add_argument("--model", dest="model", required=True, type=str,
                        help="model to test")
    parser.add_argument("--model_path", dest="model_path", default=None, type=str,
                        help="directory to store all models")
    parser.add_argument("--testset", dest="test_set", default=cfg.DATA.TESTSET, type=str,
                        help="dataset for testing")
    parser.add_argument("--iter", dest="iter", required=True, type=int,
                        help="which checkpoint to load, identified by iter")
    parser.add_argument("--mode", dest="mode", default="2D", type=str, 
                        choices=["2D", "3D"],
                        help="test mode (2D/3D image, default is 2D)")
    parser.add_argument("--output", dest="output", default=None, type=str,
                        help="directory to store test output")
    parser.add_argument("--tag", dest="tag", default="default", type=str,
                        help="tag of the model directory")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    logfile = osp.join(cfg.LOG_DIR, "test_%s_%s_iter_%d" % (args.rag, args.model, args.iter))
    logger = create_logger(log_file=log_file, withtime=True)

    logger.info("Args:")
    logger.info(args)

    logger.info("Configuration: ")
    logger.info(cfg)

    model_path = osp.join(cfg.SRC_DIR, cfg.model_path or cfg.OUTPUT_DIR, args.tag)
    model_file = osp.join(model_path, "{:s}_iter_{:d}.ckpt".format(args.model, args.iter))

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

        if osp.exists(model_file):
            logger.info("Loading checkpoint from " + model_file)
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
            logger.info("Model loaded")
        else:
            raise FileNotFoundError("Invalid model tag or iters!")
    
    test_path = osp.join(cfg.SRC_DIR, args.output) if args.output else None
    if args.mode == "2D":
        test_model_2D(sess, net, args.test_set, test_path)
    elif args.mode == "3D":
        test_model_3D(sess, net, args.test_set, test_path)
    else:
        raise ValueError("Only support 2D and 3D test routine.")

    sess.close()
