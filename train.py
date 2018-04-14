import argparse
import os
import os.path as osp
import sys
from pprint import pprint

from config import cfg, update_cfg
from fcn import FCN
from solver import train_model
from utils.logger import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FCN-DenseNet network")
    parser.add_argument("GPU", metavar="gpu_id", type=str, nargs=1,
                        help="GPU id for training")
    parser.add_argument("--trainset", dest="trainset", default=cfg.DATA.TRAINSET, type=str,
                        help="dataset for training")
    parser.add_argument("--valset", dest="valset", default=cfg.DATA.VALSET, type=str,
                        help="dataset for validation")
    parser.add_argument("--output", dest="output_dir", default=None, type=str,
                        help="directory to store all models")
    parser.add_argument("--logdir", dest="tbdir", default=None, type=str,
                        help="directory to store all summary event files")
    parser.add_argument("--max_iters", dest="max_iters", default=100000, type=str,
                        help="number of iterations to train")
    parser.add_argument("--tag", dest="tag", default="default", type=str,
                        help="tag of the model directory")
    parser.add_argument("--model", dest="model", default="default", type=str,
                        help="model prefix")
    parser.add_argument("--cfg", dest="cfg_file", default=None, type=str,
                        help="extra configuration (it will cover default config in config.py)")


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    logfile = osp.join(cfg.LOG_DIR, "train_%s_%s_iter_%d" % (args.tag, args.model, args.max_iters))
    logger = create_logger(log_file=logfile, withtime=True)

    logger.info("\nArgs:")
    logger.info(args)

    if args.cfg_file:
        update_cfg(args.cfg_file)
    logger.info("\nConfiguration: ")
    for handler in logger.handlers:
        pprint(cfg, handler.stream)

    # define network
    net = FCN(cfg.MODEL.INIT_CHANNELS,
              cfg.MODEL.BLOCKS,
              cfg.MODEL.NUM_LAYERS_PER_BLOCK,
              cfg.MODEL.GROWTH_RATE,
              bc_mode=True,
              name="FCN-DenseNet")
    
    # define output directory
    output_dir = osp.join(cfg.SRC_DIR, args.output_dir or cfg.OUTPUT_DIR, args.tag)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # define tensorboard directory
    tb_dir = osp.join(cfg.SRC_DIR, args.tbdir or cfg.TB_DIR, args.tag)
    if not osp.exists(tb_dir):
        os.makedirs(tb_dir)

    train_model(net, args.trainset, args.valset, output_dir, tb_dir, args.model, args.max_iters)