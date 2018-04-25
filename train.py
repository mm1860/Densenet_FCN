import argparse
import os
import os.path as osp
import sys
from pprint import pprint

from config import cfg, update_cfg
from fcn import FC_DenseNet
from unet import UNet
import udn
from solver import train_model
from utils.logger import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FCN-DenseNet network")
    parser.add_argument("--cfg", dest="cfg_file", default=None, type=str,
                        help="extra configuration (it will cover default config in config.py)")


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.cfg_file:
        update_cfg(args.cfg_file)

    logdir = osp.join(cfg.SRC_DIR, cfg.LOG_DIR)
    if not osp.exists(logdir):
        os.makedirs(logdir)
    logfile = osp.join(logdir, "train_%s_%s_iter_%d" % (cfg.TAG, cfg.PREFIX, cfg.TRAIN.MAX_ITERS))
    logger = create_logger(log_file=logfile, withtime=True, propagate=False, name=cfg.LOGGER)

    logger.info("Configuration: ")
    for handler in logger.handlers:
        pprint(cfg, handler.stream)

    # define network
    if cfg.BACKBONE == "FC-Densenet":
        net = FC_DenseNet(cfg.MODEL.INIT_CHANNELS,
                          cfg.MODEL.BLOCKS,
                          cfg.MODEL.NUM_LAYERS_PER_BLOCK,
                          cfg.MODEL.GROWTH_RATE,
                          bc_mode=True,
                          name="FCN-DenseNet")
    elif cfg.BACKBONE == "UNet":
        net = UNet(cfg.UNET.INIT_CHANNELS,
                   cfg.UNET.NUM_DOWN_SAMPLE,
                   cfg.UNET.NUM_CONV_PER_LAYER,
                   name="UNet")
    elif cfg.BACKBONE == "UDN":
        
    else:
        raise ValueError("Un supported backbone: {:s}".format(cfg.BACKBONE))

    # define output directory
    output_dir = osp.join(cfg.SRC_DIR, cfg.OUTPUT_DIR, cfg.TAG)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # define tensorboard directory
    tb_dir = osp.join(cfg.SRC_DIR, cfg.TB_DIR, cfg.TAG)
    if not osp.exists(tb_dir):
        os.makedirs(tb_dir)

    train_model(net, 
                cfg.DATA.TRAINSET, 
                cfg.DATA.VALSET, 
                output_dir, 
                tb_dir, 
                cfg.PREFIX, 
                cfg.TRAIN.MAX_ITERS)
