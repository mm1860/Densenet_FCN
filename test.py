import argparse
import os.path as osp
import sys
from pprint import pprint

import cv2
import numpy as np
import tensorflow as tf

from config import cfg
from data_loader import MedImageLoader2D
from fcn import FCN
from utils.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Test a FCN-DenseNet network")
    parser.add_argument("--model", dest="model", required=True, type=str,
                        help="model to test")
    parser.add_argument("--testset", dest="test_set", required=True, type=str,
                        help="dataset to test")
    parser.add_argument("--iter", dest="iter", required=True, type=int,
                        help="which checkpoint to load, identified by iter")
    parser.add_argument("--mode", dest="mode", default="2D", type=str, 
                        choices=["2D", "3D"],
                        help="test mode (2D/3D image, default is 2D)")
    parser.add_argument("--output", dest="output", default=None, type=str,
                        help="directory to store test output")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def save_prediction(prediction, test_path, test_names):
    for i, slice in enumerate(prediction):
        image = (prediction * 255).astype(np.uint8)
        save_path = osp.join(test_path, test_names[i].replace(".mhd", ".jpg"))
        cv2.imwrite(save_path, image)

def test_model_2D(sess, net:FCN, test_set, test_path):
    np.random.seed(cfg.RNG_SEED)
    
    dataloader = MedImageLoader2D(cfg.DATA.ROOT_DIR, test_set, cfg.TEST.BS_2D, once=True)
    ret_image = True if test_path else False

    timer = Timer()

    total_dice = []
    total_voe = []
    total_vd = []
    for test_batch in dataloader:
        timer.tic()
        pred, dice, voe, vd = net.test_step_2D(sess, test_batch, ret_image, keep_prob=1.0)
        timer.toc()

        total_dice.append(dice)
        total_voe.append(voe)
        total_vd.append(vd)

        save_prediction(pred, test_path, test_batch["names"])

    mean_dice = np.mean(total_dice)
    mean_voe = np.mean(total_voe)
    mean_vd = np.mean(total_vd)
    print(" mean dice: {:.3f}\n mean_voe: {:.3f}\n mean_vd: {:.3f}".format(
        mean_dice, mean_voe, mean_vd
    ))

def test_model_3D(sess, net:FCN, test_set, test_path):
    raise NotImplementedError

if __name__ == '__main__':
    args = parse_args()
    print("Args:")
    print(args)

    print("Configuration: ")
    pprint(cfg)

    model_path = osp.join(cfg.SRC_DIR, "output")
    model_file = osp.join(model_path, "{:s}_iter_{:d}.ckpt".format(args.model, args.iter))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # define computation graph
    main_graph = tf.Graph()

    sess = tf.Session(config=tfconfig, graph=main_graph)

    with main_graph.as_default():
        net = FCN(cfg.MODEL.INIT_CHANNELS, cfg.MODEL.BLOCKS, cfg.MODEL.NUM_LAYERS_PER_BLOCK,
                    cfg.MODEL.GROWTH_RATE, bc_mode=True, "FCN-DenseNet")
        net.create_architecture("TEST")

        if osp.exists(model_file):
            print("Loading checkpoint from " + model_file)
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
            print("Model loaded")
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
