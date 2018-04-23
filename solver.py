import os
import os.path as osp
import time
from glob import glob
from collections import defaultdict as ddict

import cv2
import numpy as np
import tensorflow as tf

from config import cfg, global_logger
from data_loader import MedImageLoader2D, MedImageLoader3D
from fcn import FC_DenseNet
from networks import metric_3D
from utils.timer import Timer
from utils.tb_logger import summary_scalar

try:
    import cPickle as pickle
except ImportError:
    import pickle



class SolverWrapper(object):
    """ A wrapper class for solver
    """
    def __init__(self, network:FC_DenseNet, train_set, val_set, output_dir, tbdir, 
                model_name="default"):
        self.net = network
        self.train_set = train_set
        self.val_set = val_set
        self.output_dir = output_dir
        self.tbdir = tbdir
        self.tbvaldir = tbdir + "_val"
        self.model_prefix = model_name
        self.logger = global_logger(cfg.LOGGER)

    def _construct_graph(self, sess:tf.Session):
        with sess.graph.as_default():
            # set random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)
            # build main computation graph
            layers = self.net.create_architecture("TRAIN")
            # define losses
            losses = layers["total_loss"]
            # set learning rate and momentum
            lr = tf.Variable(cfg.TRAIN.LR, trainable=False)
            if cfg.OPTIMIZER.METHOD == "adam":
                optimizer = tf.train.AdamOptimizer(lr, **cfg.OPTIMIZER.ADAM.ARGS)
            else:
                raise ValueError("Not supported optimizer")
            #optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
            
            # compute gradients
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(losses)

            # handle saver ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
            self.writer_val = tf.summary.FileWriter(self.tbvaldir)

        return lr, train_op

    def _find_previous(self):
        sfiles = osp.join(self.output_dir, self.model_prefix + "_iter_*.ckpt.meta")
        sfiles = glob(sfiles)
        sfiles.sort(key=osp.getmtime)
        
        # get snapshot
        redfiles = []
        for step in cfg.TRAIN.LR_STEP:
            redfiles.append(osp.join(self.output_dir, self.model_prefix + "_iter_{:d}.ckpt.meta".format(step + 1)))

        sfiles = [ss.replace(".meta", "") for ss in sfiles if ss not in redfiles]
        
        nfiles = osp.join(self.output_dir, self.model_prefix + "_iter_*.pkl")
        nfiles = glob(nfiles)
        nfiles.sort(key=osp.getmtime)
        redfiles = [redfile.replace(".ckpt.meta", ".pkl") for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

    def _snapshot(self, sess, iter):
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # store the weights
        sfilename = self.model_prefix + "_iter_{:d}.ckpt".format(iter)
        sfilename = osp.join(self.output_dir, sfilename)
        self.saver.save(sess, sfilename)
        self.logger.info("Write snapshot to {:s}".format(sfilename))

        nfilename = self.model_prefix + "_iter_{:d}.pkl".format(iter)
        nfilename = osp.join(self.output_dir, nfilename)
        
        with open(nfilename, "wb") as fid:
            pickle.dump(np.random.get_state(), fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataloader._cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataloader._perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataloader_val._cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataloader_val._perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return sfilename, nfilename

    def _snapshot_best(self, sess):
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # store the weights
        sfilename = self.model_prefix + "_best" + ".ckpt"
        sfilename = osp.join(self.output_dir, sfilename)
        self.saver.save(sess, sfilename)
        self.logger.info("Write better snapshot to {:s}".format(sfilename))

    def _from_snapshot(self, sess, sfile, nfile):
        self.logger.info('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        self.logger.info('Restored.')
        # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
        # tried my best to find the random states so that it can be recovered exactly
        # However the Tensorflow state is currently not available
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            cur = pickle.load(fid)
            perm = pickle.load(fid)
            cur_val = pickle.load(fid)
            perm_val = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)

            np.random.set_state(st0)
            self.dataloader._cur = cur
            self.dataloader._perm = perm
            self.dataloader_val._cur = cur_val
            self.dataloader_val._perm = perm_val

        return last_snapshot_iter

    def _del_snapshot(self, np_paths, ss_paths):
        rm_num = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for _ in range(rm_num):
            nfile = np_paths[0]
            os.remove(nfile)
            np_paths.remove(nfile)

        rm_num = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for _ in range(rm_num):
            sfile = ss_paths[0]
            if osp.exists(sfile):
                os.remove(sfile)
            else:
                os.remove(sfile + '.data-00000-of-00001')
                os.remove(sfile + '.index')
            os.remove(sfile + '.meta')
            ss_paths.remove(sfile)

    def _initialize(self, sess):
        np_paths = []
        ss_paths = []
        # initalize variables
        sess.run(tf.global_variables_initializer())

        last_snapshot_iter = 0
        rate = cfg.TRAIN.LR
        lr_step = list(cfg.TRAIN.LR_STEP)

        return rate, last_snapshot_iter, lr_step, np_paths, ss_paths


    def _restore(self, sess, sfile, nfile):
        np_paths = [nfile]
        ss_paths = [sfile]
        # restore model
        last_snapshot_iter = self._from_snapshot(sess, sfile, nfile)
        # learning rate
        rate = cfg.TRAIN.LR
        lr_step = []
        for step in cfg.TRAIN.LR_STEP:
            if last_snapshot_iter > step:
                rate *= cfg.TRAIN.LR_DECAY
            else:
                lr_step.append(step)

        return rate, last_snapshot_iter, lr_step, np_paths, ss_paths

    def train_model(self, sess, max_iters):
        # build data layer
        self.dataloader = MedImageLoader2D(cfg.DATA.ROOT_DIR, self.train_set, cfg.TRAIN.BS, 
                                           wwidth=cfg.IMG.W_WIDTH, wlevel=cfg.IMG.W_LEVEL,
                                           random=False, shuffle=True)
        self.dataloader_val = MedImageLoader2D(cfg.DATA.ROOT_DIR, self.val_set, cfg.TRAIN.BS, 
                                               wwidth=cfg.IMG.W_WIDTH, wlevel=cfg.IMG.W_LEVEL, 
                                               random=False, shuffle=True)
        self.dataloader_val_total = MedImageLoader2D(cfg.DATA.ROOT_DIR, self.val_set, cfg.TRAIN.BS,
                                                     wwidth=cfg.IMG.W_WIDTH, wlevel=cfg.IMG.W_LEVEL, 
                                                     random=False, shuffle=False)

        # construct graph
        lr, train_op = self._construct_graph(sess)
        # find previous snapshots
        lsf, nfiles, sfiles = self._find_previous()

        # initialize or restore variables
        rate, last_snapshot_iter, lr_step, np_paths, ss_paths = \
            self._initialize(sess) if lsf == 0 else self._restore(sess, sfiles[-1], nfiles[-1])
        
        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        lr_step.append(max_iters)
        lr_step.reverse()
        next_lr_step = lr_step.pop()
        
        best_dice = 0.0
        best_voe = 100.0
        while iter < max_iters + 1:
            # learning rate
            if iter == next_lr_step + 1:
                self._snapshot(sess, iter)
                rate *= cfg.TRAIN.LR_DECAY
                sess.run(tf.assign(lr, rate))
                next_lr_step = lr_step.pop()

            # train step            
            timer.tic()
            train_batch = next(self.dataloader)
            now = time.time()
            if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                loss, cross_entropy, dice, voe, vd, summary = self.net.train_step(
                    sess, train_batch, train_op, cfg.TRAIN.KEEP_PROB, with_summary=True)
                self.writer.add_summary(summary, float(iter))
                val_batch = next(self.dataloader_val)
                summary_val = self.net.get_val_summary(sess, val_batch, cfg.TRAIN.KEEP_PROB)
                self.writer_val.add_summary(summary_val, float(iter))
                last_summary_time = now
            else:
                loss, cross_entropy, dice, voe, vd = self.net.train_step(
                    sess, train_batch, train_op, cfg.TRAIN.KEEP_PROB)
            timer.toc()

            # display step
            if iter % cfg.TRAIN.DISPLAY == 0:
                info = "iter {:d}/{:d}, total loss: {:.6f}\n" + " " * 23 + \
                       ">>> cross entropy: {:.6f}\n" + " " * 23 + \
                       ">>> metric Dice:   {:.2f}\n" + " " * 23 + \
                       ">>> metric VOE:    {:.2f}\n" + " " * 23 + \
                       ">>> metric VD:     {:.2f}\n" + " " * 23 + \
                       ">>> lr: {:f}"
                info = info.format(
                    iter, max_iters, loss, cross_entropy, dice, voe, vd, lr.eval())
                self.logger.info(info)
                self.logger.info("speed: {:.3f}s / iter".format(timer.average_time))

            # validation step
            if cfg.VAL.NEED and iter % cfg.VAL.STEPS == 0:
                all_dice = []
                all_voe = []
                all_vd = []
                for i in range(cfg.VAL.NUMBER):
                    val_batch = next(self.dataloader_val_total)
                    dice, voe, vd = self.net.test_step_2D(sess, val_batch, 1.0)
                    all_dice.append(dice)
                    all_voe.append(voe)
                    all_vd.append(vd)
                mean_dice = np.mean(all_dice)
                mean_voe = np.mean(all_voe)
                mean_vd = np.mean(all_vd)
                info = "Validation ...\n" + " " * 23 + \
                       ">>> mean Dice: {:.2f}\n" + " " * 23 + \
                       ">>> mean VOE:  {:.2f}\n" + " " * 23 + \
                       ">>> mean VD:   {:.2f}"
                info = info.format(mean_dice, mean_voe, mean_vd)
                self.logger.info(info)
                # write to tensorboard
                summary_scalar(self.writer, iter, 
                               tags=["Metric/Dice_val", "Metric/VOE_val", "Metric/VD_val"],
                               values=[mean_dice, mean_voe, mean_vd])
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    self._snapshot_best(sess)
                elif mean_dice == best_dice and mean_voe < best_voe:
                    best_voe = mean_voe
                    self._snapshot_best(sess)

            # snapshot step
            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self._snapshot(sess, iter)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove old snapshots if they are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self._del_snapshot(np_paths, ss_paths)

            iter += 1
        
        if last_snapshot_iter != iter - 1:
            self._snapshot(sess, iter - 1)
        
        self.writer.close()
        self.writer_val.close()

def train_model(network, train_set, val_set, output_dir, tb_dir, 
                model_name="default", 
                max_iters=100000):
    """ Entry of training model
    
    Params
    ------
    `network`: network for training, it should inherit DenseNet class  
    `train_set`: a string, specify the training dataset (the final part of the dataset path)  
    `val_set`: a string, specify the validation dataset (the final part of the dataset path)  
    `output_dir`: a string, directory for saving model  
    `tb_dir`: a string, directory for saving tensorboard events files  
    `model_name`: a string, prefix of model file  
    `max_iters`: max training steps  
    `logger`: logger
    """
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        logger = global_logger(cfg.LOGGER)
        solver = SolverWrapper(network, train_set, val_set, output_dir, tb_dir, model_name)
        logger.info("Begin training...")
        solver.train_model(sess, max_iters)
        logger.info("Training done!")

def save_prediction(prediction, test_path, test_names, mode):
    if mode == "2D": # 4D tensor
        for i, slice in enumerate(prediction):
            image = (slice * 255).astype(np.uint8)
            save_path = osp.join(test_path, test_names[i].replace(".mhd", ".jpg"))
            try:
                cv2.imwrite(save_path, image)
            except cv2.error as e:
                print("Got error: {:s}".format(e))
    elif mode == "3D":  # 5D tensor, the shape of last dim must be 1
        prediction = np.squeeze(prediction, axis=-1)
        for i, volume in enumerate(prediction):
            pass


def test_model_2D(sess, net:FC_DenseNet, test_set, test_path):
    np.random.seed(cfg.RNG_SEED)
    logger = global_logger(cfg.LOGGER)
    
    dataloader = MedImageLoader2D(cfg.DATA.ROOT_DIR, test_set, cfg.TEST.BS_2D, 
                                  wwidth=cfg.IMG.W_WIDTH, wlevel=cfg.IMG.W_LEVEL, 
                                  random=False, shuffle=False, once=True)
    if test_path:
        if "prob" in cfg.VAL.MAP.lower():
            ret_image = "Prediction"
        elif "bin" in cfg.VAL.MAP.lower():
            ret_image = "Binary_Pred"
        else:
            raise ValueError("Wrong validation map type ({:s})."
                             " Please choice from [probability, binary].".format(cfg.VAL.MAP))
    else:
        ret_image = False

    timer = Timer()

    total_dice = []
    total_voe = []
    total_vd = []
    for test_batch in dataloader:
        timer.tic()
        logger.info("Processing image: {:s}".format(test_batch["names"][0].replace("_p", "").replace(".mhd", "")))
        result = net.test_step_2D(sess, test_batch, keep_prob=1.0, ret_image=ret_image)

        if ret_image:
            save_prediction(result[0], test_path, test_batch["names"], mode="2D")
            result = result[1:]
        
        info = "batch Dice: {:.3f}\n" + \
               " " * 23 + ">>> batch VOE:  {:.3f}\n" + \
               " " * 23 + ">>> batch VD:   {:.3f}"
        logger.info(info.format(*result))
        timer.toc()

        total_dice.append(result[0])
        total_voe.append(result[1])
        total_vd.append(result[2])

    mean_dice = np.mean(total_dice)
    mean_voe = np.mean(total_voe)
    mean_vd = np.mean(total_vd)
    info = "mean dice: {:.3f}\n" + \
           " " * 23 + ">>> mean_voe:  {:.3f}\n" + \
           " " * 23 + ">>> mean_vd:   {:.3f}"
    info = info.format(mean_dice, mean_voe, mean_vd)
    logger.info(info)

def test_model_3D(sess, net:FC_DenseNet, test_set, test_path):
    np.random.seed(cfg.RNG_SEED)
    logger = global_logger(cfg.LOGGER)

    dataloader = MedImageLoader3D(cfg.DATA.ROOT_DIR, test_set, cfg.TEST.BS_3D,
                                  wwidth=cfg.IMG.W_WIDTH, wlevel=cfg.IMG.W_LEVEL, 
                                  random=False, shuffle=False, once=True)
    if "prob" in cfg.VAL.MAP.lower():
        ret_image = "Prediction"
    elif "bin" in cfg.VAL.MAP.lower():
        ret_image = "Binary_Pred"
    else:
        raise ValueError("Wrong validation map type ({:s})."
                            " Please choice from [probability, binary].".format(cfg.VAL.MAP))

    timer = Timer()

    metrics = ddict(list)
    bs = cfg.TEST.BS_2D
    for test_volumes in dataloader:
        timer.tic()
        for test_image, test_label, test_name, test_meta in zip(test_volumes["images"], 
                                                                test_volumes["labels"], 
                                                                test_volumes["names"],
                                                                test_volumes["meta"]):
            logger.info("Processing image: {:s}".format(test_name[:-6]))
            logits3D = np.zeros_like(test_image, np.float32)
            for j in range(0, len(test_image) - bs + 1, bs):
                test_batch = {"images": test_image[j:j + bs],
                              "labels": test_label[j:j + bs]}
                pred = net.test_step_2D(sess, test_batch, keep_prob=1.0, ret_image=ret_image, ret_metrics=False)
                logits3D[j:j + bs] = pred[0]
            
            last = len(test_image) % bs
            if last != 0:
                j = len(test_image) - last
                # because now network need fixed batch size, we need fill BS_2D with zeros
                #remainder, times = bs % last, bs // last
                images = np.zeros((bs,) + test_image.shape[1:], dtype=test_image.dtype)
                labels = np.zeros_like(images)
                images[:last] = test_image[j:]
                labels[:last] = test_label[j:]
                #for k in range(times):
                #   images[2*k:2*k+2] = test_volume[j:]
                #if remainder > 0:
                #    k = bs - remainder
                #    images[k:] = test_volume[]

                test_batch = {"images": images, "labels": labels}
                pred = net.test_step_2D(sess, test_batch, keep_prob=1.0, ret_image=ret_image, ret_metrics=False)
                logits3D[j:] = pred[0][:last]

            # save 3d volume data as mhd
            if ret_image:
                save_prediction(logits3D, test_path, test_name, mode="3D")

            # Calculate 3D metrics
            metrics_3D = metric_3D(logits3D, test_label, 
                                   sampling=list(reversed(test_meta["ElementSpacing"])), # (z, y, x)
                                   connectivity=3)
            for key, val in metrics_3D.items():
                metrics[key].append(val)
            info = "mean Dice: {:.3f}\n" + \
                   " " * 23 + ">>> batch VOE:  {:.3f}\n" + \
                   " " * 23 + ">>> batch VD:   {:.3f}\n" + \
                   " " * 23 + ">>> batch ASD:  {:.3f}\n" + \
                   " " * 23 + ">>> batch RMSD: {:.3f}\n" + \
                   " " * 23 + ">>> batch MSD:  {:.3f}"
            info = info.format(
                    metrics_3D["Dice"], metrics_3D["VOE"], metrics_3D["VD"],
                    metrics_3D["ASD"], metrics_3D["RMSD"], metrics_3D["MSD"])
            logger.info(info)
                
        timer.toc()

    info = "mean Dice: {:.3f}\n" + \
           " " * 23 + ">>> mean VOE:  {:.3f}\n" + \
           " " * 23 + ">>> mean VD:   {:.3f}\n" + \
           " " * 23 + ">>> mean ASD:  {:.3f}\n" + \
           " " * 23 + ">>> mean RMSD: {:.3f}\n" + \
           " " * 23 + ">>> mean MSD:  {:.3f}"
    info = info.format(
              np.mean(metrics["Dice"]), np.mean(metrics["VOE"]), np.mean(metrics["VD"]),
              np.mean(metrics["ASD"]), np.mean(metrics["RMSD"]), np.mean(metrics["MSD"]))
    logger.info(info)


if __name__ == "__main__":
    if False:
        # check computation graph
        net = FC_DenseNet(24, 3, 12, 12, bc_mode=True, name="FCN-DenseNet")
        solver = SolverWrapper(net, None, None, "a", "b")  
        
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True

        with tf.Session(config=tfconfig) as sess:
            solver._construct_graph(sess)
            solver.writer.close()
    if True:
        # test save_prediction function
        prediction = np.random.randn(2, 512, 512, 1)
        save_prediction(prediction, "./", ["a.mhd", "b.mhd"])