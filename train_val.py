import os
import os.path as osp
import time
from glob import glob

import numpy as np
import tensorflow as tf

from config import cfg
from data_loader import FullImageLoader
from fcn import FCN
from utils.timer import Timer

try:
    import cPickle as pickle
except ImportError:
    import pickle



class SolverWrapper(object):
    """ A wrapper class for solver

    """
    def __init__(self, network:FCN, train_set, val_set, output_dir, tbdir, model_tag="default"):
        self.net = network
        self.train_set = train_set
        self.val_set = val_set
        self.output_dir = output_dir
        self.tbdir = tbdi
        self.tbvaldir = tbdir + "_val"
        if not osp.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)
        self.model_prefix = model_tag

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
            optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
            # compute gradients
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
        print("Write snapshot to {:s}".format(sfilename))

        nfilename = self.model_prefix + "_iter_{:d}.pkl".format(iter)
        nfilename = osp.join(self.output_dir, nfilename)
        
        with open(nfilename, "wb") as fid:
            pickle.dump(np.random.get_state(), fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataloader._cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataloader._perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataloader_val._cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataloader_val._perm, fid, pickle.HIGHEST_PROTOCOL)

        return sfilename, nfilename

    def _from_snapshot(self, sess, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        print('Restored.')
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
            if osp.exists(sfile)
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
        last_snapshot_iter = self._restore(sess, sfile, nfile)
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
        self.dataloader = FullImageLoader(self.train_set)
        self.dataloader_val = FullImageLoader(self.val_set, random=True)

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
        lr_step.append(max_iter)
        lr_step.reverse()
        next_lr_step = lr_step.pop()
        
        while iter < max_iter + 1:
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
            if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL
                loss, cross_entropy, dice, voe, vd, summary = self.net.train_step(
                    sess, train_batch, train_op, cfg.TRAIN.KEEP_PROB, with_summary=True)
                self.writer.add_summary(summary, float(iter))
                val_batch = next(self.dataloader_val)
                summary_val = self.net.get_val_summary(sess, cfg.TRAIN.KEEP_PROB, val_batch)
                self.writer_val.add_summary(summary_val, float(iter))
                last_summary_time = now
            else:
                loss, cross_entropy, dice, voe, vd = self.net.train_step(
                    sess, train_batch, train_op, cfg.TRAIN.KEEP_PROB)
            timer.toc()

            # display step
            if iter % cfg.TRAIN.DISPLAY == 0:
                print("iter {:d}/{:d}, total loss: {:.6f}\n"
                        " >>> cross entropy: {:.6f}\n"
                        " >>> metric Dice:   {:.2f}\n"
                        " >>> metric Voe:    {:.2f}\n"
                        " >>> metric Vd:     {:.2f}\n"
                        " >>>lr: {:f}".format(
                    iter, max_iter, loss, cross_entropy, dice, voe, vd, lr.eval()
                ))

            # snapshot step
            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self._snapshot(sess, iter)
                np_paths.append(np_path)
                ss_paths.append(ss_path)
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self._del_snapshot(np_paths, ss_paths)

            iter += 1
        
        if last_snapshot_iter != iter - 1:
            self._snapshot(sess, iter - 1)
        
        self.writer.close()
        self.writer_val.close()

def train_net(network, train_set, val_set, output_dir, tb_dir, model_tag="default", max_iters=100000):

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        solver = SolverWrapper(network, train_set, val_set, output_dir, tb_dir, model_tag)
        print("\nBegin training...")
        solver.train_model(sess, max_iters)
        print("Training done!")
