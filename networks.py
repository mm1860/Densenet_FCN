import numpy as np
import tensorflow as tf
from scipy.ndimage import morphology as mph
from tensorflow.contrib import slim as slim
from tensorflow.python.layers import utils

from config import cfg

class Networks(object):
    def __init__(self):
        self._act_summaries = []
        self._image_summaries = []
        self._layers = {}
        self._losses = {}
        self._metrics_2D = {}

    def _build_network(self, is_training=True, reuse=None, name=None):
        raise NotImplementedError

    def _net_arg_scope(self, training=True):
        raise NotImplementedError

    def _add_summaries(self):
        self.val_summaries = []
        with tf.device("/cpu:0"):
            # trainable variables
            for var in tf.trainable_variables():
                tf.summary.histogram('Trainable/' + var.op.name, var)

            # denseblock output
            for out in self._act_summaries:
                tf.summary.histogram('Activation/' + out.op.name, out)

            # add image summaries
            for image in self._image_summaries:
                image = tf.cast(image, tf.float32, name="ToFloat32")
                self.val_summaries.append(tf.summary.image("Image/" + image.op.name, image))

            # add losses
            for key, var in self._losses.items():
                self.val_summaries.append(tf.summary.scalar("Loss/" + key, var))

            # add metrics
            for key, var in self._metrics_2D.items():
                self.val_summaries.append(tf.summary.scalar("Metric/" + key, var))

    def _metric_dice(self, logits, labels, eps=1e-5, name=None):
        """ Dice coefficient

        Params
        ------
        `logits`: binary seg prediction of the model, shape is the same with `logits`
        `labels`: labeled segmentation map, shape [batch_size, None, None, 1]
        `eps`: epsilon is set to avoid dividing zero
        `name`: operation name used in tensorflow

        Returns
        -------
        `dice`: average dice coefficient
        """
        dim = len(logits.get_shape())
        sum_axis = list(range(1, dim))
        with tf.variable_scope(name, "Dice"):
            logits = tf.cast(logits, tf.float32)
            labels = tf.cast(labels, tf.float32)
            
            intersection = tf.reduce_sum(logits * labels, axis=sum_axis)
            left = tf.reduce_sum(logits, axis=sum_axis)
            right = tf.reduce_sum(labels, axis=sum_axis)
            dice = (2 * intersection) / (left + right + eps)

        return tf.reduce_mean(dice)

    def _metric_VOE(self, logits, labels, eps=1e-5, name=None):
        """ Volumetric Overlap Error

        numerator / denominator
 
        Params
        ------
        reference `self._metric_dice`
        
        Returns
        -------
        `dice`: average voe 
        """
        dim = len(logits.get_shape())
        sum_axis = list(range(1, dim))
        with tf.variable_scope(name, "VOE"):
            logits = tf.cast(logits, tf.float32)
            labels = tf.cast(labels, tf.float32)

            nume = tf.reduce_sum(logits * labels, axis=sum_axis)
            deno = tf.reduce_sum(tf.clip_by_value(logits + labels, 0.0, 1.0), axis=sum_axis)
            voe = 100 * (1.0 - nume / (deno + eps))

        return tf.reduce_mean(voe)

    def _metric_VD(self, logits, labels, eps=1e-5, name=None):
        """ Relative Volume Difference

        Since the measure is not symmetric, it is no metric. 
 
        Params
        ------
        reference `self._metric_dice`

        Returns
        -------
        `dice`: average vd
        """
        dim = len(logits.get_shape())
        sum_axis = list(range(1, dim))
        with tf.variable_scope(name, "VD"):
            logits = tf.cast(logits, tf.float32)
            labels = tf.cast(labels, tf.float32)

            A = tf.reduce_sum(logits, axis=sum_axis)
            B = tf.reduce_sum(labels, axis=sum_axis)
            vd = 100 * (tf.abs(A - B) / (B + eps))

        return tf.reduce_mean(vd)

    def _add_2D_metries(self, logits, labels, name=None):
        with tf.variable_scope(name, "Metries_2D"):
            dice = self._metric_dice(logits, labels)
            voe = self._metric_VOE(logits, labels)
            vd = self._metric_VD(logits, labels)

        self._metrics_2D["Dice"] = dice
        self._metrics_2D["VOE"] = voe
        self._metrics_2D["VD"] = vd

    def _add_2D_metries_with_prob(self):
        self._add_2D_metries(self._layers["Prediction"], self._mask)

    def _add_2D_metries_with_bin(self):
        self._add_2D_metries(self._layers["Binary_Pred"], self._mask)

    def _add_losses(self, name=None):
        with tf.variable_scope(name, "Loss"):
            labels = tf.squeeze(self._mask, axis=[-1])
            logits = self._layers["logits"]
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels, logits) * cfg.MODEL.CROSS_ENTROPY
            #cross_entropy = tf.losses.sigmoid_cross_entropy(tf.one_hot(labels, 2), logits) * cfg.MODEL.CROSS_ENTROPY
            self._losses["cross_entropy"] = cross_entropy

            if cfg.MODEL.WEIGHT_DECAY != 0.0:
                regularization_losses = tf.add_n(tf.losses.get_regularization_losses(), "Regu")
            else:
                regularization_losses = 0.0
            self._losses["total_loss"] = cross_entropy + regularization_losses

        return cross_entropy

    def create_architecture(self, mode, name=None):
        shape = (cfg.TRAIN.BS, None, None, cfg.IMG.CHANNEL)
        self._image = tf.placeholder(tf.float32, shape, name="Image")
        self._mask = tf.placeholder(tf.int32, shape, name="Mask")
        self._keep_prob = tf.placeholder(tf.float32, (), name="KeepProb")
        self._image_summaries.append(self._image)
        self._image_summaries.append(self._mask)

        self._mode = mode
        
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.MODEL.WEIGHT_DECAY)
        if cfg.MODEL.WEIGHT_INITIALIZER == "trunc_norm":
            weights_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=0.01)
        elif cfg.MODEL.WEIGHT_INITIALIZER == "rand_norm":
            weights_initializer = tf.initializers.random_normal(mean=0.0, stddev=0.01)
        elif cfg.MODEL.WEIGHT_INITIALIZER == "xavier":
            weights_initializer = tf.contrib.layers.xavier_initializer()
        else:
            raise ValueError("Not defined weight initializer: %s" % cfg.MODEL.WEIGHT_INITIALIZER)
        
        if cfg.MODEL.USE_BIAS:
            biases_regularizer = weights_regularizer if cfg.MODEL.BIAS_DECAY else tf.no_regularizer
            biases_initializer = tf.initializers.constant(0.0)
        else:
            biases_regularizer = None
            biases_initializer = None

        training = mode == "TRAIN"
        testing = mode == "TEST"

        layers_out = {}
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=weights_regularizer,
                            weights_initializer=weights_initializer,
                            biases_regularizer=biases_regularizer,
                            biases_initializer=biases_initializer):
            with slim.arg_scope(self._net_arg_scope(training=training)):
                prediction = self._build_network(training, name=name)
        layers_out["prediction"] = prediction

        self._add_losses()
        layers_out.update(self._losses)

        if "prob" in cfg.TRAIN.MAP.lower():
            pred = True
        elif "bin" in cfg.TRAIN.MAP.lower():
            pred = False
        else:
            raise ValueError("Wrong validation map type ({:s})."
                             " Please choice from [probability, binary].".format(cfg.VAL.MAP))
        utils.smart_cond(pred, self._add_2D_metries_with_prob, self._add_2D_metries_with_bin, "Metrics2D")

        self._add_summaries()
        self._summary_op = tf.summary.merge_all()
        self._summary_op_val = tf.summary.merge(self.val_summaries)

        return layers_out

    def train_step(self, sess:tf.Session, train_batch, train_op, keep_prob, with_summary=False):
        """ A train step
        """
        feed_dict = {self._image: train_batch["images"], self._mask: train_batch["labels"],
                    self._keep_prob: keep_prob}
        
        if with_summary:
            fetches = [self._losses["total_loss"], self._losses["cross_entropy"], self._metrics_2D["Dice"], 
                        self._metrics_2D["VOE"], self._metrics_2D["VD"], self._summary_op, train_op]
            loss, cross_entropy, dice, voe, vd, summary, _ = sess.run(fetches, feed_dict=feed_dict)
            return loss, cross_entropy, dice, voe, vd, summary
        else:
            fetches = [self._losses["total_loss"], self._losses["cross_entropy"], self._metrics_2D["Dice"], 
                        self._metrics_2D["VOE"], self._metrics_2D["VD"], train_op]
            loss, cross_entropy, dice, voe, vd, _ = sess.run(fetches, feed_dict=feed_dict)
            return loss, cross_entropy, dice, voe, vd

    def get_val_summary(self, sess:tf.Session, data_batch, keep_prob):
        feed_dict = {self._image: data_batch["images"], self._mask: data_batch["labels"],
                    self._keep_prob: keep_prob}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def test_step_2D(self, sess:tf.Session, test_batch, keep_prob, ret_image=None, ret_metrics=True):
        if not (ret_image or ret_metrics):
            return None

        feed_dict = {self._image: test_batch["images"], self._mask: test_batch["labels"],
                    self._keep_prob: keep_prob}
        
        fetches = []
        if ret_image:
            assert ret_image in ["Prediction", "Binary_Pred"], "ret_image should choice in [None, 'Prediction', 'Binary_Pred']"
            fetches.append(self._layers[ret_image])
        if ret_metrics:
            fetches.extend([self._metrics_2D["Dice"], self._metrics_2D["VOE"], self._metrics_2D["VD"]])
            
        res = sess.run(fetches, feed_dict=feed_dict)

        return res


def metric_3D(logits3D, labels3D, surface=False, eps=1e-6, **kwargs):
    """ Compute 3D metrics:  
    * (Dice) Dice Coefficient
    * (VOE)  Volumetric Overlap Error
    * (VD)   Relative Volume Difference
    * (ASD)  Average Symmetric Surface Distance
    * (RMSD) Root Mean Square Symmetric Surface Distance
    * (MSD)  Maximum Symmetric Surface Distance

    Params
    ------
    `logits3D`: 3D binary prediction, shape is the same with `labels3D`, it should be 
    a int array or boolean array.  
    `labels3D`: 3D labels for segmentation, shape [None, None, None], it
    shoule be a int array or boolean array. If the dimensions of `logits3D` and `labels3D`
    are greater than 3, then `np.squeeze` will be applied to remove extra single dimension
    and then please make sure these two variables are still have 3 dimensions. For example, 
    shape [None, None, None, 1] or [1, None, None, None, 1] are allowed.  
    `surface`: `logits3D` and `labels3D` represent object surface or not  
    `eps`: epsilon is set to avoid dividing zero  

    Other key word arguments
    ------
    `sampling`: the pixel resolution or pixel size. This is entered as an n-vector
    where n is equal to the number of dimensions in the segmentation i.e. 2D or 3D.
    The default value is 1 which means pixls are 1x1x1 mm in size  
    `connectivity`: creates either a 2D(3x3) or 3D(3x3x3) matirx defining the neghbour-
    hood around which the function looks for neighbouring pixels. Typically, this is 
    defined as a six-neighbour kernel which is the default behaviour of the function  
    `require`: a string or a list of string to specify which metrics need to be return, 
    default this function will return all the metrics listed above. If `surface` is set,
    only ASD, RMSD and MSD can be computed. For example, if use
    ```python
    _metric_3D(logits3D, labels3D, require=["Dice", "VOE", "ASD"])
    ```
    then only these three metrics will be returned.

    Note: `logtis3D` and `labels3D` are all the binary tensor with {0, 1}. If flag 
    `surface` is set, then we ask two input tensors represent 3D object surface, which 
    means that voxels on the surface are set to 1 while others (inside or outside the 
    surface) are set to 0. If flag is not set, then `logits3D` and `labels3D` should
    represent the whole object (solid segmentation).

    Acknowledgement: Thanks to the code snippet from @MLNotebook's blog.
    [Blog link](https://mlnotebook.github.io/post/surface-distance-function/).

    Returns
    -------
    metrics required
    """
    metrics = ["Dice", "VOE", "VD", "ASD", "RMSD", "MSD"]
    need_dist_map = False

    required = kwargs.get("required", None)
    if required is None:
        required = metrics
    elif isinstance(required, str):
        required = [required]
        if required[0] not in metrics:
            raise ValueError("Not supported metric: %s" % required[0])
        elif required in metrics[3:]:
            need_dist_map = True
        else:
            need_dist_map = False

    for req in required:
        if req not in metrics:
            raise ValueError("Not supported metric: %s" % req)
        if (not need_dist_map) and req in metrics[3:]:
            need_dist_map = True

    shape = logits3D.shape
    if logits3D.ndim > 3:
        logits3D = np.squeeze(logits3D).astype(np.int32)
    if labels3D.ndim > 3:
        labels3D = np.squeeze(labels3D).astype(np.int32)

    assert logits3D.shape == labels3D.shape, "Shape mismatch of logits3D and labels3D." \
                                "Logits3D has shape %r while labels3D has shape %r" % (
                                logits3D.shape, labels3D.shape)
    metrics_3D = {}

    if need_dist_map:
        import math

        if surface:
            A = logtis3D
            B = labels3D
        else:
            sampling = kwargs.get("sampling", [1., 1., 1.])
            connectivity = kwargs.get("connectivity", 1)
            disc = mph.generate_binary_structure(logits3D.ndim, connectivity)

            A = logits3D - mph.binary_erosion(logits3D, disc)
            B = labels3D - mph.binary_erosion(labels3D, disc)
        
        dist_mapA = mph.distance_transform_edt(np.logical_not(A), sampling)
        dist_mapB = mph.distance_transform_edt(np.logical_not(B), sampling)

        dist_A2B = dist_mapB[A != 0]
        dist_B2A = dist_mapA[B != 0]

        sum_A_and_B = np.sum(A) + np.sum(B)
        
        if "ASD" in required:
            asd = (np.sum(dist_A2B) + np.sum(dist_B2A)) / (sum_A_and_B + eps)
            metrics_3D["ASD"] = asd
            required.remove("ASD")
        if "RMSD" in required:
            rmsd = math.sqrt((np.sum(dist_A2B ** 2) + np.sum(dist_B2A ** 2)) / (sum_A_and_B + eps))
            metrics_3D["RMSD"] = rmsd
            required.remove("RMSD")
        if "MSD" in required:
            msd = np.maximum(np.max(dist_A2B), np.max(dist_B2A))
            metrics_3D["MSD"] = msd
            required.remove("MSD")

    if required != []:
        logits3D = logits3D.astype(np.float32)
        labels3D = labels3D.astype(np.float32)

        intersection = np.sum(logits3D * labels3D)
        if "Dice" in required:
            dice = (2 * intersection) / (np.sum(logits3D) + np.sum(labels3D) + eps)
            metrics_3D["Dice"] = dice
        if "VOE" in required:
            denominator = np.sum(np.clip(logits3D + labels3D, 0, 1))
            voe = 100 * (1 - intersection / (denominator + eps))
            metrics_3D["VOE"] = voe
        if "VD" in required:
            logits_sum = np.sum(logits3D)
            labels_sum = np.sum(labels3D)
            vd = 100 * (np.abs(np.sum(logits3D) - np.sum(labels3D)) / (labels_sum + eps))
            metrics_3D["VD"] = vd

    return metrics_3D

if __name__ == "__main__":
    # test metric_3D() function
    from utils.Liver_Kits import mhd_reader
    import matplotlib.pyplot as plt
    mhd_file = "D:/DataSet/LiverQL/3Dircadb1_mhd/mask/A001_m.mhd"
    meta, mask = mhd_reader(mhd_file)
    mask = (mask / np.max(mask)).astype(np.int32)
    disc = mph.generate_binary_structure(mask.ndim, 3)
    logits = mph.binary_erosion(mask, disc, 2)
    
    metrics = metric_3D(logits, mask, sample=[1, 1, 1], connectivity=3)
    for k, v in metrics.items():
        print(k, v)
    
