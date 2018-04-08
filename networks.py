import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.ndimage import morphology as mph

from config import cfg

def PReLU(tensor_in:tf.Tensor, name=None):
    with tf.variable_scope(name, "PReLU", [tensor_in]):
        alpha = tf.get_variable("alpha", shape=tensor_in.get_shape(),
                                initializer=tf.constant_initializer(0.),
                                dtype=tensor_in.dtype)
        pos = tf.nn.relu(tensor_in)
        neg = alpha * (tensor_in - tf.abs(tensor_in)) * 0.5

        tensor_out = pos + neg

    return tensor_out

class DenseNet(object):
    def __init__(self, init_channels, num_blocks, num_layers_per_block, 
                    growth_rate=12, bc_mode=True):
        self._tensor_in = tensor_in
        self._init_channels = init_channels
        self._num_blocks = num_blocks
        if isinstance(num_layers_per_block, int):
            self._num_layers_per_block = [num_layers_per_block] * self._num_blocks
        elif isinstance(num_layers_per_block, list):
            self._num_layers_per_block = num_layers_per_block
            if len(num_layers_per_block) < num_blocks:
                extra_len = (num_blocks - 1) // len(num_layers_per_block)
                self._num_layers_per_block.extend(num_layers_per_block * extra_len)
        else:
            raise TypeError("Error type for `num_layers_per_block`")
        self._growth_rate = growth_rate
        self._bc_mode = bc_mode

        self._act_summaries = []
        self._image_summaries = []
        self._layers = {}
        self._losses = {}
        self._metrics_2D = {}

    def _unit_layer(self, tensor_in:tf.Tensor, out_channels, kernel_size, name, training=True):
        if isinstance(out_channels, float):
            out_channels = int(tensor_in.shape.as_list()[-1] * out_channels)
        with tf.variable_scope(name):
            # batch_norm need UPDATE_OPS
            tensor_out = slim.batch_norm(tensor_in, is_training=training, activation_fn=PReLU)
            tensor_out = slim.conv2d(tensor_out, out_channels, [kernel_size]*2)
            tensor_out = slim.dropout(tensor_out, self._keep_prob)
            
        return tensor_out

    def _internal_layer(self, tensor_in, growth_rate, training=True, bc_mode=False, name=None):
        with tf.variable_scope(name, "InternalLayer"):
            if bc_mode:
                bottleneck_out = self._unit_layer(tensor_in, growth_rate * 4, 1, "Bottleneck", training)
                tensor_out = self._unit_layer(bottleneck_out, growth_rate, 3, "CompositeFunction", training)
            else:
                tensor_out = self._unit_layer(tensor_in, growth_rate, 3, "CompositeFunction", training)

            tensor_out = tf.concat((tensor_in, tensor_out), axis=-1)

        return tensor_out

    def _transition_layer(self, tensor_in, out_channels, training=True, name=None):
        raise NotImplementedError

    def create_dense_layer(self, tensor_in):
        for i in range(self._num_blocks):
            with tf.variable_scope("DenseBlock%d" % (i + 1))
                tensor_out = slim.repeat(tensor_in, self._num_layers_per_block[i], self._internal_layer,
                                        self._growth_rate, is_training, self._bc_mode)
                self._act_summaries.append(tensor_out)
                if i < self._num_blocks - 1:
                    tensor_out = self._transition_layer(tensor_out, 0.5, is_training)

        return tensor_out

    def _build_network(self, is_training=True, reuse=None, name=None):
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
                self.val_summaries.append(tf.summary.image("Image/" + image.op.name, image))

            # add losses
            for key, var in self._losses.items():
                self.val_summaries.append(tf.summary.scalar(key, var))

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
            left = tf.reduce_sum(logits, axis=sum_axis) ** 2
            right = tf.reduce_sum(labels, axis=sum_axis) ** 2
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
            deno = tf.reduce_sum(tf.clip_by_value(logits + labels, 0., 1.), axis=sum_axis)
            voe = 100 * (1. - nume / (deno + eps))

        return tf.reduce_mean(voe)

    def _metric_VD(self, logits, labels, eps=1e-5, name=None):
        """ Relative Volume Difference

        Since the measure is not symmetric, it is no metric. 
 
        Params
        ------
        reference `self._metric_dice`

        Returns
        -------
        `dice`: average dice coefficient
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

    @staticmethod
    def _metric_3D(logtis3D, labels3D, surface=False, eps=1e-6, **kwargs):
        """ Compute 3D metrics:  
        * (ASD) Average Symmetric Surface Distance
        * (RMSD) Root Mean Square Symmetric Surface Distance
        * (MSD) Maximum Symmetric Surface Distance

        Params
        ------
        `logits3D`: 3D binary prediction, shape is the same with `labels3D`
        `labels3D`: 3D labels for segmentation, shape [batch_size, None, None, None, 1]
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

        Note: `logtis3D` and `labels3D` are all the binary tensor with {0, 1}. If flag 
        `surface` is set, then we ask two input tensors represent 3D object surface, which 
        means that voxels in the surface are set to 1 while others (inside or outside the 
        surface) are set to 0. If flag is not set, then `logits3D` and `labels3D` should
        represent the whole object (solid segmentation).

        Acknowledgement: Thanks to the code snippet from @MLNotebook's blog.
        [Blog link](https://mlnotebook.github.io/post/surface-distance-function/).

        Returns
        -------
        `asd`: average asd
        """
        import math

        if surface:
            A = logtis3D
            B = labels3D
        else:
            sampling = kwargs.get("sampling", [1., 1., 1.])
            connectivity = kwargs.get("connectivity", 1)
            disc = mph.generate_binary_structure(logtis3D.ndim, connectivity)

            A = logtis3D - mph.binary_erosion(logtis3D, disc)
            B = labels3D - mph.binary_erosion(labels3D, disc)

        dist_mapA = mph.distance_transform_edt(~A, sampling)
        dist_mapB = mph.distance_transform_edt(~B, sampling)

        dist_A2B = dist_mapB[A != 0]
        dist_B2A = dist_mapA[B != 0]
        sum_A_and_B = np.sum(A) + np.sum(B)

        asd = (np.sum(dist_A2B) + np.sum(dist_B2A)) / (sum_A_and_B + eps)
        rmsd = math.sqrt((np.sum(dist_A2B ** 2) + np.sum(dist_B2A ** 2)) / (sum_A_and_B + eps))
        msd = max([np.max(dist_A2B), np.max(dist_B2A)])

        metrics_3D = {"ASD": asd, "RMSD": rmsd, "msd": msd}

        return metrics_3D

    def _add_2D_metries(self, name=None):
        logits = self._layers["binary_pred"]
        labels = self._mask
        with tf.vairable_scope(name, "Metries_2D"):
            dice = self._metric_dice(logits, labels)
            voe = self._metric_VOE(logits, labels)
            vd = self._metric_VD(logits, labels)

        self._metrics_2D["Dice"] = dice
        self._metrics_2D["VOE"] = voe
        self._metrics_2D["VD"] = vd


    def _add_losses(self, name=None):
        with tf.variable_scope(name, "Loss"):
            labels = tf.reshape(self._mask, shape=self._mask.shape[:-1])
            logits = self._layers["logits"]
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels, logits)
            self._losses["cross_entropy"] = cross_entropy

            regularization_losses = tf.add_n(tf.losses.get_regularization_losses(), "Regu")
            self._losses["total_loss"] = cross_entropy + regularization_losses

        return cross_entropy

    def create_architecture(self, mode, name=None):
        shape = (cfg.TRAIN.BS, None, None, cfg.IMG.CHANNEL)
        self._image = tf.placeholder(tf.float32, shape, name="Image")
        self._mask = tf.placeholder(tf.int32, shape, name="Mask")
        self._keep_prob = tf.placeholder(tf.float32, (1), name="KeepProb")
        self._image_summaries.append(self._image)
        self._image_summaries.append(self._mask)

        self._mode = mode

        weights_regularizer = tf.contrib.layers.l2_regularier(cfg.MODEL.WEIGHT_DECAY)
        weights_initializer = tf.truncated_normal_initializer(mean=0., stddev=0.01)
        if cfg.MODEL.USE_BIAS:
            biases_regularizer = weights_regularizer if cfg.MODEL.BIAS_DECAY else tf.no_regularizer
            biases_initializer = tf.constant_initializer(0.)
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
            prediction = self._build_network(training, name=name)
        layers_out["prediction"] = prediction

        self._add_losses()
        layers_out.update(self._losses)

        self._add_2D_metries()

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

    def get_val_summary(self, sess:tf.Session, keep_prob, data_batch):
        feed_dict = {self._image: train_batch["images"], self._mask: train_batch["labels"],
                    self._keep_prob: keep_prob}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def test2D_step(self, sess:tf.Session, test_batch, ret_image=False, keep_prob):
        feed_dict = {self._image: test_batch["images"], self._mask: test_batch["labels"],
                    self._keep_prob: keep_prob}
        
        pred = None
        if ret_image:
            fetches = [self._layers["Prediction"], self._metrics_2D["Dice"], self._metrics_2D["VOE"], self._metrics_2D["VD"]]
            pred, dice, voe, vd = sess.run(fetches, feed_dict=feed_dict)
        else:
            fetches = [self._metrics_2D["Dice"], self._metrics_2D["VOE"], self._metrics_2D["VD"]]
            dice, voe, vd = sess.run(fetches, feed_dict=feed_dict)

        return pred, dice, voe, vd

        