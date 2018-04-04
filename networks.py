import tensorflow as tf
import tensorflow.contrib.slim as slim

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

        self._custom_summries = {}
        self._act_summaries = []

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
        # trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram('Trainable/' + var.op.name, var)

        # denseblock output
        for out in self._act_summaries:
            tf.summary.histogram('Activation/' + out.op.name, out)

        # custom summries
        for key in self._custom_summries:
            tf.summary.histogram('Custom/' + key, self._custom_summries[key])

        # add image summaries
        tf.summary.image("Image/" + self._image.op.name, self._image)
        tf.summary.image("Image/" + self._mask.op.name, self._mask)

    def _metric_dice(self, logits, labels, eps=1e-5):
        """ Dice coefficient
        """
        logits = tf.cast(logits, tf.bool)
        labels = tf.cast(labels, tf.bool)
        
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(logits, labels), tf.float32), axis=[1,2,3])
        left = tf.reduce_sum(tf.cast(logits * logits, tf.float32), axis=[1,2,3])
        right = tf.reduce_sum(tf.cast(labels * labels, tf.float32), axis=[1,2,3])
        dice = (2 * intersection) / (left + right + eps)

        return dice

    def _metric_VOE(self, logits, labels, eps=1e-5):
        """ Volumetric Overlap Error

        numerator / denominator
        """
        logits = tf.cast(logits, tf.bool)
        labels = tf.cast(labels, tf.bool)

        nume = tf.reduce_sum(tf.cast(tf.logical_and(logits, labels), tf.float32), axis=[1,2,3])
        deno = tf.reduce_sum(tf.cast(tf.logical_or(logits, labels), tf.float32), axis=[1,2,3])
        voe = 100 * (1 - nume / (deno + eps))

        return voe

    def _metric_VD(self, logits, labels, eps=1e-5):
        """ Relative Volume Difference

        Since the measure is not symmetric, it is no metric. 
        """
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)

        A = tf.reduce_sum(logits, axis=[1,2,3])
        B = tf.reduce_sum(labels, axis=[1,2,3])
        vd = 100 * (tf.abs(A - B) / (B + eps))

        return vd

    def _metric_ASD(self, logtis, labels, eps=1e-5)
        """ Average Symmetric Surface Distance
        """
        raise NotImplementedError

    def _metric_RMSD(self, logits, labels, eps=1e-5):
        """ Root Mean Square Symmetric Surface Distance
        """
        raise NotImplementedError

    def _metric_MSD(self, logits, labels, eps=1e-5):
        """ Maximum Symmetric Surface Distance
        """
        raise NotImplementedError

    def _add_losses(self):
        pass

    def create_architecture(self, mode):
        shape = (cfg.TRAIN.BS, None, None, cfg.IMG.CHANNEL)
        self._image = tf.placeholder(tf.float32, shape, name="Image")
        self._mask = tf.placeholder(tf.float32, shape, name="Mask")
        self._keep_prob = tf.placeholder(tf.float32, (1), name="KeepProb")

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

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=weights_regularizer,
                            weights_initializer=weights_initializer,
                            biases_regularizer=biases_regularizer,
                            biases_initializer=biases_initializer):
            prediction = self._build_network(training, name="DenseFCN")

        self._add_summaries()







        