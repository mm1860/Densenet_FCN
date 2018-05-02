import tensorflow as tf
from tensorflow.contrib import slim as slim
from tensorflow.python.ops import array_ops
from networks import Networks, prelu

from config import cfg

class FC_DenseNet(Networks):
    """ FC-DenseNet implementation

    Params
    ------
    `init_channels`:  
    `num_blocks`:  
    `num_layers_per_block`:  
    `growth_rate`:  
    `bc_mode`:  
    `name`:  
    """
    def __init__(self, init_channels, num_blocks, num_layers_per_block,
                growth_rate, bc_mode, name=None):
        self._name = name if name is not None else "FC_DenseNet"
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
        super(FC_DenseNet, self).__init__()
 
    def _net_arg_scope(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=None) as scope:
            return scope

    def _activation(self):
        if cfg.MODEL.ACTIVATION == "relu":
            activation = tf.nn.relu
        elif cfg.MODEL.ACTIVATION == "prelu":
            activation = networks.prelu
        elif cfg.MODEL.ACTIVATION == "leaky_relu":
            activation = tf.nn.leaky_relu
        else:
            raise ValueError("Unsupported activation function: %s" % cfg.MODEL.ACTIVATION)

        return activation

    def _normalization(self, tensor_in, training=True):
        if cfg.MODEL.NORMALIZATION == "batch_norm":
            # batch_norm need UPDATE_OPS
            tensor_out = slim.batch_norm(tensor_in, scale=True, is_training=training, activation_fn=self._activation())
        elif cfg.MODEL.NORMALIZATION == "instance_norm":
            tensor_out = slim.instance_norm(tensor_in, trainable=training, activation_fn=self._activation())
        elif cfg.MODEL.NORMALIZATION == "layer_norm":
            tensor_out = slim.layer_norm(tensor_in, trainable=training, activation_fn=self._activation())
        else:
            raise ValueError("Unsuppoerted normalization function: %s" % cfg.MODEL.NORMALIZATION)

        return tensor_out

    def _unit_layer(self, tensor_in:tf.Tensor, out_channels, kernel_size, name, keep_prob=1.0, training=True):
        """ A simple bn-relu-conv implementation
        """
        if isinstance(out_channels, float):
            out_channels = int(tensor_in.shape.as_list()[-1] * out_channels)
        with tf.variable_scope(name):
            tensor_out = self._normalization(tensor_in, training)
            tensor_out = slim.conv2d(tensor_out, out_channels, [kernel_size]*2)
            tensor_out = slim.dropout(tensor_out, keep_prob)
            
        return tensor_out
    
    def _internal_layer(self, tensor_in, growth_rate, training=True, bc_mode=False, scope=None):
        with tf.variable_scope(scope, "InternalLayer"):
            if bc_mode:
                bottleneck_out = self._unit_layer(tensor_in, growth_rate * 4, 1, "Bottleneck", 
                                                  keep_prob=cfg.TRAIN.KEEP_PROB, training=training)
                tensor_out = self._unit_layer(bottleneck_out, growth_rate, 3, "CompositeFunction", 
                                              keep_prob=cfg.TRAIN.KEEP_PROB, training=training)
            else:
                tensor_out = self._unit_layer(tensor_in, growth_rate, 3, "CompositeFunction", 
                                              keep_prob=cfg.TRAIN.KEEP_PROB, training=training)

            tensor_out = tf.concat((tensor_in, tensor_out), axis=-1)

        return tensor_out

    def create_dense_layer(self, tensor_in, training=True):
        tensor_out = tensor_in
        for i in range(self._num_blocks):
            with tf.variable_scope("DenseBlock%d" % (i + 1)):
                tensor_out = slim.repeat(tensor_out, self._num_layers_per_block[i], self._internal_layer,
                                        self._growth_rate, training, self._bc_mode)
                self._act_summaries.append(tensor_out)
                if cfg.MODEL.SKIP_CONNECT and i < self._num_blocks - 1:
                    self._layers["DenseBlock%d" % (i + 1)] = tensor_out
                if i < self._num_blocks - 1:
                    tensor_out = self._transition_layer(tensor_out, cfg.MODEL.THETA, training=training)
        return tensor_out

    def _transition_layer(self, tensor_in:tf.Tensor, out_channels, training=True, scope=None):
        with tf.variable_scope(scope, "TransitionLayer"):
            if isinstance(out_channels, float):
                out_channels = int(tensor_in.shape.as_list()[-1] * out_channels)
            tensor_out = self._unit_layer(tensor_in, out_channels, 1, "TransitionUnit", training=training)
            tensor_out = slim.avg_pool2d(tensor_out, [2, 2])
        
        return tensor_out

    def _build_network(self, is_training=True, reuse=None, name=None):
        with tf.variable_scope(name, self._name, reuse=reuse):
            # First convolution
            with tf.variable_scope("FirstConv"):
                first_conv = slim.conv2d(self._image, self._init_channels, [3, 3])
                first_conv = self._normalization(first_conv, is_training)
                self._layers["DenseBlock0"] = first_conv
                first_conv = slim.max_pool2d(first_conv, [2, 2])
            self._act_summaries.append(first_conv)
            
            # Dense blocks
            tensor_out = self.create_dense_layer(first_conv, is_training)

            # Deconv block
            with tf.variable_scope("Deconv"):
                channels = tensor_out.get_shape()[-1]
                for i in range(cfg.MODEL.BLOCKS):
                    tensor_out = slim.conv2d_transpose(tensor_out, channels, [2, 2], 2, scope="DeconvLayer{:d}".format(i + 1))
                    if cfg.MODEL.SKIP_CONNECT:
                        if cfg.MODEL.SKIP_CONNECT_V2 or i < self._num_blocks - 1:
                            tensor_out = tf.concat((self._layers["DenseBlock%d" % (self._num_blocks - 1 - i)], tensor_out), axis=-1)
                    channels = channels // 2
                    tensor_out = self._unit_layer(tensor_out, channels, 3, "UnitLayer{:d}".format(i + 1), 
                                                  training=is_training)
                tensor_out = self._unit_layer(tensor_out, channels, 3, "UnitLayer{:d}".format(cfg.MODEL.BLOCKS + 1),
                                              training=is_training)
                tensor_out = self._unit_layer(tensor_out, 2, 1, "UnitLayer{:d}".format(cfg.MODEL.BLOCKS + 2),
                                              training=is_training)

            self._act_summaries.append(tensor_out)
            self._layers["logits"] = tensor_out
            
            tensor_out = slim.softmax(tensor_out, scope="Softmax")
            #tensor_out = tf.sigmoid(tensor_out, name="Sigmoid")
            with tf.name_scope("Prediction"):
                _, prediction = tf.split(tensor_out, 2, -1)
                self._layers["Prediction"] = prediction
                self._image_summaries.append(prediction)
                self._act_summaries.append(prediction)

            with tf.name_scope("BinaryPred"):
                zeros = array_ops.zeros_like(prediction, dtype=tf.int32)
                ones = array_ops.ones_like(prediction, dtype=tf.int32)
                threshold = tf.constant(cfg.MODEL.THRESHOLD, dtype=tf.float32, shape=(), name="threshold")
                binary_pred = array_ops.where(prediction > threshold, ones, zeros, name="Pred2Binary")
                self._layers["Binary_Pred"] = binary_pred

        return tensor_out
