import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops

import networks
from config import cfg

def Tiramisu103(init_channels=48):
    return Tiramisu(init_channels, 5, [4, 5, 7, 10, 12, 15], 16, True, 
                    name="Tiramisu103")

def Tiramisu64(init_channels=32):
    return Tiramisu(init_channels, 4, [4, 4, 6, 8, 12], 16, True,
                    name="Tiramisu64")

def Tiramisu56(init_channels=32):
    return Tiramisu(init_channels, 4, [4, 4, 6, 6, 8], 16, True,
                    name="Tiramisu56")

class Tiramisu(networks.Networks):
    """ Densenet for sementic segmentation

    Implementation of paper https://arxiv.org/abs/1611.09326v2:
        The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation

    Params
    ------
    `init_channels`: a integer, number of output channels in first conv layer  
    `num_blocks`: a integer, number of dense blocks in down(up) sampling  
    `num_layers_per_block`: integer or list, number of conv layers in each dense block.
    If a integer is passed in, then all dense blocks will use the same number, while a list
    or tuple is passed in, then it must specify the number of conv layers in each
    dense block. Note: because the existance of encoder-decoder-bridge, if a list is passed in,
    then length of num_layers_per_block must be equal to num_blocks + 1.  
    `growth_rate`: growth rate in dense block, please reference DenseNet structure  
    `bc_mode`: use BC mode or not, please reference DenseNet structure  
    `name`: a string, name of the network  
    """
    def __init__(self, init_channels, 
                       num_blocks, 
                       num_layers_per_block,
                       growth_rate, 
                       bc_mode, 
                       name=None):
        self._name = name if name is not None else "Tiramisu"
        self._init_channels = init_channels
        self._num_blocks = num_blocks
        if isinstance(num_layers_per_block, int):
            self._num_layers = [num_layers_per_block] * (self._num_blocks + 1)
        elif isinstance(num_layers_per_block, list):
            if len(num_layers_per_block) == self._num_blocks + 1:
                self._num_layers = num_layers_per_block
            else:
                raise ValueError("Length of num_layers_per_block is {:d}, but expect {:d}"
                                 .format(len(num_layers_per_block), self._num_blocks + 1))
        else:
            raise TypeError("Error type for `num_layers_per_block`")
        self._growth_rate = growth_rate
        self._bc_mode = bc_mode
        super(Tiramisu, self).__init__()
 
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

    def _unit_layer(self, tensor_in:tf.Tensor, out_channels, kernel_size, name, keep_prob=1.0, training=True):
        """ A simple bn-relu-conv implementation
        """
        if isinstance(out_channels, float):
            out_channels = int(tensor_in.shape.as_list()[-1] * out_channels)
        with tf.variable_scope(name):
            # batch_norm need UPDATE_OPS
            tensor_out = slim.batch_norm(tensor_in, scale=True, is_training=training, activation_fn=self._activation())
            tensor_out = slim.conv2d(tensor_out, out_channels, [kernel_size]*2)
            if cfg.UDN.USE_DROPOUT:
                tensor_out = slim.dropout(tensor_out, keep_prob)
            
        return tensor_out
    
    def _internal_layer(self, tensor_in, growth_rate, training=True, bc_mode=True, concat=False, scope=None):
        with tf.variable_scope(scope, "InternalLayer"):
            if bc_mode:
                bottleneck_out = self._unit_layer(tensor_in, growth_rate * 4, 1, "Bottleneck", 
                                                  keep_prob=cfg.TRAIN.KEEP_PROB, training=training)
                tensor_out = self._unit_layer(bottleneck_out, growth_rate, 3, "CompositeFunction", 
                                              keep_prob=cfg.TRAIN.KEEP_PROB, training=training)
            else:
                tensor_out = self._unit_layer(tensor_in, growth_rate, 3, "CompositeFunction", 
                                              keep_prob=cfg.TRAIN.KEEP_PROB, training=training)

            if concat:
                tensor_out = tf.concat((tensor_in, tensor_out), axis=-1)

        return tensor_out

    def _dense_block(self, tensor_in, growth_rate, n_layers, training=True, bc_mode=True, upsample=False, scope=None):
        tensor_out = tensor_in
        if not upsample:
            tensor_out = slim.repeat(tensor_out, n_layers, self._internal_layer, growth_rate, training, bc_mode, 
                                        concat=True, scope=scope or "DenseBlock")
        else:
            with tf.variable_scope(scope, "DenseBlock"):
                tensor_new = []
                for i in range(n_layers):
                    tensor_out = self._internal_layer(tensor_in, growth_rate, training, bc_mode, 
                                                      concat=False, scope="InternalLayer_{:d}".format(i + 1))
                    tensor_in = tf.concat((tensor_in, tensor_out), axis=-1)
                    tensor_new.append(tensor_out)
                tensor_out = tf.concat(tensor_new, axis=-1)
    
        return tensor_out

    def _transition_down(self, tensor_in, out_channels, training=True, scope=None):
        with tf.variable_scope(scope, "TransDown"):
            if isinstance(out_channels, float):
                out_channels = int(tensor_in.shape[-1] * out_channels)
            tensor_out = self._unit_layer(tensor_in, out_channels, 1, "TransitionUnit", training=training)
            tensor_out = slim.avg_pool2d(tensor_out, [2, 2])
        
        return tensor_out
    
    def _transition_up(self, tensor_in, out_channels, tensor_skip, training=True, scope=None):
        with tf.variable_scope(scope, "TransUp"):
            if isinstance(out_channels, float):
                out_channels = int(tensor_in.shape[-1] * out_channels)
            tensor_out = slim.conv2d_transpose(tensor_in, out_channels, [3, 3], 2)
            tensor_out = tf.concat((tensor_skip, tensor_out), axis=-1)

        return tensor_out

    def _build_network(self, is_training=True, reuse=None, name=None):
        with tf.variable_scope(name, self._name, reuse=reuse):
            # First convolution
            tensor_out = slim.conv2d(self._image, self._init_channels, [3, 3], scope="FirstConv")
            self._act_summaries.append(tensor_out)
            self._layers["FirstConv"] = tensor_out
            
            # Encoder
            for i in range(self._num_blocks):
                with tf.variable_scope("Encode{:d}".format(i + 1)):
                    tensor_out = self._dense_block(tensor_out, self._growth_rate, self._num_layers[i],
                                                   training=is_training, bc_mode=self._bc_mode, upsample=False)
                    self._layers["Encode{:d}".format(i + 1)] = tensor_out
                    tensor_out = self._transition_down(tensor_out, cfg.UDN.THETA, is_training)

            # Encode-Decode-Bridge
            tensor_out = self._dense_block(tensor_out, self._growth_rate, self._num_layers[-1], 
                                           training=is_training, bc_mode=self._bc_mode, upsample=True,
                                           scope="ED-Bridge")
            self._layers["ED-Bridge"] = tensor_out

            # Decoder
            for i in reversed(range(self._num_blocks)):
                with tf.variable_scope("Decode{:d}".format(i + 1)):
                    tensor_skip = self._layers["Encode{:d}".format(i + 1)]
                    tensor_out = self._transition_up(tensor_out, tensor_out.shape[-1], tensor_skip, training=is_training)
                    tensor_out = self._dense_block(tensor_out, self._growth_rate, self._num_layers[i], 
                                                   training=is_training, bc_mode=self._bc_mode, 
                                                   upsample=True if i != 0 else False)
                    self._layers["Decode{:d}".format(i + 1)] = tensor_out

            # Final convolution
            with tf.variable_scope("Tail"):
                tensor_out = slim.conv2d(tensor_out, 2, [1, 1], scope="FinalConv")
                self._act_summaries.append(tensor_out)
                self._layers["logits"] = tensor_out
                tensor_out = slim.softmax(tensor_out, scope="Softmax")
                
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

    def __repr__(self):
        cur_channels = self._init_channels
        layer_name = ["Conv3x3, {:d}".format(cur_channels)]
        skip_channels = []

        for i in range(self._num_blocks):
            cur_channels += self._growth_rate * self._num_layers[i]
            skip_channels.append(cur_channels)
            layer_name.append("DB({:d}) + TD, {:d}".format(self._num_layers[i], cur_channels))

        cur_channels += self._growth_rate * self._num_layers[-1]        
        layer_name.append("DB({:d}), {:d}".format(self._num_layers[-1], cur_channels))

        for i in reversed(range(self._num_blocks)):
            cur_channels = skip_channels[i] + (self._num_layers[i + 1] + self._num_layers[i]) * self._growth_rate
            if i != 0:
                layer_name.append("TU + DBN({:d}), {:d}".format(self._num_layers[i], cur_channels))
            else:
                layer_name.append("TU + DB({:d}), {:d}".format(self._num_layers[i], cur_channels))

        layer_name.append("Conv1x1, 2")

        return "\n".join(layer_name)

if __name__ == "__main__":
    # check network architecture
    net = Tiramisu103()
    sess = tf.Session()
    with sess.graph.as_default():
        net.create_architecture("TRAIN")
    print(net)
    writer = tf.summary.FileWriter("./", sess.graph)
    writer.close()