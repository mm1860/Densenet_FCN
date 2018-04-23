import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops

import networks
from config import cfg

class UNet(networks.Networks):
    """ U-Net implementation

    Params
    ------
    `init_channels`: a integer, number of output channels in first conv layer  
    `num_down_sample`: a integer, down(up) sample times  
    `num_conv_per_layer`: integer or list, number of conv operations in each unet layer.
    If a integer is passed in, then all the layers will use the same number, while a list
    or tuple is passed in, then it must specify the number of conv operations in each
    layer.  
    `name`: a string, name of the network  
    """
    def __init__(self, 
                 init_channels=64,
                 num_down_sample=4, 
                 num_conv_per_layer=2, 
                 name=None):
        self._name = name if name is not None else "UNet"
        self._init_channels = init_channels
        self._num_down_sample = num_down_sample

        self._num_layers = num_down_sample * 2 + 1
        if isinstance(num_conv_per_layer, int):
            self._num_conv_per_layer = [num_conv_per_layer] * self._num_layers
        elif isinstance(num_conv_per_layer, (list, tuple)):
            if len(num_conv_per_layer) == self._num_layers:
                self._num_conv_per_layer = num_conv_per_layer
            else:
                raise ValueError("length of num_conv_per_layer({:d}) isn't equal to total layers: {:d}"
                                 .format(len(num_conv_per_layer), self._num_layers))
        else:
            raise TypeError("num_conv_per_layer must be a integer or list/tuple, got {:r}"
                            .format(type(num_conv_per_layer)))
        super(UNet, self).__init__()

    def _net_arg_scope(self):
        return slim.current_arg_scope()

    def _build_network(self, is_training=True, reuse=None, name=None):
        with tf.variable_scope(name, self._name, reuse=reuse):
            tensor_out = self._image
            out_channels = self._init_channels

            # encoder
            for i in range(self._num_down_sample):
                with tf.variable_scope("Encode{:d}".format(i + 1)):
                    tensor_out = slim.repeat(tensor_out, self._num_conv_per_layer[i], slim.conv2d, out_channels, [3, 3])
                    self._layers["Encode{:d}".format(i + 1)] = tensor_out
                    tensor_out = slim.max_pool2d(tensor_out, [2, 2])
                out_channels *= 2

            # Encode-Decode-Bridge
            with tf.variable_scope("ED-Bridge"):            
                tensor_out = slim.repeat(tensor_out, self._num_conv_per_layer[self._num_down_sample], 
                                         slim.conv2d, out_channels, [3, 3])

            # decoder
            for i in reversed(range(self._num_down_sample)):
                out_channels /= 2
                with tf.variable_scope("Decode{:d}".format(i + 1)):
                    tensor_out = slim.conv2d_transpose(tensor_out, tensor_out.get_shape()[-1] // 2, [2, 2], 2,
                                                       scope="Deconv")
                    tensor_out = tf.concat((self._layers["Encode{:d}".format(i + 1)], tensor_out), axis=-1)
                    tensor_out = slim.repeat(tensor_out, self._num_conv_per_layer[-(i + 1)], slim.conv2d, out_channels, [3, 3])
                    self._layers["Decode{:d}".format(i + 1)] = tensor_out
        
        with tf.variable_scope("Tail"):
            # final
            tensor_out = slim.conv2d(tensor_out, 2, [1, 1], scope="AdjustChannels")
            self._act_summaries.append(tensor_out)
            self._layers["logits"] = tensor_out

            tensor_out = slim.softmax(tensor_out, scope="Softmax")
            # prediction
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

if __name__ == "__main__":
    # check network structure
    sess = tf.Session()
    net = UNet()

    with sess.graph.as_default():
        net.create_architecture("TRAIN")

    writer = tf.summary.FileWriter("./", sess.graph)
    writer.close()