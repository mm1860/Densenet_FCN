import tensorflow as tf
from tensorflow.contrib import slim as slim
from tensorflow.python.ops import array_ops
from networks import DenseNet

from config import cfg

class FCN(DenseNet):
    """ FC-DenseNet implementation
    """
    def __init__(self, init_channels, num_blocks, num_layers_per_block,
                growth_rate, bc_mode, name=None):
        self._name = name if name is not None else "FC_DenseNet"
        super(FCN, self).__init__(init_channels, num_blocks, num_layers_per_block,
                                growth_rate, bc_mode, self._name)

    
    def _transition_layer(self, tensor_in:tf.Tensor, out_channels, training=True, name=None):
        with tf.variable_scope(name, "TransitionLayer"):
            if isinstance(out_channels, float):
                out_channels = int(tensor_in.shape.as_list()[-1] * out_channels)
            tensor_out = self._unit_layer(tensor_in, out_channels, 1, "TransitionUnit", training)
            tensor_out = slim.avg_pool2d(tensor_out, [2, 2])
        
        return tensor_out

    def _build_network(self, is_training=True, reuse=None, name=None):
        with tf.variable_scope(name, self._name, reuse=reuse):
            # First convolution
            with tf.variable_scope("FirstConv"):
                first_conv = slim.conv2d(self._image, self._init_channels, [3, 3])
                first_conv = slim.batch_norm(first_conv, is_training=is_training, activation_fn=PReLU)
                first_conv = slim.max_pool2d(first_conv, [2, 2])
            self._act_summries.append(first_conv)
            
            # Dense blocks
            tensor_out = self.create_dense_layer(first_conv)

            # Deconv block
            with tf.variable_scope("Deconv"):
                tensor_out = slim.conv2d_transpose(tensor_out, 128, [2, 2], 2)
                tensor_out = self._unit_layer(tensor_out, 64, 3, "DeconvUnit1")
                tensor_out = slim.conv2d_transpose(tensor_out, 32, [2, 2], 2)
                tensor_out = self._unit_layer(tensor_out, 16, 3, "DeconvUnit2")
                tensor_out = slim.conv2d_transpose(tensor_out, 2, [2, 2], 2)
            self._act_summries.append(tensor_out)
            self._layers["logits"] = tensor_out
            
            softmax_tensor_out = slim.softmax(tensor_out)
            prediction = softmax_tensor_out[..., 1]
            self._layers["Prediction"] = prediction
            self._image_summaries.append(prediction)

            zeros = array_ops.zeros_like(prediction, dtype=tf.int32)
            ones = array_ops.ones_like(prediction, dtype=tf.int32)
            binary_pred = array_ops.where(prediction > cfg.MODEL.THRESHOLD, ones, zeros)
            self._layers["Binary_Pred"] = binary_pred

        return tensor_out
