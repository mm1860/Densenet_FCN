import tensorflow as tf
import tensorflow.contrib.slim as slim
from networks import DenseNet

class FCN(DenseNet):
    def __init__(self, init_channels, num_blocks, num_layers_per_block,
                growth_rate, bc_mode, name=None):
        self._name = name
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
            self._custom_summries["FirstConv"] = first_conv
            
            # Dense blocks
            tensor_out = self.create_dense_layer(first_conv)

            # Deconv block
            tensor_out = slim.conv2d_transpose(tensor_out, 128, [2, 2], 2)
            tensor_out = self._unit_layer(tensor_out, 64, 3, "DeconvUnit1")
            tensor_out = slim.conv2d_transpose(tensor_out, 32, [2, 2], 2)
            tensor_out = self._unit_layer(tensor_out, 16, 3, "DeconvUnit2")
            tensor_out = slim.conv2d_transpose(tensor_out, 2, [2, 2], 2)
            self._custom_summries["Prediction"] = tensor_out

        return tensor_out
