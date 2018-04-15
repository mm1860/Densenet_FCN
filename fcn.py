import tensorflow as tf
from tensorflow.contrib import slim as slim
from tensorflow.python.ops import array_ops
from networks import DenseNet, prelu

from config import cfg

class FCN(DenseNet):
    """ FC-DenseNet implementation
    """
    def __init__(self, init_channels, num_blocks, num_layers_per_block,
                growth_rate, bc_mode, name=None):
        self._name = name if name is not None else "FC_DenseNet"
        super(FCN, self).__init__(init_channels, num_blocks, num_layers_per_block,
                                growth_rate, bc_mode)

    
    def _transition_layer(self, tensor_in:tf.Tensor, out_channels, training=True, scope=None):
        with tf.variable_scope(scope, "TransitionLayer"):
            if isinstance(out_channels, float):
                out_channels = int(tensor_in.shape.as_list()[-1] * out_channels)
            tensor_out = self._unit_layer(tensor_in, out_channels, 1, "TransitionUnit", training)
            tensor_out = slim.avg_pool2d(tensor_out, [2, 2])
        
        return tensor_out

    def _build_network(self, is_training=True, reuse=None, name=None):
        with tf.variable_scope(name, self._name, reuse=reuse):
            # First convolution
            with tf.variable_scope("FirstConv"):
                if cfg.MODEL.ACTIVATION == "relu":
                    activation = tf.nn.relu
                elif cfg.MODEL.ACTIVATION == "prelu":
                    activation = prelu
                elif cfg.MODEL.ACTIVATION == "leaky_relu":
                    activation = tf.nn.leaky_relu
                else:
                    raise ValueError("Unsupported activation function: %s" % cfg.MODEL.ACTIVATION)

                first_conv = slim.conv2d(self._image, self._init_channels, [3, 3])
                first_conv = slim.batch_norm(first_conv, scale=True, is_training=is_training, activation_fn=activation)
                first_conv = slim.max_pool2d(first_conv, [2, 2])
            self._act_summaries.append(first_conv)
            
            # Dense blocks
            tensor_out = self.create_dense_layer(first_conv, is_training)

            # Deconv block
            with tf.variable_scope("Deconv"):
                tensor_out = slim.conv2d_transpose(tensor_out, 128, [2, 2], 2)
                tensor_out = self._unit_layer(tensor_out, 64, 3, "DeconvUnit1")
                tensor_out = slim.conv2d_transpose(tensor_out, 32, [2, 2], 2)
                tensor_out = self._unit_layer(tensor_out, 16, 3, "DeconvUnit2")
                tensor_out = slim.conv2d_transpose(tensor_out, 2, [2, 2], 2)
            self._act_summaries.append(tensor_out)
            self._layers["logits"] = tensor_out
            
            softmax_tensor_out = slim.softmax(tensor_out)
            with tf.name_scope("Prediction"):
                _, prediction = tf.split(softmax_tensor_out, 2, -1)
                self._layers["Prediction"] = prediction
                self._image_summaries.append(prediction)

            with tf.name_scope("BinaryPred"):
                zeros = array_ops.zeros_like(prediction, dtype=tf.int32)
                ones = array_ops.ones_like(prediction, dtype=tf.int32)
                threshold = tf.constant(cfg.MODEL.THRESHOLD, dtype=tf.float32, shape=(), name="threshold")
                binary_pred = array_ops.where(prediction > threshold, ones, zeros, name="Pred2Binary")
                self._layers["Binary_Pred"] = binary_pred

        return tensor_out
