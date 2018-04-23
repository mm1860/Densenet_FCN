import tensorflow as tf
import tensorflow.contrib.slim as slim

import networks

class UNet(networks.Networks):
    """ U-Net implementation

    Params
    ------
    """
    def __init__(self)