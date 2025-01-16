import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Dense, LayerNormalization, Dropout, Activation
from tensorflow.keras.initializers import Constant


class ConvNeXtBlock(tf.keras.layers.Layer):
    """
    ConvNeXt Block implemented in TensorFlow.
    From: https://arxiv.org/pdf/2201.03545v2

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super(Block, self).__init__()

        self.dwconv = DepthwiseConv2D(kernel_size=7, padding='same', depth_multiplier=1)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.pwconv1 = Dense(4 * dim)  # pointwise/1x1 convolution
        self.act = Activation('gelu')
        self.pwconv2 = Dense(dim)

        if layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                shape=(dim,),
                initializer=Constant(layer_scale_init_value),
                trainable=True,
                name='gamma'
            )
        else:
            self.gamma = None

        self.drop_path = Dropout(drop_path) if drop_path > 0.0 else tf.identity

    def call(self, inputs, training=False):
        x = inputs
        x = self.dwconv(x)

        # Permute to (N, H, W, C) -> (N, C, H, W) -> (N, H, W, C)
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = tf.transpose(x, perm=[0, 3, 1, 2])
        x = inputs + self.drop_path(x, training=training)

        return x
