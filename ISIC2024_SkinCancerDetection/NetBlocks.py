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



import tensorflow as tf
import tensorflow_models as tfm


def create_model(heads,
                 token_num=1,
                 embeddings_size=1408,
                 learning_rate=0.07,
                 end_lr_factor=1.0,
                 dropout=0.5,
                 loss_weights=None,
                 hidden_layer_sizes=[128, 32],
                 weight_decay=0.0001,
                 seed=None) -> tf.keras.Model:
  """
  Creates linear probe or multilayer perceptron using LARS.

  """
  inputs = tf.keras.Input(shape=(token_num * embeddings_size,))
  inputs_reshape = tf.keras.layers.Reshape((token_num, embeddings_size))(inputs)
  inputs_pooled = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(inputs_reshape)
  hidden = inputs_pooled
  # If no hidden_layer_sizes are provided, model will be a linear probe.
  for size in hidden_layer_sizes:
    hidden = tf.keras.layers.Dense(
        size,
        activation='relu',
        kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(
            hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(dropout, seed=seed)(hidden)
  output = tf.keras.layers.Dense(
      units=len(heads),
      activation='sigmoid',
      kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(
          hidden)

  outputs = {}
  for i, head in enumerate(heads):
    outputs[head] = tf.keras.layers.Lambda(
        lambda x: x[..., i:i + 1], name=head.lower())(
            output)

  model = tf.keras.Model(inputs, outputs)
  model.compile(
      optimizer=tfm.optimization.lars.LARS(
         learning_rate=.1),
      loss=dict([(head, 'binary_focal_crossentropy') for head in heads]),
      loss_weights=loss_weights or 1.0,
      weighted_metrics=[
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.AUC(curve='PR', name='auc_pr')])
  return model

def plot_curve(x, y, auc, x_label=None, y_label=None, label=None):
  fig = plt.figure(figsize=(10, 10))
  plt.plot(x, y, label=f'{label} (AUC: %.3f)' % auc, color='black')
  plt.legend(loc='lower right', fontsize=18)
  plt.xlim([-0.01, 1.01])
  plt.ylim([-0.01, 1.01])
  if x_label:
    plt.xlabel(x_label, fontsize=24)
  if y_label:
    plt.ylabel(y_label, fontsize=24)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.grid(visible=True)