import numpy as np
import tensorflow as tf


class CustomConvNet(tf.keras.Model):
    """Custom model."""

    def __init__(self, input_shape, activation_func, initializer, dense_initializer, pos_neg_ratio=0.5, seed=476):

        super(CustomConvNet, self).__init__()

        self.conv0 = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7),
                                            input_shape=input_shape,
                                            padding="same",
                                            # activation=activation_func,
                                            data_format="channels_last",  # def
                                            kernel_initializer=initializer,
                                            )
        # TODO: 4x4 kernel with stride 4
        self.batch_norm0 = tf.keras.layers.BatchNormalization()
        self.activation0 = tf.keras.layers.Activation(activation_func)
        # self.pool0 = (tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                            # input_shape=(batch_size, 384, 384, 3),
                                            padding="same",
                                            use_bias=False,
                                            # activation=activation_func,
                                            data_format="channels_last",  # def
                                            kernel_initializer=initializer,
                                            )
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                            # activation=activation_func,
                                            padding="same",
                                            use_bias=False,
                                            # strides=(2, 2)
                                            kernel_initializer=initializer,
                                            )
        self.conv25 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                             activation=activation_func,
                                             padding="same",
                                             # strides=(2, 2)
                                             kernel_initializer=initializer,
                                             )
        self.pool1 = (tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                            # activation=activation_func,
                                            # strides=(2, 2),
                                            use_bias=False,
                                            kernel_initializer=initializer,
                                            )
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                            # activation=activation_func,
                                            padding="same",
                                            use_bias=False,
                                            kernel_initializer=initializer,
                                            )
        self.conv45 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                             activation=activation_func,
                                             padding="same",
                                             kernel_initializer=initializer,
                                             )
        self.pool2 = (tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                            # activation=activation_func,
                                            padding="same",
                                            use_bias=False,
                                            kernel_initializer=initializer,
                                            )

        self.conv55 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                             activation=activation_func,
                                             # bias=False,
                                             padding="same",
                                             kernel_initializer=initializer,
                                             )
        self.pool3 = (tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                            # activation=activation_func,
                                            padding="same",
                                            use_bias=False,
                                            kernel_initializer=initializer,
                                            )
        self.conv65 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                             # activation=activation_func,
                                             padding="same",
                                             use_bias=False,
                                             kernel_initializer=initializer,
                                             )
        self.conv67 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                             activation=activation_func,
                                             padding="same",
                                             kernel_initializer=initializer,
                                             )

        self.pool4 = (tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.conv7 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                            # activation=activation_func,
                                            padding="same",
                                            use_bias=False,
                                            kernel_initializer=initializer, )
        self.conv8 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                            # activation=activation_func,
                                            padding="same",
                                            use_bias=False,
                                            kernel_initializer=initializer,
                                            )
        self.conv85 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                             activation=activation_func,
                                             padding="same",
                                             # use_bias=False,
                                             kernel_initializer=initializer,
                                             )

        self.pool5 = (tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.conv9 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                            # activation=activation_func,
                                            kernel_initializer=initializer,
                                            padding="same",
                                            use_bias=False,
                                            # kernel_regularizer=tf.keras.regularizers.L1L2(0.001, 0.001),
                                            # activity_regularizer=tf.keras.regularizers.L1L2(0.001, 0.001)
                                            )

        self.conv10 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                             activation=activation_func,
                                             kernel_initializer=initializer,
                                             padding="same",
                                             # kernel_regularizer=tf.keras.regularizers.L1L2(0.001, 0.001),
                                             # activity_regularizer=tf.keras.regularizers.L1L2(0.001, 0.001)
                                             )
        self.pool6 = (tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.conv11 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                             activation=activation_func,
                                             kernel_initializer=initializer,
                                             use_bias=False,
                                             kernel_regularizer=tf.keras.regularizers.L1L2(0.001, 0.001),
                                             activity_regularizer=tf.keras.regularizers.L1L2(0.001, 0.001)
                                             )

        # self.conv12 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
        #                                     activation=activation_func,
        #                                     kernel_initializer=initializer,
        # )
        # self.pool7 = (tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.flatten = tf.keras.layers.Flatten()

        # self.feature_extractor = []
        # for i in range(5):
        #     self.hidden_layers.append(tf.keras.layers.Dense(2048, activation=activation_func))
        #     self.hidden_layers.append(tf.keras.layers.Dropout(0.2))

        self.hidden_layers = []
        for _ in range(2):
            self.hidden_layers.append(tf.keras.layers.Dense(1024,
                                                            activation=activation_func,
                                                            kernel_initializer=dense_initializer,
                                                            kernel_regularizer=tf.keras.regularizers.L1L2(0.001, 0.001),
                                                            activity_regularizer=tf.keras.regularizers.L1L2(0.001,
                                                                                                            0.001),
                                                            )
                                      )
            self.hidden_layers.append(tf.keras.layers.Dropout(0.1, seed=seed))
            self.hidden_layers.append(tf.keras.layers.BatchNormalization())

        self.fc = []
        for _ in range(1):
            self.fc.append(tf.keras.layers.Dense(512,
                                                 activation=activation_func,
                                                 kernel_initializer=dense_initializer,
                                                 kernel_regularizer=tf.keras.regularizers.L1L2(0.001, 0.001),
                                                 activity_regularizer=tf.keras.regularizers.L1L2(0.001, 0.001),
                                                 )
                           )
            self.fc.append(tf.keras.layers.BatchNormalization())

        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid',
                                                  # kernel_initializer=dense_initializer,
                                                  bias_initializer=tf.keras.initializers.Constant(
                                                      np.log([pos_neg_ratio]))
                                                  )

        self.bach_norm_layers = [tf.keras.layers.BatchNormalization() for _ in range(17)]
        # self.gauss_noise_layers = [tf.keras.layers.GaussianNoise(0.2, seed=seed) for _ in range(4)]
        self.spatial_dropout_layers = [tf.keras.layers.SpatialDropout2D(0.1, seed=seed) for _ in range(1)]

    def call(self, x):
        x = self.conv0(x)
        x = self.spatial_dropout_layers[0](x)
        x = self.batch_norm0(x)
        x = self.activation0(x)

        x = self.conv1(x)
        x = self.bach_norm_layers[0](x)
        x = self.conv2(x)
        # x = self.spatial_dropout_layers[1](x)
        x = self.bach_norm_layers[1](x)
        x = self.conv25(x)
        # x = self.spatial_dropout_layers[2](x)
        x = self.bach_norm_layers[2](x)
        x = self.pool1(x)

        x = self.conv3(x)
        # x = self.spatial_dropout_layers[3](x)
        x = self.bach_norm_layers[3](x)
        x = self.conv4(x)
        x = self.bach_norm_layers[4](x)
        x = self.conv45(x)
        x = self.bach_norm_layers[5](x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.bach_norm_layers[6](x)
        x = self.conv55(x)
        x = self.bach_norm_layers[7](x)
        x = self.pool3(x)

        x = self.conv6(x)
        x = self.bach_norm_layers[8](x)
        x = self.conv65(x)
        x = self.bach_norm_layers[9](x)
        x = self.conv67(x)
        x = self.bach_norm_layers[10](x)
        x = self.pool4(x)

        x = self.conv7(x)
        x = self.bach_norm_layers[11](x)
        x = self.conv8(x)
        x = self.bach_norm_layers[12](x)
        x = self.conv85(x)
        x = self.bach_norm_layers[13](x)
        x = self.pool5(x)

        x = self.conv9(x)
        x = self.bach_norm_layers[14](x)
        x = self.conv10(x)
        x = self.bach_norm_layers[15](x)
        x = self.pool6(x)

        x = self.conv11(x)
        x = self.bach_norm_layers[16](x)
        # x = self.conv12(x)
        # x = self.pool7(x)

        x = self.flatten(x)

        # for layer in self.feature_extractor:
        #     x = layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        for layer in self.fc:
            x = layer(x)

        return self.output_layer(x)
