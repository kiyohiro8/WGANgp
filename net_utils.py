# -*- coding: utf-8 -*-

import os

import numpy as np

from keras.layers import Input, Dense, Activation, Flatten, Add, Lambda, Concatenate, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.engine.network import Network, Layer
from keras.initializers import TruncatedNormal
from keras.models import Model
import keras.backend as K


def set_trainable(model, prefix_list, trainable=False):
    for prefix in prefix_list:
        for layer in model.layers:
            if layer.name.startswith(prefix):
                layer.trainable = trainable
    return model


def generator(latent_dim, image_shape, num_res_blocks, base_name):
    initializer = TruncatedNormal(mean=0, stddev=0.2, seed=42)
    in_x = Input(shape=(latent_dim,))

    h, w, c = image_shape

    x = Dense(64*8*h//8*w//8, activation="relu", name=base_name+"_dense")(in_x)
    x = Reshape((h//8, w//8, -1))(x)

    x = Conv2DTranspose(64*4, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_deconv1")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(x, training=1)

    for i in range(num_res_blocks):
        x = residual_block(x, base_name=base_name, block_num=i, initializer=initializer, num_channels=64*4)

    # size//8→size//4→size//2→size
    x = Conv2DTranspose(64*2, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_deconv2")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(x, training=1)
    x = Activation("relu")(x)
    x = Conv2DTranspose(64*1, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_deconv3")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(x,training=1)
    x = Activation("relu")(x)
    out = Conv2DTranspose(3, kernel_size=7, strides=1, padding='same', activation="tanh",
                          kernel_initializer=initializer, name=base_name + "_out")(x)
    model = Model(in_x, out, name=base_name)
    return model


def discriminator(input_shape, base_name, num_res_blocks=0,is_D=True, use_res=False):
    initializer_d = TruncatedNormal(mean=0, stddev=0.1, seed=42)

    D = in_D = Input(shape=input_shape)
    D = Conv2D(64, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv1")(D)
    D = LeakyReLU(0.2)(D)
    D = Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv2")(D)
    #D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(D, training=1)
    D = LeakyReLU(0.2)(D)

    D = Conv2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv3")(D)
    #D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(D, training=1)
    D = LeakyReLU(0.2)(D)

    if use_res:
        for i in range(5):
            D = residual_block(D, base_name=base_name, block_num=i,
                               initializer=initializer_d, num_channels=256, is_D=is_D)

    D = Conv2D(512, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv4")(D)
    #D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(D, training=1)
    D = LeakyReLU(0.2)(D)
    D = Conv2D(1, kernel_size=1, strides=1, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv5")(D)


    out = Flatten()(D)
    """
    D = Dense(units=128, name=base_name + "_dense1")(D)
    D = LeakyReLU(0.2)(D)

    out = Dense(units=1, activation=None, name=base_name + "_out")(D)
    """

    model = Model(in_D, out, name=base_name)

    return model


def residual_block(x, base_name, block_num, initializer, num_channels=128,is_D=False):
    y = Conv2D(num_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False,
               name=base_name + "_resblock" + str(block_num) + "_conv1")(x)
    if not is_D:
        y = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_resblock" + str(block_num) + "_bn1")(y, training=1)
    y = Activation("relu")(y)
    y = Conv2D(num_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False,
               name=base_name + "_resblock" + str(block_num) + "_conv2")(y)
    if not is_D:
        y = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_resblock" + str(block_num) + "_bn2")(y, training=1)
    return Add()([x, y])


def gradient_penalty(D, input_merged, base_name):
    gradients = K.gradients(D, [input_merged])[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))

    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    out = Network(input=[D, input_merged], output=[gradient_penalty], name=base_name+"_gp")
    return out


def save_weights(model, path, counter, base_name=""):
    filename = base_name +str(counter) + ".hdf5"
    output_path = os.path.join(path, filename)
    model.save_weights(output_path)