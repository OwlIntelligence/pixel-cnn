import numpy as np
import os
import tensorflow as tf
import time
import data.celeba_data as celeba_data
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn
from utils import plotting


#tf.flags.DEFINE_integer("nr_mix", default_value=10, docstring="number of logistic mixture components")
tf.flags.DEFINE_integer("z_dim", default_value=100, docstring="latent dimension")
tf.flags.DEFINE_integer("batch_size", default_value=100, docstring="")
tf.flags.DEFINE_string("data_dir", default_value="/data/ziz/not-backed-up/jxu/CelebA", docstring="")
tf.flags.DEFINE_string("save_dir", default_value="/data/ziz/jxu/models/vae-test", docstring="")
tf.flags.DEFINE_string("data_set", default_value="celeba128", docstring="")

FLAGS = tf.flags.FLAGS

kernel_initializer = None #tf.random_normal_initializer()

def generative_network(z):
    with tf.variable_scope("generative_network"):
        net = tf.reshape(z, [-1, 1, 1, FLAGS.z_dim])

        net = tf.layers.conv2d_transpose(net, 2048, 4, strides=1, padding='VALID', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4
        net = tf.layers.conv2d_transpose(net, 1024, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8
        net = tf.layers.conv2d_transpose(net, 512, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16
        net = tf.layers.conv2d_transpose(net, 256, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32
        net = tf.layers.conv2d_transpose(net, 128, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 64x64
        net = tf.layers.conv2d_transpose(net, 64, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 128x128
        net = tf.layers.conv2d_transpose(net, 3, 1, strides=1, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.nn.sigmoid(net)
    return net

def inference_network(x):
    with tf.variable_scope("inference_network"):
        net = tf.reshape(x, [-1, 128, 128, 3]) # 128x128x3
        net = tf.layers.conv2d(net, 64, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 64x64
        net = tf.layers.conv2d(net, 128, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32
        net = tf.layers.conv2d(net, 256, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16
        net = tf.layers.conv2d(net, 512, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8
        net = tf.layers.conv2d(net, 1024, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4
        net = tf.layers.conv2d(net, 2048, 4, strides=1, padding='VALID', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 1x1
        net = tf.reshape(net, [-1, 2048])
        net = tf.layers.dense(net, FLAGS.z_dim * 2, activation=None, kernel_initializer=kernel_initializer)
        loc = net[:, :FLAGS.z_dim]
        log_var = net[:, FLAGS.z_dim:]
    return loc, log_var

def sample_z(loc, log_var):
    scale = tf.sqrt(tf.exp(log_var))
    dist = tf.distributions.Normal(loc=loc, scale=scale)
    z = dist.sample()
    return z

with tf.variable_scope("vae"):
    x = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))

    loc, log_var = inference_network(x)
    z = sample_z(loc, log_var)
    x_hat = generative_network(z)

    flatten = tf.contrib.layers.flatten
    # BCE = tf.reduce_sum(tf.keras.backend.binary_crossentropy(flatten(x), flatten(x_hat)), 1)
    MSE = tf.reduce_sum(tf.square(flatten(x)-flatten(x_hat)), 1)

    KLD = - 0.5 * tf.reduce_mean(1 + log_var - tf.square(loc) - tf.exp(log_var), axis=-1)
    #prior_scale = 1.
    #latent_KL = 0.5 * tf.reduce_sum((tf.square(loc) + tf.square(scale))/prior_scale**2 - tf.log(tf.square(scale/prior_scale)+1e-5) - 1,1)

    lam = 0.5
    beta = 250.
    loss = tf.reduce_mean( MSE + beta * tf.maximum(lam, KLD) )

    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()
