import numpy as np
import os
import tensorflow as tf
import time
import data.celeba_data as celeba_data
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn


#tf.flags.DEFINE_integer("nr_mix", default_value=10, docstring="number of logistic mixture components")
tf.flags.DEFINE_integer("z_dim", default_value=500, docstring="latent dimension")
tf.flags.DEFINE_integer("batch_size", default_value=50, docstring="")
tf.flags.DEFINE_string("data_dir", default_value="/data/ziz/not-backed-up/jxu/CelebA", docstring="")

FLAGS = tf.flags.FLAGS

# def generative_network(z, init=False, ema=None, dropout_p=0.0):
#     counters = {}
#     with arg_scope([nn.conv2d, nn.deconv2d, nn.dense], counters=counters, init=init, ema=ema, dropout_p=dropout_p, nonlinearity=tf.nn.elu):
#         net = tf.reshape(z, [FLAGS.batch_size, 1, 1, FLAGS.z_dim])
#         net = nn.deconv2d(net, 512, filter_size=[4,4], stride=[1,1], pad='VALID')
#         net = nn.deconv2d(net, 256, filter_size=[5,5], stride=[2,2], pad='SAME')
#         net = nn.deconv2d(net, 128, filter_size=[5,5], stride=[2,2], pad='SAME')
#         net = nn.deconv2d(net, 64, filter_size=[5,5], stride=[2,2], pad='SAME')
#         net = nn.deconv2d(net, 32, filter_size=[5,5], stride=[2,2], pad='SAME')
#         net = nn.deconv2d(net, 3, filter_size=[1,1], stride=[1,1], pad='SAME', nonlinearity=tf.nn.tanh)
#         return net
#
# def inference_network(x, init=False, ema=None, dropout_p=0.0):
#     counters = {}
#     with arg_scope([nn.conv2d, nn.deconv2d, nn.dense], counters=counters, init=init, ema=ema, dropout_p=dropout_p, nonlinearity=tf.nn.elu):
#         net = tf.reshape(x, [FLAGS.batch_size, 64, 64, 3])
#         net = nn.conv2d(net, 32, filter_size=[5,5], stride=[2,2], pad='SAME')
#         net = nn.conv2d(net, 64, filter_size=[5,5], stride=[2,2], pad='SAME')
#         net = nn.conv2d(net, 128, filter_size=[5,5], stride=[2,2], pad='SAME')
#         net = nn.conv2d(net, 256, filter_size=[5,5], stride=[2,2], pad='SAME')
#         net = nn.conv2d(net, 512, filter_size=[4,4], stride=[1,1], pad='VALID')
#         net = tf.reshape(net, [FLAGS.batch_size, -1])
#         net = nn.dense(net, FLAGS.z_dim * 2)
#         loc = net[:, :FLAGS.z_dim]
#         scale = tf.nn.softplus(net[:, FLAGS.z_dim:])
#         return loc, scale

def generative_network(z):
    with tf.variable_scope("generative_network"):
        net = tf.reshape(z, [FLAGS.batch_size, 1, 1, FLAGS.z_dim])

        net = tf.layers.conv2d_transpose(net, 1024, 4, strides=1, padding='VALID', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4
        net = tf.layers.conv2d_transpose(net, 512, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8
        net = tf.layers.conv2d_transpose(net, 256, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16
        net = tf.layers.conv2d_transpose(net, 128, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32
        net = tf.layers.conv2d_transpose(net, 64, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 64x64
        net = tf.layers.conv2d_transpose(net, 3, 1, strides=1, padding='SAME', kernel_initializer=kernel_initializer)
    return net

def inference_network(x):
    with tf.variable_scope("inference_network"):
        net = tf.reshape(x, [FLAGS.batch_size, 64, 64, 3]) # 64x64x3
        net = tf.layers.conv2d(net, 64, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32
        net = tf.layers.conv2d(net, 128, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16
        net = tf.layers.conv2d(net, 256, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8
        net = tf.layers.conv2d(net, 512, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4
        net = tf.layers.conv2d(net, 1024, 4, strides=1, padding='VALID', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 1x1
        net = tf.reshape(net, [FLAGS.batch_size, -1])
        net = tf.layers.dense(net, FLAGS.z_dim * 2, activation=None, kernel_initializer=kernel_initializer)
        loc = net[:, :FLAGS.z_dim]
        scale = tf.nn.softplus(net[:, FLAGS.z_dim:])
    return loc, scale

def sample_z(loc, scale):
    dist = tf.distributions.Normal(loc=loc, scale=scale)
    z = dist.sample()
    return z


# def sample_x(params):
#     x_hat = nn.sample_from_discretized_mix_logistic(params, FLAGS.nr_mix)
#     return x_hat


x = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 64, 64, 3))

loc, scale = inference_network(x)
z = sample_z(loc, scale)
x_hat = generative_network(z)

reconstruction_loss = tf.reduce_mean(tf.square(x_hat - x), [1,2,3])

latent_KL = 0.5 * tf.reduce_sum(tf.square(loc) + tf.square(scale) - tf.log(tf.square(scale)) - 1,1)
loss = tf.reduce_mean(reconstruction_loss + latent_KL)

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

initializer = tf.global_variables_initializer()


train_data = celeba_data.DataLoader(FLAGS.data_dir, 'valid', FLAGS.batch_size, shuffle=True, size=64)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # init
    sess.run(initializer)

    num_epoch = 100
    for i in range(num_epoch):
        loss_epoch = []
        print(i, "----------")
        for data in train_data:
            data = np.cast[np.float32]((data - 127.5) / 127.5)
            feed_dict = {x: data}
            l, _ = sess.run([loss, train_step], feed_dict=feed_dict)
            loss_epoch.append(l)
        l = np.mean(loss_epoch)
        print("loss", l)
