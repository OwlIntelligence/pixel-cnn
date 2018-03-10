import numpy as np
import os
import tensorflow as tf
import time
import data.celeba_data as celeba_data
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn
from utils import plotting


#tf.flags.DEFINE_integer("nr_mix", default_value=10, docstring="number of logistic mixture components")
tf.flags.DEFINE_integer("z_dim", default_value=200, docstring="latent dimension")
tf.flags.DEFINE_integer("batch_size", default_value=50, docstring="")
tf.flags.DEFINE_string("data_dir", default_value="/data/ziz/not-backed-up/jxu/CelebA", docstring="")
tf.flags.DEFINE_string("save_dir", default_value="/data/ziz/jxu/models/vae-test", docstring="")
tf.flags.DEFINE_string("data_set", default_value="celeba64", docstring="")

FLAGS = tf.flags.FLAGS

kernel_initializer = tf.random_normal_initializer(0, 0.05)

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
        net = tf.nn.tanh(net) * (1+1e-5)
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
saver = tf.train.Saver()

train_data = celeba_data.DataLoader(FLAGS.data_dir, 'valid', FLAGS.batch_size, shuffle=True, size=64)
test_data = celeba_data.DataLoader(FLAGS.data_dir, 'valid', FLAGS.batch_size, shuffle=False, size=64)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # init
    sess.run(initializer)

    max_num_epoch = 1000
    for epoch in range(max_num_epoch):
        print("epoch:", epoch, "----------")
        train_loss_epoch = []
        for data in train_data:
            data = np.cast[np.float32]((data - 127.5) / 127.5)
            feed_dict = {x: data}
            l, _ = sess.run([loss, train_step], feed_dict=feed_dict)
            train_loss_epoch.append(l)
        train_loss_epoch = np.mean(train_loss_epoch)

        test_loss_epoch = []
        for data in test_data:
            data = np.cast[np.float32]((data - 127.5) / 127.5)
            feed_dict = {x: data}
            l = sess.run([loss], feed_dict=feed_dict)
            test_loss_epoch.append(l)
        test_loss_epoch = np.mean(test_loss_epoch)

        print("train loss:", train_loss_epoch, " test loss:", test_loss_epoch)

        if epoch % 10==0:
            saver.save(sess, FLAGS.save_dir + '/params_' + 'celeba' + '.ckpt')

            data = next(test_data)
            feed_dict = {x: data}
            sample_x, = sess.run([x_hat], feed_dict=feed_dict)
            test_data.reset()

            img_tile = plotting.img_tile(sample_x[:25], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=FLAGS.data_set + ' samples')
            plotting.plt.savefig(os.path.join(FLAGS.save_dir,'%s_vae_sample%d.png' % (FLAGS.data_set, epoch)))
