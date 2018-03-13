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
tf.flags.DEFINE_integer("nr_gpu", default_value=2, docstring="number of GPUs")
tf.flags.DEFINE_string("data_dir", default_value="/data/ziz/not-backed-up/jxu/CelebA", docstring="")
tf.flags.DEFINE_string("save_dir", default_value="/data/ziz/jxu/models/vae-test", docstring="")
tf.flags.DEFINE_string("data_set", default_value="celeba128", docstring="")
tf.flags.DEFINE_boolean("load_params", default_value=False, docstring="load_parameters from save_dir?")

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

def vae_model(x, z_dim):
    with tf.variable_scope("vae_model"):
        loc, log_var = inference_network(x)
        z = sample_z(loc, log_var)
        x_hat = generative_network(z)
        return loc, log_var, z, x_hat


model_opt = {"z_dim":100}
model = tf.make_template('vae', vae_model)

##
xs = [tf.placeholder(tf.float32, shape=(None, 128, 128, 3)) for i in range(FLAGS.nr_gpu)]

model_opt = {"z_dim":100}
model = tf.make_template('vae_model', vae_model)

locs = [None for i in range(FLAGS.nr_gpu)]
log_vars = [None for i in range(FLAGS.nr_gpu)]
zs = [None for i in range(FLAGS.nr_gpu)]
x_hats = [None for i in range(FLAGS.nr_gpu)]
MSEs = [None for i in range(FLAGS.nr_gpu)]
KLDs = [None for i in range(FLAGS.nr_gpu)]
losses = [None for i in range(FLAGS.nr_gpu)]

lam = 0.5
beta = 100.
flatten = tf.contrib.layers.flatten

for i in range(FLAGS.nr_gpu):
    with tf.device('/gpu:%d' % i):
        locs[i], log_vars[i], zs[i], x_hats[i] = model(xs[i], **model_opt)
        MSEs[i] = tf.reduce_sum(tf.square(flatten(xs[i])-flatten(x_hats[i])), 1)
        KLDs[i] = - 0.5 * tf.reduce_mean(1 + log_vars[i] - tf.square(locs[i]) - tf.exp(log_vars[i]), axis=-1)
        losses[i] = tf.reduce_mean( MSEs[i] + beta * tf.maximum(lam, KLDs[i]) )

with tf.device('/gpu:%d' % 0):
    x = tf.concat(xs, axis=0)
    loc = tf.concat(locs, axis=0)
    log_var = tf.concat(log_vars, axis=0)
    z = tf.concat(zs, axis=0)
    x_hat = tf.concat(x_hats, axis=0)

MSE = tf.reduce_sum(MSEs)
KLD = tf.reduce_sum(KLDs)
loss = tf.reduce_sum(losses)

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

def make_feed_dict(data):
    data = np.cast[np.float32](data/255.)
    ds = np.split(data, FLAGS.nr_gpu)
    for i in range(FLAGS.nr_gpu):
        feed_dict = { xs[i]:ds[i] for i in range(FLAGS.nr_gpu) }
    return feed_dict



train_data = celeba_data.DataLoader(FLAGS.data_dir, 'valid', FLAGS.batch_size, shuffle=True, size=128)
test_data = celeba_data.DataLoader(FLAGS.data_dir, 'valid', FLAGS.batch_size, shuffle=False, size=128)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # init
    sess.run(initializer)

    if FLAGS.load_params:
        ckpt_file = FLAGS.save_dir + '/params_' + 'celeba' + '.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    max_num_epoch = 1000
    for epoch in range(max_num_epoch):
        tt = time.time()
        ls, mses, klds = [], [], []
        for data in train_data:
            feed_dict = make_feed_dict(data)
            l, mse, kld, _ = sess.run([loss, MSE, KLD, train_step], feed_dict=feed_dict)
            ls.append(l)
            mses.append(mse)
            klds.append(kld)
        train_loss, train_mse, train_kld = np.mean(ls), np.mean(mses), np.mean(klds)

        ls, mses, klds = [], [], []
        for data in test_data:
            feed_dict = make_feed_dict(data)
            l, mse, kld = sess.run([loss, MSE, KLD], feed_dict=feed_dict)
            ls.append(l)
            mses.append(mse)
            klds.append(kld)
        test_loss, test_mse, test_kld = np.mean(ls), np.mean(mses), np.mean(klds)

        print("epoch {0} --------------------- Time {1:.2f}s".format(epoch, time.time()-tt))
        print("train loss:{0:.3f}, train mse:{1:.3f}, train kld:{2:.3f}".format(train_loss, train_mse, train_kld))
        print("test loss:{0:.3f}, test mse:{1:.3f}, test kld:{2:.3f}".format(test_loss, test_mse, test_kld))

        if epoch % 10==0:

            saver.save(sess, FLAGS.save_dir + '/params_' + 'celeba' + '.ckpt')

            data = next(test_data)
            feed_dict = make_feed_dict(data)
            sample_x, = sess.run([x_hat], feed_dict=feed_dict)
            test_data.reset()

            img_tile = plotting.img_tile(sample_x[:25], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=FLAGS.data_set + ' samples')
            plotting.plt.savefig(os.path.join(FLAGS.save_dir,'%s_vae_sample%d.png' % (FLAGS.data_set, epoch)))
