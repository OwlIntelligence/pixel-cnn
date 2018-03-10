import numpy as np
import os
import tensorflow as tf
import time
import data.celeba_data as celeba_data



tf.flags.DEFINE_integer("nr_mix", default_value=10, docstring="number of logistic mixture components")
tf.flags.DEFINE_integer("z_dim", default_value=50, docstring="latent dimension")
tf.flags.DEFINE_integer("batch_size", default_value=16, docstring="")
tf.flags.DEFINE_string("data_dir", default_value="/data/ziz/not-backed-up/jxu/CelebA", docstring="")

FLAGS = tf.flags.FLAGS

def int_shape(x):
    return list(map(int, x.get_shape()))

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat([x, -x], axis))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keep_dims=True))

def discretized_mix_logistic_loss(x,l,sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])


def sample_from_discretized_mix_logistic(l,nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
    return tf.concat([tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])],3)

def generative_network(z):
    with tf.variable_scope("generative_network"):
        net = tf.reshape(z, [FLAGS.batch_size, 1, 1, FLAGS.z_dim])
        net = tf.layers.conv2d_transpose(net, 2048, 4, padding='VALID')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4x2048
        net = tf.layers.conv2d_transpose(net, 1024, 5, strides=2, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8x1024
        net = tf.layers.conv2d_transpose(net, 512, 5, strides=2, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16x512
        net = tf.layers.conv2d_transpose(net, 256, 5, strides=2, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32x256
        net = tf.layers.conv2d_transpose(net, 128, 5, strides=2, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 64x64x128
        net = tf.layers.conv2d_transpose(net, FLAGS.nr_mix*10, 5, strides=2, padding='SAME') # 128x128x(10 nr_mix)
    return net

def inference_network(x):
    with tf.variable_scope("inference_network"):
        net = tf.reshape(x, [FLAGS.batch_size, 128, 128, 3]) # 128x128x3
        net = tf.layers.conv2d(net, 128, 5, strides=2, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 64x64x128
        net = tf.layers.conv2d(net, 256, 5, strides=2, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net)
        net = tf.layers.conv2d(net, 512, 5, strides=2, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net)
        net = tf.layers.conv2d(net, 1024, 5, strides=2, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net)
        net = tf.layers.conv2d(net, 2048, 5, strides=2, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net)
        net = tf.layers.conv2d(net, FLAGS.z_dim, 4, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net)
        net = tf.layers.dropout(net, 0.1)
        net = tf.reshape(net, [FLAGS.batch_size, -1])
        net = tf.layers.dense(net, FLAGS.z_dim * 2, activation=None)
        loc = net[:, :FLAGS.z_dim]
        scale = tf.nn.softplus(net[:, FLAGS.z_dim:])
    return loc, scale

def sample_z(loc, scale):
    with tf.variable_scope("sample_z"):
        dist = tf.distributions.Normal(loc=loc, scale=scale)
        z = dist.sample()
    return z

def sample_x(params):
    x_hat = sample_from_discretized_mix_logistic(params, FLAGS.nr_mix)
    return x_hat


x = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 128, 128, 3))
loc, scale = inference_network(x)
z = sample_z(loc, scale)
params = generative_network(z)

reconstruction_loss = discretized_mix_logistic_loss(x, params, False)
latent_KL = 0.5 * tf.reduce_sum(tf.square(loc) + tf.square(scale) - tf.log(tf.square(scale)) - 1,1)
loss = tf.reduce_mean(reconstruction_loss + latent_KL)

train_step = tf.train.AdamOptimizer().minimize(loss)

initializer = tf.global_variables_initializer()


train_data = celeba_data.DataLoader(args.data_dir, 'valid', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional, size=128)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(initializer)
    for data in train_data:
        feed_dict = {x: data}
        l, _ = sess.run([loss, train_step], feed_dict=feed_dict)
        print(l)
