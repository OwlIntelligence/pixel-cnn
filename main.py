"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr_gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting
import utils.mask as um
import utils.mfunc as uf

# self define modules
from configs import config_args, configs

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/local_home/tim/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/local_home/tim/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-sc', '--spatial_conditional', dest='spatial_conditional', action='store_true', help='Condition on spatial latent codes?')
parser.add_argument('-gc', '--global_conditional', dest='global_conditional', action='store_true', help='Condition on global latent codes?')
parser.add_argument('-ms', '--map_sampling', dest='map_sampling', action='store_true', help='use MAP sampling?')
parser.add_argument('-cn', '--config_name', type=str, default='None', help='what is the config name?')
parser.add_argument('-ed', '--energy_distance', dest='energy_distance', action='store_true', help='use energy distance in place of likelihood')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
#config_args(args, configs['cifar'])
config_args(args, configs[args.config_name])
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
exp_label = "celeba64-mouth-e0.3"

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# energy distance or maximum likelihood?
if args.energy_distance:
    loss_fun = nn.energy_distance
else:
    loss_fun = nn.discretized_mix_logistic_loss

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
if args.data_set == 'cifar':
    import data.cifar10_data as cifar10_data
    DataLoader = cifar10_data.DataLoader
elif args.data_set == 'imagenet':
    import data.imagenet_data as imagenet_data
    DataLoader = imagenet_data.DataLoader
elif args.data_set == 'celeba64':
    import data.celeba_data as celeba_data
    DataLoader = celeba_data.DataLoader
else:
    raise("unsupported dataset")
# train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
obs_shape = test_data.get_observation_size() # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]
gh_init, sh_init = None, None
ghs, shs, gh_sample, sh_sample = [[None] * args.nr_gpu for i in range(4)]

if args.global_conditional:
    latent_dim = num_labels
    gh_init = tf.placeholder(tf.int32, shape=(args.init_batch_size, latent_dim))
    ghs = [tf.placeholder(tf.int32, shape=(args.batch_size, latent_dim)) for i in range(args.nr_gpu)]
if args.spatial_conditional:
    latent_shape = obs_shape[0], obs_shape[1], obs_shape[2]+1 ##
    sh_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + latent_shape)
    shs = [tf.placeholder(tf.float32, shape=(args.batch_size,) + latent_shape ) for i in range(args.nr_gpu)]
    sh_sample = shs




# if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on

# if args.class_conditional:
#     num_labels = train_data.get_num_labels()
#     y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
#     h_init = tf.one_hot(y_init, num_labels)
#     y_sample = np.split(np.mod(np.arange(args.batch_size*args.nr_gpu), num_labels), args.nr_gpu)
#     h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
#     ys = [tf.placeholder(tf.int32, shape=(args.batch_size,)) for i in range(args.nr_gpu)]
#     hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
# elif args.spatial_conditional:
#     lat_shape = obs_shape
#     hs = [tf.placeholder(tf.float32, shape=(args.batch_size,) + lat_shape ) for i in range(args.nr_gpu)]
#     h_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
#     h_sample = [None] * args.nr_gpu
# else:
#     h_init = None
#     h_sample = [None] * args.nr_gpu
#     hs = h_sample


# create the model
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance, 'global_conditional':args.global_conditional, 'spatial_conditional':args.spatial_conditional}
model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
init_pass = model(x_init, gh_init, sh_init, init=True, dropout_p=args.dropout_p, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))
ema_params = [ema.average(p) for p in all_params]

# get loss gradients over multiple GPUs + sampling
grads = []
loss_gen = []
loss_gen_test = []
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # train
        out = model(xs[i], ghs[i], shs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(loss_fun(tf.stop_gradient(xs[i]), out, masks=shs[i][:, :, :, -1]))

        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params, colocate_gradients_with_ops=True))

        # test
        out = model(xs[i], ghs[i], shs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(loss_fun(xs[i], out, masks=shs[i][:, :, :, -1]))

        # sample
        out = model(xs[i], gh_sample[i], sh_sample[i], ema=ema, dropout_p=0, **model_opt)
        if args.energy_distance:
            new_x_gen.append(out[0])
        else:
            if args.map_sampling:
                epsilon = 0.5 - 1e-5
            else:
                epsilon = 1e-5
            epsilon = 0.3
            new_x_gen.append(nn.sample_from_discretized_mix_logistic(out, args.nr_logistic_mix, epsilon=epsilon))

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = tf.group(nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)

# mask generator
train_mgen = um.RandomRectangleMaskGenerator(obs_shape[0], obs_shape[1])
#test_mgen = um.CenterMaskGenerator(obs_shape[0], obs_shape[1])
#test_mgen = um.RectangleMaskGenerator(obs_shape[0], obs_shape[1], (28, 62, 38, 2))
test_mgen = um.RectangleMaskGenerator(obs_shape[0], obs_shape[1], (54, 52, 64, 12))

# sample from the model
def sample_from_model(sess, data=None):
    if data is not None and type(data) is not tuple:
        x = data
    x = np.cast[np.float32]((x - 127.5) / 127.5)
    x = np.split(x, args.nr_gpu)
    h = [x[i].copy() for i in range(args.nr_gpu)]
    for i in range(args.nr_gpu):
        h[i] = uf.mask_inputs(h[i], test_mgen)
    feed_dict = {shs[i]: h[i] for i in range(args.nr_gpu)}
    #x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
    x_gen = [h[i][:,:,:,:3].copy() for i in range(args.nr_gpu)]
    m_gen = [h[i][:,:,:,-1].copy() for i in range(args.nr_gpu)]
    #assert m_gen[0]==m_gen[-1], "we currently assume all masks are the same during sampling"
    m_gen = m_gen[0][0]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            if m_gen[yi,xi] == 0:
                print((yi, xi))
                feed_dict.update({xs[i]: x_gen[i] for i in range(args.nr_gpu)})
                new_x_gen_np = sess.run(new_x_gen, feed_dict=feed_dict)
                for i in range(args.nr_gpu):
                    x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    if init:
        feed_dict = {x_init: x}
        if gh_init is not None:
            pass #feed_dict.update({gh_init: x})
        if sh_init is not None:
            h = x.copy()
            h = uf.mask_inputs(h, train_mgen)
            feed_dict.update({sh_init: h})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if args.spatial_conditional:
            h = [x[i].copy() for i in range(args.nr_gpu)]
            for i in range(args.nr_gpu):
                h[i] = uf.mask_inputs(h[i], train_mgen)
            feed_dict.update({shs[i]: h[i] for i in range(args.nr_gpu)})
    return feed_dict

# def make_feed_dict(data, init=False):
#     if type(data) is tuple:
#         x,y = data
#     else:
#         x = data
#         y = None
#     x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
#     if init:
#         feed_dict = {x_init: x}
#         if h_init is not None:
#             feed_dict.update({h_init: x})
#         if y is not None:
#             feed_dict.update({y_init: y})
#     else:
#         x = np.split(x, args.nr_gpu)
#         feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
#         if y is not None:
#             y = np.split(y, args.nr_gpu)
#             feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
#
#     if args.spatial_conditional:
#         if not init:
#             feed_dict.update({hs[i]: x[i] for i in range(args.nr_gpu)})
#     return feed_dict

# //////////// perform training //////////////
# for k in range(k)
# _ = next(test_data)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
test_bpd = []
lr = args.learning_rate

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    # generate samples from the model
    sample_x = []
    for i in range(args.num_samples):
        sample_x.append(sample_from_model(sess, data=next(test_data))) ##
    sample_x = np.concatenate(sample_x,axis=0)
    np.savez(os.path.join("plots",'%s_complete_%s.npz' % (args.data_set, exp_label)), sample_x)


    for i in range(sample_x.shape[0]):
        ms = test_mgen.gen(1)[0]
        contour = 1-uf.find_contour(ms)[:, :, None]
        contour[contour<1] = 0.0
        sample_x[i] *= contour


    img_tile = plotting.img_tile(sample_x[:100], aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
    plotting.plt.savefig(os.path.join("plots",'%s_complete_%s.png' % (args.data_set, exp_label)))
    plotting.plt.close('all')
    # np.savez(os.path.join("plots",'%s_complete_%s.npz' % (args.data_set, exp_label)), sample_x)
