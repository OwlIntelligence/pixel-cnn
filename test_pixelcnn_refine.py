"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr_gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os

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
import utils.grid as grid



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
# new features
parser.add_argument('-sc', '--spatial_conditional', dest='spatial_conditional', action='store_true', help='Condition on spatial latent codes?')
parser.add_argument('-gc', '--global_conditional', dest='global_conditional', action='store_true', help='Condition on global latent codes?')
parser.add_argument('-ms', '--map_sampling', dest='map_sampling', action='store_true', help='use MAP sampling?')
parser.add_argument('-cn', '--config_name', type=str, default='None', help='what is the config name?')
parser.add_argument('-gd', '--global_latent_dim', type=int, default=10, help='dimension for the global latent variables')
parser.add_argument('-sn', '--spatial_latent_num_channel', type=int, default=4, help='number of channels for spatial latent variables')
parser.add_argument('-is', '--input_size', type=int, default=-1, help='input size')
parser.add_argument('-cc', '--context_conditioning', dest='context_conditioning', action='store_true', help='Condition on context (masked inputs)?')
parser.add_argument('-dg', '--debug', dest='debug', action='store_true', help='Under debug mode?')
#
args = parser.parse_args()

config_args(args, configs[args.config_name])
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

if args.nr_gpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif args.nr_gpu == 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
elif args.nr_gpu == 3:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
elif args.nr_gpu == 4:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

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
elif 'celeba' in args.data_set:
    import data.celeba_data as celeba_data
    DataLoader = celeba_data.DataLoader
else:
    raise("unsupported dataset")
if args.data_set=='celeba128':
    if args.debug:
        train_data = DataLoader(args.data_dir, 'valid', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional, size=128)
    else:
        train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional, size=128)
    test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional, size=128)
else:
    if args.debug:
        train_data = DataLoader(args.data_dir, 'valid', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
    else:
        train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
    test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
img_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
if args.input_size < 0:
    obs_shape = img_shape
else:
    obs_shape = args.input_size, args.input_size, img_shape[-1]

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]
gh_init, sh_init = None, None
ghs, shs, gh_sample, sh_sample = [[None] * args.nr_gpu for i in range(4)]

if args.global_conditional:
    gh_init = tf.placeholder(tf.float32, shape=(args.init_batch_size, args.global_latent_dim))
    ghs = [tf.placeholder(tf.float32, shape=(args.batch_size, args.global_latent_dim)) for i in range(args.nr_gpu)]
    gh_sample = ghs
if args.spatial_conditional:
    # spatial_latent_shape = obs_shape[0], obs_shape[1], args.spatial_latent_num_channel ##
    spatial_latent_shape = obs_shape[0]+4, obs_shape[1]+4, args.spatial_latent_num_channel
    sh_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + spatial_latent_shape)
    shs = [tf.placeholder(tf.float32, shape=(args.batch_size,) + spatial_latent_shape ) for i in range(args.nr_gpu)]
    sh_sample = shs



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
        mask_tf = None
        if args.context_conditioning:
            mask_tf = shs[i][:, :, :, -1]
        loss_gen.append(loss_fun(tf.stop_gradient(xs[i]), out, masks=mask_tf))

        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params, colocate_gradients_with_ops=True))

        # test
        out = model(xs[i], ghs[i], shs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(loss_fun(xs[i], out, masks=mask_tf))

        # sample
        out = model(xs[i], gh_sample[i], sh_sample[i], ema=ema, dropout_p=0, **model_opt)
        if args.energy_distance:
            new_x_gen.append(out[0])
        else:
            if args.map_sampling:
                epsilon = 0.5 - 1e-5
            else:
                epsilon = 1e-5
            epsilon = 0.05
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
train_mgen = um.RandomRectangleMaskGenerator(obs_shape[0], obs_shape[1], max_ratio=1.0)
#train_mgen = um.CenterMaskGenerator(obs_shape[0], obs_shape[1])
test_mgen = um.RandomRectangleMaskGenerator(obs_shape[0], obs_shape[1], max_ratio=1.0)
sample_mgen = um.CenterMaskGenerator(obs_shape[0], obs_shape[1], 0.875)

def sample_from_model(sess, data=None, **params):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) ## preprocessing

    if 'use_coordinates' in params and params['use_coordinates']:
        g = grid.generate_grid((x.shape[1], x.shape[2]), batch_size=x.shape[0])
        if 'x_hats' in params:
            x_hats = params['x_hats']
            x_hats = (x_hats * 2.) - 1.
            xgx = np.concatenate([x, g, x_hats], axis=-1)
            #xgx, _ = uf.random_crop_images(xgx, output_size=(args.input_size, args.input_size))
            xgx, _ = uf.random_crop_images(xgx, output_size=(args.input_size+4, args.input_size+4))
            x, g, x_hats = xgx[:, :, :, :3], xgx[:, :, :, 3:5], xgx[:, :, :, 5:]
            x = x[:, 2:args.input_size+2, 2:args.input_size+2, :] ##
        else:
            xg = np.concatenate([x, g], axis=-1)
            xg, _ = uf.random_crop_images(xg, output_size=(args.input_size, args.input_size))
            x, g = xg[:, :, :, :3], xg[:, :, :, 3:]

    # global conditioning
    if args.global_conditional:
        global_lv = []
        if 'z' in params:
            global_lv.append(params['z'])
        global_lv = np.concatenate(global_lv, axis=-1)

    # spatial conditioning
    if args.spatial_conditional:
        spatial_lv = []
        if 'use_coordinates' in params and params['use_coordinates']:
            spatial_lv.append(g)
        if 'x_hats' in params:
            spatial_lv.append(x_hats)
        spatial_lv = np.concatenate(spatial_lv, axis=-1)

    if args.global_conditional:
        global_lv = np.split(global_lv, args.nr_gpu)
        feed_dict.update({ghs[i]: global_lv[i] for i in range(args.nr_gpu)})
    if args.spatial_conditional:
        spatial_lv = np.split(spatial_lv, args.nr_gpu)
        feed_dict.update({shs[i]: spatial_lv[i] for i in range(args.nr_gpu)})

    x = np.split(x, args.nr_gpu)
    x_gen = [np.zeros_like(x[0]) for i in range(args.nr_gpu)]

    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            feed_dict.update({xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            new_x_gen_np = sess.run(new_x_gen, feed_dict=feed_dict)
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)



# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()


def make_feed_dict(data, init=False, **params):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) ## preprocessing

    if 'use_coordinates' in params and params['use_coordinates']:
        g = grid.generate_grid((x.shape[1], x.shape[2]), batch_size=x.shape[0])
        if 'x_hats' in params:
            x_hats = params['x_hats']
            x_hats = (x_hats * 2.) - 1.
            xgx = np.concatenate([x, g, x_hats], axis=-1)
            #xgx, _ = uf.random_crop_images(xgx, output_size=(args.input_size, args.input_size))
            xgx, _ = uf.random_crop_images(xgx, output_size=(args.input_size+4, args.input_size+4))
            x, g, x_hats = xgx[:, :, :, :3], xgx[:, :, :, 3:5], xgx[:, :, :, 5:]
            x = x[:, 2:args.input_size+2, 2:args.input_size+2, :] ##
        else:
            xg = np.concatenate([x, g], axis=-1)
            xg, _ = uf.random_crop_images(xg, output_size=(args.input_size, args.input_size))
            x, g = xg[:, :, :, :3], xg[:, :, :, 3:]

    # global conditioning
    if args.global_conditional:
        global_lv = []
        if 'z' in params:
            global_lv.append(params['z'])
        global_lv = np.concatenate(global_lv, axis=-1)

    # spatial conditioning
    if args.spatial_conditional:
        spatial_lv = []
        if 'use_coordinates' in params and params['use_coordinates']:
            spatial_lv.append(g)
        if 'x_hats' in params:
            spatial_lv.append(x_hats)
        spatial_lv = np.concatenate(spatial_lv, axis=-1)

    if init:
        feed_dict = {x_init: x}
        if args.global_conditional:
            feed_dict.update({gh_init: global_lv})
        if args.spatial_conditional:
            feed_dict.update({sh_init: spatial_lv})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if args.global_conditional:
            global_lv = np.split(global_lv, args.nr_gpu)
            feed_dict.update({ghs[i]: global_lv[i] for i in range(args.nr_gpu)})
        if args.spatial_conditional:
            spatial_lv = np.split(spatial_lv, args.nr_gpu)
            feed_dict.update({shs[i]: spatial_lv[i] for i in range(args.nr_gpu)})
    return feed_dict


def complete(sess, data, mask, **params):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) ## preprocessing
    # mask images
    masks = uf.broadcast_mask(mask, 3, x.shape[0])
    x *= masks

    if 'x_hats' in params:
        x_hats = params['x_hats']
        x_hats = (x_hats * 2.) - 1.

    x_ret = np.split(x, args.nr_gpu)

    # global conditioning
    if args.global_conditional:
        global_lv = []
        if 'z' in params:
            global_lv.append(params['z'])
        global_lv = np.concatenate(global_lv, axis=-1)

    global_g = grid.generate_grid((x.shape[1], x.shape[2]), batch_size=x.shape[0])

    if args.global_conditional:
        global_lv = np.split(global_lv, args.nr_gpu)
        feed_dict.update({ghs[i]: global_lv[i] for i in range(args.nr_gpu)})

    while True:
        # find the next pixel and the corresonding window
        p = uf.find_next_missing_pixel(mask)
        if p is None:
            break
        window = uf.find_maximally_conditioned_window(mask, 32, p)
        print(p, window)
        [[h0, h1], [w0, w1]] = window
        g = global_g[:, h0-2:h1+2, w0-2:w1+2, :]
        # mw = mask[h0:h1, w0:w1]
        # xw = x[:, h0:h1, w0:w1, :]
        x_hatsw = x_hats[:, h0-2:h1+2, w0-2:w1+2, :]
        x_hatsws = np.split(x_hatsw, args.nr_gpu)
        yi, xi = p[0]-h0, p[1]-w0

        # spatial conditioning
        if args.spatial_conditional:
            spatial_lv = []
            if 'use_coordinates' in params and params['use_coordinates']:
                spatial_lv.append(g)
            if 'x_hats' in params:
                spatial_lv.append(x_hatsw)
            spatial_lv = np.concatenate(spatial_lv, axis=-1)

        if args.spatial_conditional:
            spatial_lv = np.split(spatial_lv, args.nr_gpu)
            feed_dict.update({shs[i]: spatial_lv[i] for i in range(args.nr_gpu)})

        x_gen = [x_ret[i][:, h0:h1, w0:w1, :].copy() for i in range(args.nr_gpu)]    # np.split(xw, args.nr_gpu)

        feed_dict.update({xs[i]: x_gen[i] for i in range(args.nr_gpu)})
        new_x_gen_np = sess.run(new_x_gen, feed_dict=feed_dict)
        for i in range(args.nr_gpu):
            x_ret[i][:,p[0],p[1],:] = new_x_gen_np[i][:,yi,xi,:]

        mask[p[0], p[1]] = 1

    return np.concatenate(x_ret, axis=0)


# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()



# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
test_bpd = []
lr = args.learning_rate

import svae_loading as vl

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    vl.load_vae(sess, vl.saver)

    ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    d = next(test_data)
    sample_mgen = um.RectangleMaskGenerator(128, 128, (40, 128-20, 125, 20))
    #sample_mgen = um.CenterMaskGenerator(128, 128, 24./64)
    mask = sample_mgen.gen(1)[0]

    feed_dict = vl.make_feed_dict(d, sample_mgen)
    ret = sess.run(vl.zs+vl.x_hats, feed_dict=feed_dict)
    zs, x_hats = ret[:args.nr_gpu], ret[args.nr_gpu:]
    zs, x_hats = np.concatenate(zs, axis=0), np.concatenate(x_hats, axis=0)
    sample_x = []
    for i in range(args.num_samples):
        sample_x.append(complete(sess, data=d, mask=mask, use_coordinates=True, z=zs, x_hats=x_hats)) ##
    sample_x = np.concatenate(sample_x,axis=0)


    #
    # feed_dict = vl.make_feed_dict(d)
    # ret = sess.run(vl.zs+vl.x_hats, feed_dict=feed_dict)
    # zs, x_hats = ret[:args.nr_gpu], ret[args.nr_gpu:]
    # zs, x_hats = np.concatenate(zs, axis=0), np.concatenate(x_hats, axis=0)
    # sample_x = []
    # vae_x = []
    # for i in range(args.num_samples):
    #     x_vae, x_s = sample_from_model(sess, data=d, use_coordinates=True, z=zs, x_hats=x_hats)
    #     sample_x.append(x_s)
    #     vae_x.append(x_vae)
    # sample_x = np.concatenate(sample_x,axis=0)
    # vae_x = np.concatenate(vae_x,axis=0)
    #
    # vae_x = np.rint(vae_x * 127.5 + 127.5)
    # from PIL import Image
    # img = Image.fromarray(uf.tile_images(vae_x.astype(np.uint8), size=(8,8)), 'RGB')
    # img.save(os.path.join("plots", '%s_ori_%s.png' % (args.data_set, "test")))
    #


    sample_x = np.rint(sample_x * 127.5 + 127.5)

    from PIL import Image
    img = Image.fromarray(uf.tile_images(sample_x.astype(np.uint8), size=(8,8)), 'RGB')
    img.save(os.path.join("plots", '%s_complete_%s.png' % (args.data_set, "test")))
