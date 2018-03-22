import os

import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf
import data.celeba_data as celeba_data
from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting
import utils.mask as um
import utils.mfunc as uf
import utils.grid as grid

import vae64_loading as vl
test_data = celeba_data.DataLoader(vl.FLAGS.data_dir, 'test', vl.FLAGS.batch_size*vl.FLAGS.nr_gpu, shuffle=False, size=64)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    vl.load_vae(sess, vl.saver)

    d = next(test_data)
    d = next(test_data)

    feed_dict = vl.make_feed_dict(d)
    ret = sess.run(vl.locs+vl.log_vars, feed_dict=feed_dict)
    locs, log_vars = np.concatenate(ret[:len(ret)//2], axis=0), np.concatenate(ret[len(ret)//2:], axis=0)
    scale = np.sqrt(np.exp(log_vars))

    img_id = 2

    locs = np.array([locs[img_id] for i in range(32*9)])
    for i in range(32):
        s = scale[img_id][i]
        for k in range(9):
            locs[i*9+k][i] = (k-3)

    feed_dict = vl.make_feed_dict_z(locs)
    ret = sess.run(vl.x_hats, feed_dict=feed_dict)
    sample_x = np.concatenate(ret, axis=0)

    sample_x = np.rint(sample_x * 255.)

    print(sample_x.shape)

    from PIL import Image
    img = Image.fromarray(uf.tile_images(sample_x.astype(np.uint8), size=(32, 9)), 'RGB')
    img.save(os.path.join("plots", '%s_vae64_%s_%d.png' % (vl.FLAGS.data_set, "test", img_id)))
