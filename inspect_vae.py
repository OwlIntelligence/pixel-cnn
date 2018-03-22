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

import vae_loading as vl
test_data = celeba_data.DataLoader(vl.FLAGS.data_dir, 'test', vl.FLAGS.batch_size*vl.FLAGS.nr_gpu, shuffle=False, size=128)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    vl.load_vae(sess, vl.saver)

    d = next(test_data)

    feed_dict = vl.make_feed_dict(d)
    ret = sess.run(vl.locs+vl.log_vars, feed_dict=feed_dict)
    locs, log_vars = ret[:len(ret)//2], ret[len(ret)//2:]
    print(locs[0])
    print(log_vars[0])
