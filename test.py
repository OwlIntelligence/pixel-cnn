import numpy as np
import os
import tensorflow as tf
import time
import data.celeba_data as celeba_data
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn
from utils import plotting
from pixel_cnn_pp.nn import adam_updates
import utils.mask as m
import vae_loading as v


test_data = celeba_data.DataLoader(v.FLAGS.data_dir, 'valid', v.FLAGS.batch_size*v.FLAGS.nr_gpu, shuffle=False, size=128)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    v.load_vae(sess, v.saver)

    test_mgen = m.CenterMaskGenerator(128, 128, .5)

    data = next(test_data)
    data = uf.mask_inputs(data, test_mgen)[:,:,:,:3]

    # img_tile = plotting.img_tile(data[:25], aspect_ratio=1.0, border_color=1.0, stretch=True)
    # img = plotting.plot_img(img_tile, title=v.FLAGS.data_set + ' samples')
    # plotting.plt.savefig(os.path.join("plots",'%s_vae_original.png' % (v.FLAGS.data_set)))


    feed_dict = v.make_feed_dict(data, test_mgen)
    sample_x = sess.run(v.x_hats, feed_dict=feed_dict)
    sample_x = np.concatenate(sample_x, axis=0)
    sample_mask = test_mgen.gen(1)[0]
    data = np.cast[np.float32](data/255.)
    sample_x = sample_x * (1-sample_mask) + data * sample_mask
    test_data.reset()

    img_tile = plotting.img_tile(sample_x[:25], aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title=v.FLAGS.data_set + ' samples')
    plotting.plt.savefig(os.path.join("plots",'%s_vae_complete.png' % (v.FLAGS.data_set)))
