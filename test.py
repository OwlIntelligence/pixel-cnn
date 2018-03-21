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
# import vae_loading as v
import svae64_loading as v
import utils.mfunc as uf


test_data = celeba_data.DataLoader(v.FLAGS.data_dir, 'valid', v.FLAGS.batch_size*v.FLAGS.nr_gpu, shuffle=False, size=64)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    v.load_vae(sess, v.saver)

    test_mgen = m.RandomRectangleMaskGenerator(64, 64)

    data = next(test_data)
    data = next(test_data)

    # img_tile = plotting.img_tile(data[:25], aspect_ratio=1.0, border_color=1.0, stretch=True)
    # img = plotting.plot_img(img_tile, title=v.FLAGS.data_set + ' samples')
    # plotting.plt.savefig(os.path.join("plots",'%s_vae_original.png' % (v.FLAGS.data_set)))


    feed_dict = v.make_feed_dict(data, test_mgen)
    ret = sess.run([v.mxs]+v.x_hats, feed_dict=feed_dict)
    mx , x_hat = ret[0], ret[1:]
    mx, x_hat = np.concatenate(mx, axis=0), np.concatenate(x_hat, axis=0)
    mx = np.rint(mx*255.)
    x_hat = np.rint(x_hat*255.)

    from PIL import Image
    img = Image.fromarray(uf.tile_images(mx.astype(np.uint8), size=(4,4)), 'RGB')
    img.save(os.path.join("plots", '%s_vae_ori_%s.png' % ('celeba64', "test")))

    img = Image.fromarray(uf.tile_images(x_hat.astype(np.uint8), size=(4,4)), 'RGB')
    img.save(os.path.join("plots", '%s_vae_recon_%s.png' % ('celeba64', "test")))
