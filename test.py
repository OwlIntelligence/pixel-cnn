import vae_loading as v



test_data = celeba_data.DataLoader(FLAGS.data_dir, 'valid', FLAGS.batch_size*FLAGS.nr_gpu, shuffle=False, size=128)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    v.load_vae(v.saver)

    test_mgen = m.CenterMaskGenerator(128, 128, 0.5)

    data = next(test_data)
    feed_dict = v.make_feed_dict(data, test_mgen)
    sample_x = sess.run(v.x_hats, feed_dict=feed_dict)
    sample_x = np.concatenate(sample_x, axis=0)
    test_data.reset()

    img_tile = plotting.img_tile(sample_x[:25], aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title=v.FLAGS.data_set + ' samples')
    plotting.plt.savefig(os.path.join("plots",'%s_vae_complete.png' % (v.FLAGS.data_set)))
