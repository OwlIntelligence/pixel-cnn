"""
The core Pixel-CNN model
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn

def model_spec(x, gh=None, sh=None, ch=None, zh=None, indices=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu', energy_distance=False, global_conditional=False, spatial_conditional=False):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.conv2d_1x1, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' + resnet_nonlinearity + ' is not supported')

        # if spatial_conditional:
        #     if type(sh)==list:
        #         sh, sh_2, sh_4 = sh
        #     else:
        #         sh = nn.latent_deconv_net(sh, scale_factor=1)
        #         with arg_scope([nn.conv2d], nonlinearity=resnet_nonlinearity):
        #             sh = nn.conv2d(sh, 2*nr_filters, filter_size=[3,3], stride=[1,1], pad='VALID')
        #             sh = nn.conv2d(sh, 2*nr_filters, filter_size=[3,3], stride=[1,1], pad='VALID')
        #
        #             sh_2 = nn.conv2d(sh, nn.int_shape(sh)[-1], filter_size=[3,3], stride=[2,2], pad='SAME')
        #             sh_4 = nn.conv2d(sh_2, nn.int_shape(sh)[-1], filter_size=[3,3], stride=[2,2], pad='SAME')
        # else:
        #     sh_2, sh_4 = None, None

        if spatial_conditional:
            with arg_scope([nn.conv2d], nonlinearity=resnet_nonlinearity):
                #sh = nn.conv2d(sh, 2*nr_filters, filter_size=[3,3], stride=[1,1], pad='SAME')
                #sh = nn.conv2d(sh, 2*nr_filters, filter_size=[3,3], stride=[1,1], pad='SAME')
                if zh is not None:
                    zh = nn.deconv_net(zh)
                    # zh = tf.stack([tf.slice(zh[k], begin=(indices[k][0], indices[k][1], 0), size=(32,32,64)) for k in range(16)])
                    zh = tf.slice(zh, begin=(0, indices[0][0], indices[0][1], 0), size=(4,32,32,64))
                    sh = tf.concat([zh, sh], axis=-1)
                sh_2 = nn.conv2d(sh, nn.int_shape(sh)[-1], filter_size=[3,3], stride=[2,2], pad='SAME')
                sh_4 = nn.conv2d(sh_2, nn.int_shape(sh)[-1], filter_size=[3,3], stride=[2,2], pad='SAME')
            if ch is not None:
                ch_1, ch_2, ch_4 = ch
                sh = tf.concat([sh, ch_1], axis=-1)
                sh_2 = tf.concat([sh_2, ch_2], axis=-1)
                sh_4 = tf.concat([sh_4, ch_4], axis=-1)

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, gh=gh, sh=sh):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on
            u_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
            for rep in range(nr_resnet):
                u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
            ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))
            for rep in range(nr_resnet):
                u_list.append(nn.gated_resnet(u_list[-1], sh=sh_2, conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], sh=sh_2, conv=nn.down_right_shifted_conv2d))

            u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
            ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))
            for rep in range(nr_resnet):
                u_list.append(nn.gated_resnet(u_list[-1], sh=sh_4, conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], sh=sh_4, conv=nn.down_right_shifted_conv2d))

            # remember nodes
            for t in u_list+ul_list:
                tf.add_to_collection('checkpoints', t)

            # /////// down pass ////////
            u = u_list.pop()
            ul = ul_list.pop()
            for rep in range(nr_resnet):
                u = nn.gated_resnet(u, u_list.pop(), sh=sh_4, conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), sh=sh_4, conv=nn.down_right_shifted_conv2d)
                tf.add_to_collection('checkpoints', u)
                tf.add_to_collection('checkpoints', ul)

            u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
            ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])
            for rep in range(nr_resnet+1):
                u = nn.gated_resnet(u, u_list.pop(), sh=sh_2, conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), sh=sh_2, conv=nn.down_right_shifted_conv2d)
                tf.add_to_collection('checkpoints', u)
                tf.add_to_collection('checkpoints', ul)

            u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
            ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])
            for rep in range(nr_resnet+1):
                u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
                tf.add_to_collection('checkpoints', u)
                tf.add_to_collection('checkpoints', ul)

            if energy_distance:
                f = nn.nin(tf.nn.elu(ul), 64)

                # generate 10 samples
                fs = []
                for rep in range(10):
                    fs.append(f)
                f = tf.concat(fs, 0)
                fs = nn.int_shape(f)
                f += nn.nin(tf.random_uniform(shape=fs[:-1] + [4], minval=-1., maxval=1.), 64)
                f = nn.nin(nn.concat_elu(f), 64)
                x_sample = tf.tanh(nn.nin(nn.concat_elu(f), 3, init_scale=0.1))

                x_sample = tf.split(x_sample, 10, 0)

                assert len(u_list) == 0
                assert len(ul_list) == 0

                return x_sample

            else:
                x_out = nn.nin(tf.nn.elu(ul),10*nr_logistic_mix)

                assert len(u_list) == 0
                assert len(ul_list) == 0

                return x_out
