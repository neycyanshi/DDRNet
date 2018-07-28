import tensorflow as tf
import tensorflow.contrib.slim as slim

import losses
import ops
from ops import resize_like, resize_conv, instance_norm
import resnet_model as res


def batch_norm_params(is_training):
    return {
        'decay': 0.9,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': None,
        'is_training': is_training
    }


def loss(depth_dn, depth_dt, depth_gt, config, normal_dn=None, normal_dt=None, normal_gt=None, color=None, mask=None, abd=None):
    '''
    :param depth_dn: NHWC denoised depth_dn.
    :param depth_gt: NHWC near ground truth depth.
    :param normal_dt: NCHW normal_dt computed from depth_dt after dt_net.
    :param normal_gt: NCHW normal_gt computed by depth_to_normal ops.
    :param color: NHWC color/gray image.
    :param mask: NHW1 mask from gt or RGB segmentation.
    :return: aggregated loss.
    '''

    if config.dtnet == "None": assert (depth_dt is None) and (normal_dt is None)

    batch_size = config.batch_size
    range_scale = (config.up_thres - config.low_thres) / (1.0 + 1.0)
    mask = ops.convertNHWC2NCHW(mask, name='mask_NCHW')
    depth_dn = ops.convertNHWC2NCHW(depth_dn, name='depth_NCHW')
    depth_gt = ops.convertNHWC2NCHW(depth_gt, name='depth_gt_NCHW')
    if config.has_abd:
        assert abd is not None
        abd = ops.convertNHWC2NCHW(abd, name='abd_NCHW')

    # dn_net Loss
    if config.dnnet != "None":
        # depthGradLoss = losses.tv_loss(depth_dn, mask, loss_weight=0.1, scope='dn_depthGradLoss')
        # normalGradLoss = losses.tv_loss(normal_dn, mask, loss_weight=1e-3, scope='dn_normalGradLoss')  # rm high freq in normal_dn
        # normaldotLoss = losses.normaldot_loss_tf(depth_dn, normal_gt, config, weight=1e-6, scope='dn_normaldottfLoss')
        # L2Loss = losses.L2loss(depth_dn, depth_gt, weight=1e2, scope='dn_L2Loss')  # deprecated. ref has mask now.
        maskedLoss = losses.masked_loss(depth_dn, depth_gt, batch_size, mask,
                                        huber=0.0, L1=1.0, L2=0.0, weight=1.0, rng_scale=range_scale, scope='dn_maskedLoss')

    # dt_net Loss
    if config.dtnet != "None":
        depth_dt = ops.convertNHWC2NCHW(depth_dt, name='depth_dt')
        fidelityLoss = losses.masked_loss(depth_dt, depth_gt, batch_size, mask,
                                          huber=0.0, L1=1.0, L2=0.0, weight=1.0, rng_scale=range_scale,
                                          scope='dt_fidelityLoss')
        shadingLoss, irrad, albedo = losses.shading_loss(normal_dt, normal_gt,
                                                         ops.convertNHWC2NCHW(color, name='color_NCHW'), mask, abd=abd, weight=100,
                                                         scope='dt_shadingLoss')
        # smoothLoss = losses.tv_loss(depth_dt, mask, loss_weight=0.1, scope='dt_smoothLoss')

    print('Logging loss value: "loss1" is {}; "loss2" is {}'.format('maskedLoss', 'fidelityLoss'))
    return tf.add_n(tf.get_collection(losses.LOSSES_COLLECTION), name='total_loss'), maskedLoss, fidelityLoss, (irrad, albedo)


def base(x, is_training, aux=None, reuse=None, scope=None, ngf=16):
    with tf.variable_scope(scope or 'model', reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        weight_collection = scp.name + 'weight'

        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=None,
                            variables_collections=[weight_collection],
                            outputs_collections=end_pts_collection):
            x = slim.conv2d(x, ngf, [3, 3], scope='preconv')

            for i in range(2):
                x = slim.conv2d(x, ngf, [3, 3], scope='block{}/cnv1'.format(i + 1))
                x = slim.conv2d(x, ngf, [3, 3], scope='block{}/cnv2'.format(i + 1))

            x = slim.conv2d(x, 1, [3, 3], activation_fn=tf.nn.tanh, scope='cnv_final')

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            weight_vars = tf.get_collection(weight_collection)
            return x, end_pts, weight_vars


def convResnet(x, is_training, aux=None, reuse=None, scope='dn_net', ngf=32, n_blocks=2, n_down=2, learn_residual=True):
    with tf.variable_scope(scope, reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        weight_collection = scp.name + 'weight'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            variables_collections=[weight_collection],
                            outputs_collections=end_pts_collection):
            scale_skips = []
            cat_axis = 3  # NHWC
            if aux is None:
                cnv = slim.conv2d(x, ngf, [3, 3], scope='cnv0')
            else:
                cnv0_d = slim.conv2d(x, ngf // 2, [3, 3], stride=2, scope='cnv0_d')
                cnv0_c = slim.conv2d(resize_like(aux, cnv0_d), ngf // 2, [1, 1], stride=1, scope='cnv0_c')
                cnv = tf.concat([cnv1_d, cnv0_c], axis=3)
            scale_skips.append(cnv)

            mult = 1
            for i in range(n_down):
                mult *= 2
                cnv = slim.conv2d(cnv, ngf * mult, [3, 3], stride=2, scope='cnv_down{}'.format(i))
                scale_skips.append(cnv)
            for i in range(n_blocks):
                cnv = ops.residual_block(cnv, ngf * mult, norm_fn=None, scope='res{}'.format(i))
            for i in range(n_down):
                mult /= 2
                cnv = tf.concat([scale_skips[-1], cnv], cat_axis, name='cat{}'.format(i + 1))
                scale_skips.pop()
                cnv = slim.conv2d_transpose(cnv, ngf * mult, [3, 3], stride=2, scope='cnv_up{}'.format(i))

            cnv = tf.concat([scale_skips[0], cnv], cat_axis, name='cat_final')
            assert len(scale_skips) == 1
            del scale_skips
            cnv = slim.conv2d(cnv, 1, [3, 3], activation_fn=tf.nn.tanh, normalizer_fn=None, scope='cnv_final')

            if learn_residual:
                out = x + cnv
                out = tf.clip_by_value(out, -1, 1)
            else:
                out = cnv  # if cnv_final act is tanh
                # out = tf.clip_by_value(cnv, -1, 1)  # if cnv_final act is None

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            weight_vars = tf.get_collection(weight_collection)
            return out, end_pts, weight_vars


def UResnet(x, is_training, aux=None, reuse=None, scope='dn_net', ngf=32, n_blocks=1, n_down=2,
                learn_residual=False, data_format="channels_first"):
    with tf.variable_scope(scope, reuse=reuse) as scp:
        if data_format == "channels_first":
            x = tf.transpose(x, [0, 3, 1, 2])
        cat_axis = 1

        scale_skips = []
        cnv = res.conv2d_fixed_padding(x, ngf, 7, strides=1, data_format=data_format, layer_name='cnv_first')
        scale_skips.append(cnv)

        mult = 1
        for i in range(n_down):
            mult *= 2
            cnv = res.block_layer(cnv, ngf * mult, block_fn=res.building_block, blocks=1, strides=2,
                                  is_training=is_training,
                                  name='down{}'.format(i + 1), data_format=data_format)
            scale_skips.append(cnv)
        for i in range(n_blocks):
            cnv = res.block_layer(cnv, ngf * mult, block_fn=res.building_block, blocks=1, strides=1,
                                  is_training=is_training,
                                  name='mid{}'.format(i + 1), data_format=data_format)
        for i in range(n_down):
            mult /= 2
            cnv = tf.concat([scale_skips[-1], cnv], cat_axis, name='cat{}'.format(i + 1))
            scale_skips.pop()
            cnv = res.block_layer(cnv, ngf * mult, block_fn=res.building_block, blocks=1, strides=0.5,
                                  is_training=is_training, name='up{}'.format(i + 1), data_format=data_format)

        cnv = tf.concat([scale_skips[0], cnv], cat_axis, name='cat_final')
        assert len(scale_skips) == 1
        del scale_skips
        cnv = tf.layers.conv2d(cnv, 1, 1, activation=tf.tanh, name='cnv_final', use_bias=False, data_format=data_format)

        if learn_residual:
            out = x + cnv
            out = tf.clip_by_value(out, -1, 1)
        else:
            out = cnv

        if data_format == "channels_first":
            out = tf.transpose(out, [0, 2, 3, 1])

        return out, None, None


def unet(x, is_training, aux=None, reuse=None, scope='dn_net', ngf=32):
    H, W = x.shape.as_list()[1:3]
    with tf.variable_scope(scope, reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        weight_collection = scp.name + 'weight'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            variables_collections=[weight_collection],
                            outputs_collections=end_pts_collection):
            if aux is None:
                cnv1 = slim.conv2d(x, ngf, [7, 7], stride=2, scope='cnv1')
            else:
                cnv1_d = slim.conv2d(x, ngf // 2, [7, 7], stride=2, normalizer_fn=slim.batch_norm,
                                     normalizer_params=batch_norm_params(is_training), scope='cnv1_d')
                cnv1_c = slim.conv2d(resize_like(aux, cnv1_d), ngf // 2, [1, 1], stride=1, normalizer_fn=slim.batch_norm,
                                     normalizer_params=batch_norm_params(is_training), scope='cnv1_c')
                cnv1 = tf.concat([cnv1_d, cnv1_c], axis=3)
            cnv2 = slim.conv2d(cnv1, ngf * 2, [5, 5], stride=2, scope='cnv2')
            cnv3 = slim.conv2d(cnv2, ngf * 4, [3, 3], stride=2, scope='cnv3')
            cnv4 = slim.conv2d(cnv3, ngf * 8, [3, 3], stride=2, scope='cnv4')
            cnv5 = slim.conv2d(cnv4, ngf * 8, [3, 3], stride=2, scope='cnv5')
            cnv6 = slim.conv2d(cnv5, ngf * 8, [3, 3], stride=2, scope='cnv6')
            cnv7 = slim.conv2d(cnv6, ngf * 8, [3, 3], stride=2, scope='cnv7')  # /128

            upcnv7 = slim.conv2d_transpose(cnv7, ngf * 8, [3, 3], stride=2, scope='upcnv7')  # /64
            upcnv7 = resize_like(upcnv7, cnv6)
            i7_in = tf.concat([upcnv7, cnv6], axis=3)
            icnv7 = slim.conv2d(i7_in, ngf * 8, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, ngf * 8, [3, 3], stride=2, scope='upcnv6')  # /32
            upcnv6 = resize_like(upcnv6, cnv5)
            i6_in = tf.concat([upcnv6, cnv5], axis=3)
            icnv6 = slim.conv2d(i6_in, ngf * 8, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, ngf * 8, [3, 3], stride=2, scope='upcnv5')  # /16
            upcnv5 = resize_like(upcnv5, cnv4)
            i5_in = tf.concat([upcnv5, cnv4], axis=3)
            icnv5 = slim.conv2d(i5_in, ngf * 8, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, ngf * 4, [3, 3], stride=2, scope='upcnv4')  # /8
            i4_in = tf.concat([upcnv4, cnv3], axis=3)
            icnv4 = slim.conv2d(i4_in, ngf * 4, [3, 3], stride=1, scope='icnv4')
            out4 = slim.conv2d(icnv4, 1, [3, 3], stride=1, activation_fn=tf.tanh, normalizer_fn=None, scope='out4')
            out4_up = tf.image.resize_bilinear(out4, [int(H / 4), int(W / 4)])

            upcnv3 = slim.conv2d_transpose(icnv4, ngf * 2, [3, 3], stride=2, scope='upcnv3')  # /4
            i3_in = tf.concat([upcnv3, cnv2, out4_up], axis=3)
            icnv3 = slim.conv2d(i3_in, ngf * 2, [3, 3], stride=1, scope='icnv3')
            out3 = slim.conv2d(icnv3, 1, [3, 3], stride=1, activation_fn=tf.tanh, normalizer_fn=None, scope='out3')
            out3_up = tf.image.resize_bilinear(out3, [int(H / 2), int(W / 2)])

            upcnv2 = slim.conv2d_transpose(icnv3, ngf, [3, 3], stride=2, scope='upcnv2')  # /2
            i2_in = tf.concat([upcnv2, cnv1, out3_up], axis=3)
            icnv2 = slim.conv2d(i2_in, ngf, [3, 3], stride=1, scope='icnv2')
            out2 = slim.conv2d(icnv2, 1, [3, 3], stride=1, activation_fn=tf.tanh, normalizer_fn=None, scope='out2')
            # out2_up = tf.image.resize_bilinear(out2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, ngf // 2, [3, 3], stride=2, scope='upcnv1')
            i1_in = tf.concat([upcnv1], axis=3)  # [upcnv1, out2_up]
            icnv1 = slim.conv2d(i1_in, ngf // 2, [3, 3], stride=1, scope='icnv1')
            out1 = slim.conv2d(icnv1, 1, [3, 3], stride=1, activation_fn=tf.tanh, normalizer_fn=None, scope='out1')

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            weight_vars = tf.get_collection(weight_collection)
            return out1, end_pts, weight_vars


def hypercolumn(depth, color, reuse=None, scope='dt_net', ngf=16):
    """
    :param depth: NHW1 
    :param color: NHWC
    :return: D_dt NHW1 
    """
    with tf.variable_scope(scope, reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        weight_collection = scp.name + 'weight'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            variables_collections=[weight_collection],
                            outputs_collections=end_pts_collection):
            cnv1_d = slim.conv2d(depth, ngf / 2, [3, 3], scope='cnv1_d')
            cnv1_c = slim.conv2d(color, ngf / 2, [3, 3], scope='cnv1_c')
            cnv1 = tf.concat([cnv1_d, cnv1_c], axis=3)
            cnv2 = slim.conv2d(cnv1, ngf, [3,3], scope='cnv2')
            hyper1 = tf.concat([cnv1, cnv2], axis=3)  # ngf*2

            pool1 = slim.max_pool2d(hyper1, [2,2], stride=2, scope='pool1')
            cnv3 = slim.conv2d(pool1, ngf*2, [3,3], scope='cnv3')
            cnv4 = slim.conv2d(cnv3, ngf*2, [3,3], scope='cnv4')
            up1 = resize_like(cnv4, cnv2, method='BI')
            # up1 = slim.conv2d_transpose(cnv4, ngf*2, [3,3], stride=2, scope='up1')
            hyper2 = tf.concat([cnv3, cnv4], axis=3)  # ngf*4

            pool2 = slim.max_pool2d(hyper2, [2,2], stride=2, scope='pool2')
            cnv5 = slim.conv2d(pool2, ngf * 4, [3, 3], scope='cnv5')
            cnv6 = slim.conv2d(cnv5, ngf * 4, [3, 3], scope='cnv6')
            cnv7 = slim.conv2d(cnv6, ngf * 4, [3, 3], scope='cnv7')
            up2 = resize_like(cnv7, cnv2, method='BI')
            # up2 = slim.conv2d_transpose(cnv7, ngf*4, [3,3], stride=2, scope='up2')
            cat = tf.concat([cnv2, up1, up2], axis=3)  # ngf*7

            # cnv8 = slim.conv2d(hyper1, ngf, [1,1], scope='cnv8')
            cnv8 = slim.conv2d(cat, ngf, [1,1], scope='cnv8')
            cnv9 = slim.conv2d(cnv8, ngf//4, [1,1], scope='cnv9')
            out = slim.conv2d(cnv9, 1, [1,1], activation_fn=tf.tanh, scope='out')
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            weight_vars = tf.get_collection(weight_collection)
            return out, end_pts, weight_vars
