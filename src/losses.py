from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ops
from lighting import get_lighting, get_H

LOSSES_COLLECTION = 'losses'


def print_args(f):
    def fn(*args, **kw):
        ret = f(*args, **kw)
        kwargs = kw if kw else f.__defaults__
        print('{}: {}'.format(f.__name__, kwargs))
        return ret
    return fn


@print_args
def L2loss(output, gt, weight=1e2, scope=None):
    with tf.name_scope(scope, 'L2loss', [output, gt]):
        L2loss = weight * tf.reduce_mean(tf.squared_difference(gt, output))  #tf.losses.mean_squared_error(gt, output)
        tf.add_to_collection(LOSSES_COLLECTION, L2loss)
        return L2loss


@print_args
def masked_loss(output, gt, batch_size, mask, huber=0.005, L1=1.0, L2=0.0, weight=1.0, rng_scale=750, scope=None):
    """    
    :param output: NCHW predicted depth
    :param gt: NCHW
    :param batch_size: 
    :param mask: NCHW tensor of type bool. relavant region in gt are True, depth < low_thre are masked out.
    :param huber: `float`, the point where the huber loss function changes from a quadratic to linear.
    :param L1: weight of L1_lossz
    :param L2: weight of L2_loss
    :param rng_scale: scale factor from [-1, 1] to original depth value. (up_thres - low_thres) / (1.0 + 1.0)
    :param scope: 
    :return: 
    """
    with tf.name_scope(scope, 'masked_loss', [output, gt]):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
        else:
            low_thres = -0.99
            mask = tf.greater(gt, low_thres, name='gt_mask')
            mask = tf.cast(mask, tf.float32)
        mask = weight * rng_scale * mask
        # tf.stop_gradient(mask, 'mask_stop')
        masked_loss = 0

        if huber > 0:  # 0.005*rng_scale=3.75mm
            weights = mask / huber
            masked_loss += tf.losses.huber_loss(gt, output, weights=weights, delta=huber, loss_collection=LOSSES_COLLECTION)
        if L1 > 0:
            weights = L1 * mask
            masked_loss += tf.losses.absolute_difference(gt, output, weights=weights, loss_collection=LOSSES_COLLECTION)
        if L2 > 0:
            weights = L2 * mask
            masked_loss += tf.losses.mean_squared_error(gt, output, weights=weights, loss_collection=LOSSES_COLLECTION)
        return masked_loss


@print_args
def normaldot_loss_tf(output, normal, config, weight=1e-6, scope=None):
    with tf.name_scope(scope, 'normaldot_loss_tf', [output, normal]):
        # convert normalized depth to real depth
        output = ops.unnormalize(output, low_thres=config.low_thres, up_thres=config.up_thres)
        fx, fy, cx, cy = [536.628, 536.606, 310.591, 234.759]

        y_size, x_size = output.shape.as_list()[-2:]
        output = tf.reshape(output, [-1, y_size, x_size])
        B = config.batch_size #output.shape[0]

        eps = 10.0
        valid_mask = tf.greater(output, config.low_thres+eps, name='valid_mask')  # dbg

        X, Y = tf.meshgrid(tf.range(x_size), tf.range(y_size))
        X = tf.cast(tf.tile(tf.expand_dims(X, axis=0), [B, 1, 1]), tf.float32)  # (B,H,W)
        Y = tf.cast(tf.tile(tf.expand_dims(Y, axis=0), [B, 1, 1]), tf.float32)
        DX = tf.multiply(output, X)
        DY = tf.multiply(output, Y)
        nx = tf.reshape(normal[:, 0, :, :], [-1, y_size, x_size])
        ny = tf.reshape(normal[:, 1, :, :], [-1, y_size, x_size])
        nz = tf.reshape(normal[:, 2, :, :], [-1, y_size, x_size])
        alpha_x = tf.div(nx[:, 1:-1, 1:-1], fx)
        alpha_y = tf.div(ny[:, 1:-1, 1:-1], fy)
        alpha_z = nz[:, 1:-1, 1:-1]
        # product without pad (N, H-2, W-2)
        product = \
            tf.square(tf.multiply(DX[:, :-2, 1:-1] - DX[:, 1:-1, 1:-1], alpha_x) + tf.multiply(
                DY[:, :-2, 1:-1] - DY[:, 1:-1, 1:-1], alpha_y) + tf.multiply(
                output[:, :-2, 1:-1] - output[:, 1:-1, 1:-1], alpha_z)) \
            + tf.square(tf.multiply(DX[:, 2:, 1:-1] - DX[:, 1:-1, 1:-1], alpha_x) + tf.multiply(
                DY[:, 2:, 1:-1] - DY[:, 1:-1, 1:-1], alpha_y) + tf.multiply(
                output[:, 2:, 1:-1] - output[:, 1:-1, 1:-1], alpha_z)) \
            + tf.square(tf.multiply(DX[:, 1:-1, :-2] - DX[:, 1:-1, 1:-1], alpha_x) + tf.multiply(
                DY[:, 1:-1, :-2] - DY[:, 1:-1, 1:-1], alpha_y) + tf.multiply(
                output[:, 1:-1, :-2] - output[:, 1:-1, 1:-1], alpha_z)) \
            + tf.square(tf.multiply(DX[:, 1:-1, 2:] - DX[:, 1:-1, 1:-1], alpha_x) + tf.multiply(
                DY[:, 1:-1, 2:] - DY[:, 1:-1, 1:-1], alpha_y) + tf.multiply(
                output[:, 1:-1, 2:] - output[:, 1:-1, 1:-1], alpha_z))

        paddings = [[0,0],[1,1],[1,1]]
        product = tf.pad(product, paddings)
        product = tf.where(valid_mask, product, tf.zeros_like(product))  # dbg
        normaldotLoss = weight * tf.reduce_sum(product) / (2 * config.batch_size)
        tf.add_to_collection(LOSSES_COLLECTION, normaldotLoss)
        return normaldotLoss


@print_args
def shading_loss(normal, normal_gt, gray, mask, abd=None, grad_metric=False, avg_pool=False, weight=1e4, scope=None):
    """
    :param normal: NCHW
    :param normal_gt: NCHW
    :param gray: NCHW
    :param abd: NCHW
    :param grad_metric: True if no uniform abd
    :param mask: NCHW
    """
    with tf.name_scope(scope, 'shading_loss', [normal_gt, gray, mask]):
        B, C, H, W = normal.shape.as_list()
        if avg_pool:
            gray = tf.nn.avg_pool(gray, [1,1,3,3], [1,1,1,1], 'SAME', data_format='NCHW', name='color_avg')
        gray = tf.div(tf.add(gray, 1.0), 2.0, name='gray_NCHW')  # convert to [0,1]

        # uniform albedo all value 1.0
        albedo = mask if not abd else None

        # # per-pixel albedo
        # albedo = gray / LH  # (B,1,H,W)
        # albedo_zero = tf.zeros_like(albedo)
        # albedo = tf.where(tf.cast(mask_n, tf.bool), albedo, albedo_zero)
        # albedo = tf.clip_by_value(albedo, 0.0, 2.0)  # non-zero albedo has mean 1.0, use 2.0 as up-threshold

        l_coefs, LH, _ = get_lighting(normal_gt, gray, None, mask)


        # render normal under lighting estimated by normal_gt and gray image.
        A = get_H(normal, mask)  # (B,9,H*W) ??mask_n
        irrad = albedo * tf.reshape(tf.matmul(tf.expand_dims(l_coefs, 1), A), [-1, 1, H, W])
        irrad = tf.clip_by_value(irrad, 0.0, 1.0)  # same range as gray
        if grad_metric:
            gray_dif1 = gray[:, :, 1:-1, 1:-1] - gray[:, :, :-2, 1:-1]
            gray_dif2 = gray[:, :, 1:-1, 1:-1] - gray[:, :, 1:-1, :-2]
            irrad_dif1 = irrad[:, :, 1:-1, 1:-1] - irrad[:, :, :-2, 1:-1]
            irrad_dif2 = irrad[:, :, 1:-1, 1:-1] - irrad[:, :, 1:-1, :-2]
            weights = weight * mask[:, :, 1:-1, 1:-1]
            shadingLoss = tf.losses.mean_squared_error(gray_dif1, irrad_dif1, weights=weights, loss_collection=LOSSES_COLLECTION) + \
                          tf.losses.mean_squared_error(gray_dif2, irrad_dif2, weights=weights, loss_collection=LOSSES_COLLECTION)
        else:
            weights = weight * mask
            shadingLoss = tf.losses.mean_squared_error(gray, irrad, weights=weights, loss_collection=LOSSES_COLLECTION)
        return shadingLoss, irrad, albedo


@print_args
def tv_loss(output, mask, weight=0.1, scope=None):
    scope = 'tv_loss' if scope is None else scope
    print('{} weight {}'.format(scope, weight))
    with tf.name_scope(scope, 'tv_loss', [output, mask]):
        tvLoss = weight * tf.reduce_mean(tf.image.total_variation(tf.multiply(output, mask)))
        tf.add_to_collection(LOSSES_COLLECTION, tvLoss)
        return tvLoss
