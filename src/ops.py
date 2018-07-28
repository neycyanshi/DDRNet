import numpy as np
import sys
import tensorflow as tf
slim = tf.contrib.slim


def convertNHWC2NCHW(data, name):
    out = tf.transpose(data, [0, 3, 1, 2], name=name)
    return out


def convertNCHW2NHWC(data, name):
    out = tf.transpose(data, [0, 2, 3, 1], name=name)
    return out


def denormalize(batch_input, low_thres, up_thres, zero2one=False, rm_zeros=False, eps=10.0):
    # denormalize depth from [-1, 1] to real depth.
    if not zero2one:  # [-1, 1]
        rel_input = (batch_input + 1.0) / 2.0
    else:  # [0, 1]
        rel_input = batch_input

    denormalized = rel_input * (up_thres - low_thres) + low_thres

    if rm_zeros:
        low_mask = tf.less(denormalized, low_thres+eps, name='low_mask')
        zero_const = tf.zeros_like(denormalized)
        denormalized = tf.where(low_mask, zero_const, denormalized)

    return denormalized


def compute_normals(depth, config, conv=False, eps=1e-4):
    # convert NHWC depth to NCHW normal
    with tf.variable_scope("depth_to_normal"):
        intrinsics = tf.constant([[536.628 / 640.0, 536.606 / 480.0, 310.591 / 640.0, 234.759 / 480.0]])
        intrinsics = tf.tile(intrinsics, [config.batch_size, 1])
        depth_real = convertNHWC2NCHW(
            denormalize(depth, low_thres=config.low_thres, up_thres=config.up_thres), name='depth_NCHW')
        normals = depth_to_normals_tf(depth_real, intrinsics)

        if conv:
            kernel_size = 3
            stride = 1
            in_channels = normals.get_shape()[1]
            assert in_channels == 3, 'normals should have 3 channel instead of {}.'.format(in_channels)
            normal_filter = tf.get_variable("filter",
                                            [kernel_size, kernel_size, 1, 1],
                                            dtype=tf.float32,
                                            initializer=tf.constant_initializer(1.0/(kernel_size*kernel_size)),
                                            trainable=False)
            normals1, normals2, normals3 = tf.split(convertNCHW2NHWC(normals, 'normals_NHWC'), 3, axis=3)
            normals1 = tf.nn.conv2d(normals1, normal_filter,
                                    [1, stride, stride, 1], 'SAME', name='normal_conv_r')
            normals2 = tf.nn.conv2d(normals2, normal_filter,
                                    [1, stride, stride, 1], 'SAME', name='normal_conv_g')
            normals3 = tf.nn.conv2d(normals3, normal_filter,
                                    [1, stride, stride, 1], 'SAME', name='normal_conv_b')
            normals = tf.concat([normals1, normals2, normals3], 3)
            unused = tf.less(tf.norm(normals, axis=3), np.sqrt(eps))
            unused = tf.stack([unused]*3, axis=3)
            normals = tf.nn.l2_normalize(normals, 3, epsilon=eps, name='normalize_normals')
            normals = tf.where(unused, tf.zeros_like(normals), normals)
            normals = convertNHWC2NCHW(normals, name='normals_NCHW')

        return normals


def depth_to_normals_tf(depth, intrinsics, scope=None, eps=1e-4):
    """
    :param depth: real depth (B,1,H,W) 
    :param intrinsics: (B,4)
    :return: normals (B,3,H,W)
    """
    with tf.name_scope(scope, 'depth_to_normals_tf', [depth, intrinsics]):
        H, W = depth.shape.as_list()[-2:]
        B = tf.shape(depth)[0]  # config.batch_size
        depth = tf.reshape(depth, [B, H, W])

        # fx_rel = fx_abs / W, cx_real = cx_abs / W
        fx, fy, cx, cy = tf.split(tf.expand_dims(intrinsics, 2), 4, axis=1)  # (B,1,1)
        inv_fx = tf.div(1.0, fx * W)
        inv_fy = tf.div(1.0, fy * H)
        cx = cx * W
        cy = cy * H

        X, Y = tf.meshgrid(tf.range(W), tf.range(H))
        X = tf.cast(tf.tile(tf.expand_dims(X, axis=0), [B, 1, 1]), tf.float32)  # (B,H,W)
        Y = tf.cast(tf.tile(tf.expand_dims(Y, axis=0), [B, 1, 1]), tf.float32)

        x_cord = (X - cx) * inv_fx * depth
        y_cord = (Y - cy) * inv_fy * depth
        p = tf.stack([x_cord, y_cord, depth], axis=3, name='p_3d')  # (B,H,W,3)

        # vector of p_3d in west, south, east, north direction
        p_ctr = p[:, 1:-1, 1:-1, :]
        vw = p_ctr - p[:, 1:-1, 2:, :]
        vs = p[:, 2:, 1:-1, :] - p_ctr
        ve = p_ctr - p[:, 1:-1, :-2, :]
        vn = p[:, :-2, 1:-1, :] - p_ctr
        normal_1 = tf.cross(vs, vw, name='cross_1')  # (B,H-2,W-2,3)
        normal_2 = tf.cross(vn, ve, name='cross_2')
        normal_1 = tf.nn.l2_normalize(normal_1, 3, epsilon=eps)
        normal_2 = tf.nn.l2_normalize(normal_2, 3, epsilon=eps)
        normal = normal_1 + normal_2
        # unused = tf.less(tf.norm(normal, axis=3), np.sqrt(eps))
        # unused = tf.stack([unused] * 3, axis=3)
        normal = tf.nn.l2_normalize(normal, 3, epsilon=eps, name='normal')
        # normal = tf.where(unused, tf.zeros_like(normal), normal)

        paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
        normal = tf.pad(normal, paddings)  # (B,H,W,3)
        normal = convertNHWC2NCHW(normal, 'normal_NCHW')
        return normal


def instance_norm(input):
    with tf.variable_scope("instance_norm"):
        input = tf.identity(input)

        channels = input.get_shape()[3]
        shift = tf.get_variable("shift", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, [1, 2], keep_dims=True)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, shift, scale, variance_epsilon=variance_epsilon,
                                               name='instancenorm')
        return normalized


@slim.add_arg_scope
def lrelu(inputs, leak=0.2, scope="lrelu"):
    """
    For tf > 1.4, use tf.nn.leaky_relu()
    decorate a func with slim.add_arg_scope so that it can be used within an arg_scope in a slim way.
    """
    with tf.variable_scope(scope):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * inputs + f2 * abs(inputs)


def conv_bn_relu(batch_input, kernel_size, stride, out_channels=None):
    with tf.variable_scope("conv_bn_relu"):
        in_channels = batch_input.get_shape()[3]
        if not out_channels: out_channels = in_channels
        filter = tf.get_variable("filter", [kernel_size, kernel_size, in_channels, out_channels],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        convolved = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="SAME")
        normed = batchnorm_u(convolved)
        rectified = tf.nn.relu(normed)
        return rectified, filter


def resize_conv(x, out_ch, k_size, size_factor):
    _, in_h, in_w, in_ch = x.shape.as_list()
    resized = tf.image.resize_nearest_neighbor(x, [in_h * size_factor, in_w * size_factor])
    conv = conv_act(resized, out_ch, k_size, 1)
    return conv


def resize_add_conv_u(input, size_factor, out_ch=None, k_size=3, axis=3, act=tf.nn.relu):
    """
    Bilinear Additive Upsampling. see:
    Wojna, Zbigniew, et al. "The Devil is in the Decoder." arXiv preprint arXiv:1707.05847 (2017).
    """
    with tf.variable_scope("resize_add_conv") as scp:
        _, in_height, in_width, in_ch = input.shape.as_list()
        if out_ch:
            assert in_ch % out_ch == 0, 'cannot add in_ch: {} to out_ch: {}'.format(in_ch, out_ch)
        else:
            out_ch, r = divmod(in_ch, (size_factor * size_factor))
            assert r == 0, 'in_ch: {} not divisible by size_factor^2'.format(in_ch)
        ch_split = in_ch / out_ch
        # bilinear upsample
        resized = tf.image.resize_images(input, [in_height * size_factor, in_width * size_factor])
        stack_list = []
        for i in range(out_ch):
            resized_split = resized[:, :, :, i * ch_split:(i + 1) * ch_split]
            stack_list.append(tf.reduce_sum(resized_split, axis=axis))
        stacked = tf.stack(stack_list, axis=axis)
        filter = tf.get_variable("filter", [k_size, k_size, out_ch, out_ch], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(stacked, filter, [1, 1, 1, 1], padding="SAME")
        if act is not None:
            conv = tf.nn.relu(conv)
        return conv


def conv_concat(input, skip, axis, conv=True):
    with tf.variable_scope("concat"):
        in_ch = input.shape[3]
        if conv:
            skip, _ = conv_bn_relu(skip, 3, 1, out_channels=in_ch)
        return tf.concat([input, skip], axis)


def resize_like(inputs, ref, method='NN'):
    iH, iW = inputs.shape[1], inputs.shape[2]
    rH, rW = ref.shape[1], ref.shape[2]
    if iH == rH and iW == rW:
        return inputs
    if method == 'NN':
        return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])
    elif method == 'BI':
        return tf.image.resize_bilinear(inputs, [rH.value, rW.value])
    else:
        raise NotImplementedError('resize method not implemented yet.')


def residual_block(inputs, ch_out, stride=1, norm_fn=slim.batch_norm, outputs_collections=None, scope=None):
    """
    Residual_block with pre-activation.
    see resnet_model.py for more detailed version.
    """
    with tf.variable_scope(scope, "residual_block") as scp:
        shortcut = tf.identity(inputs, name="shortcut")
        if norm_fn:
            preact = norm_fn(inputs, activation_fn=tf.nn.relu, scope="preact")
        else:
            preact = tf.nn.relu(inputs, name="preact")
        residual = slim.conv2d(preact, ch_out, [3, 3], stride=stride, normalizer_fn=norm_fn, activation_fn=tf.nn.relu,
                               scope="conv1")
        residual = slim.conv2d(residual, ch_out, [3, 3], stride=stride, normalizer_fn=None, activation_fn=None,
                               scope="conv2")
        output = shortcut + residual

    return output


def rand_shift_depth(depths, low_th, up_th, seed=666):
    """
    :param depths: list of depth maps to be randomly shifted together.
    depths values shoud be in range [low_th, up_th]
    :return: list of shifted depth maps
    """
    if len(depths) > 1:
        depth_ref = depths[1]
    else:
        depth_ref = depths[0]

    ref_min = tf.reduce_min(depth_ref)
    ref_max = tf.reduce_max(depth_ref)
    shift_min = low_th - ref_min
    shift_max = up_th - ref_max
    shift_val = tf.random_uniform([], minval=shift_min, maxval=shift_max, seed=seed, name='shift_val')
    depths_shifted = [tf.clip_by_value(d + shift_val, low_th, up_th) for d in depths]
    return depths_shifted


def read_image_from_filename(filename, batch_size, num_threads=4, has_mask=True, has_abd=False,
                             aux_type="JPEG", depth_type=tf.uint16,
                             low_th=500.0, up_th=3000.0, diff_th=5.0,
                             output_height=256, output_width=256,
                             min_after_dequeue=128, use_shuffle_batch=False,
                             rand_crop=True, rand_scale=False, rand_depth_shift=False, rand_flip=True, rand_brightness=True,
                             scope=None):
    """
    :param filename: index csv file for training.
    :param batch_size: 16 or 32 recommended for Titan X.
    :param num_threads: 4 or 8.
    :param has_mask: single channel [0, 255]. offline mask obtained by threshold, instance segmentation or other methods.
    :param has_abd: offline albedo obtained by intrinsic decomposition methods, if False assume uniform albedo.
    :param aux_type: auxiliary(e.g. color) image file type.
    :param depth_type: data type of depth maps.
    :param low_th: limited lower bound of depth range.
    :param up_th: limited upper bound of depth range.
    :param diff_th: threshold to reject bad training pairs with large L1 diff. 
    :param output_height: patch height.
    :param output_width: patch width.
    :param min_after_dequeue: see docs of tf.train.shuffle_batch.
    :param use_shuffle_batch: see docs of tf.train.shuffle_batch.
    :param rand_crop: random cropping patches for training, change cx, cy.
    :param rand_flip: random flipping patches, change cx, cy.
    :param rand_scale: random scaling, change fx, fy, cx, cy.
    :param rand_depth_shift: only shift depth value, no change in intrinsics.
    :param rand_brightness: augment color image.
    :param scope: visualize graphs in tensorboard.
    :return: depth_raw_batch, depth_ref_batch, color_batch, mask_batch, albedo_batch
    """

    with tf.variable_scope(scope, "image_producer"):
        # Load index csv file
        textReader = tf.TextLineReader()
        csv_path = tf.train.string_input_producer([filename], shuffle=True)
        _, csv_content = textReader.read(csv_path)
        if has_mask and has_abd:
            depth_raw_filename, depth_ref_filename, color_filename, mask_filename, albedo_filename = \
                tf.decode_csv(csv_content, [[""], [""], [""], [""], [""]])
        elif has_mask:
            depth_raw_filename, depth_ref_filename, color_filename, mask_filename = \
                tf.decode_csv(csv_content, [[""], [""], [""], [""]])
        else:
            depth_raw_filename, depth_ref_filename, color_filename = \
                tf.decode_csv(csv_content, [[""], [""], [""]])

        # Read and decode image data to tf.float32 tensor
        depth_raw_data = tf.read_file(depth_raw_filename)
        depth_ref_data = tf.read_file(depth_ref_filename)
        color_data = tf.read_file(color_filename)
        depth_raw_im = tf.image.decode_png(depth_raw_data, channels=1, dtype=depth_type)
        depth_ref_im = tf.image.decode_png(depth_ref_data, channels=1, dtype=depth_type)
        if has_mask:
            mask_data = tf.read_file(mask_filename)
            mask = tf.image.decode_png(mask_data, channels=1) / 255
            mask = tf.cast(mask, tf.float32)
        if has_abd:
            albedo_data = tf.read_file(albedo_filename)
            albedo_im = tf.image.decode_png(albedo_data, channels=1)
            albedo_im = tf.cast(albedo_im, tf.float32)

        if aux_type == "JPEG":
            color_im = tf.image.decode_jpeg(color_data, channels=1)
        elif aux_type == "PNG":
            color_im = tf.image.decode_png(color_data, channels=1)
        else:
            raise NotImplementedError("unsupport auxiliary image type for now!")
        depth_raw_im = tf.cast(depth_raw_im, tf.float32)
        depth_ref_im = tf.cast(depth_ref_im, tf.float32)
        color_im = tf.cast(color_im, tf.float32)
        # color_im = tf.image.resize_images(color_im, depth_raw_shape[:2], method=2)  # return float Tensor

        # Concat all images in channel axis to randomly crop together
        if has_mask and has_abd:
            concated_im = tf.concat([depth_raw_im, depth_ref_im, color_im, mask, albedo_im], axis=2)
            n_concat = 5
        elif has_mask:
            concated_im = tf.concat([depth_raw_im, depth_ref_im, color_im, mask], axis=2)
            n_concat = 4
        else:
            concated_im = tf.concat([depth_raw_im, depth_ref_im, color_im], axis=2)
            n_concat = 3

        # Prepose rand_crop here to reduce unnecessary computation of subsequent data augmentations.
        if rand_crop:
            concated_im = tf.random_crop(concated_im, [output_height, output_width, n_concat])
            # concated_im = tf.image.crop_to_bounding_box(concated_im, 80, 250, output_height, output_width)  # dbg
        else:
            concated_im = tf.image.resize_image_with_crop_or_pad(concated_im, output_height, output_width)

        if has_mask and has_abd:
            depth_raw_im, depth_ref_im, color_im, mask, albedo_im = tf.split(concated_im, n_concat, axis=2)
        elif has_mask:
            depth_raw_im, depth_ref_im, color_im, mask = tf.split(concated_im, n_concat, axis=2)
        else:
            depth_raw_im, depth_ref_im, color_im = tf.split(concated_im, 3, axis=2)

        # Filter bad inputs use diff_mean or mse
        n_holes = tf.count_nonzero(tf.less(depth_ref_im, tf.constant(50.0)), dtype=tf.float32)
        diff = tf.abs(tf.subtract(depth_raw_im, depth_ref_im, name='diff'))
        diff = tf.where(diff<up_th/10, diff, tf.zeros_like(diff))
        diff_mean = tf.reduce_mean(diff, name='diff_mean')
        # mse = tf.reduce_mean(tf.square(diff), name='mse')
        enqueue_cond = tf.logical_and(tf.less(n_holes, output_height*output_width*2/3), tf.less(diff_mean, diff_th))

        def zero_img():
            return tf.constant(0, shape=[0, output_height, output_width, n_concat])

        def one_img():
            # Data augmentation: rand_flip, rand_scale and rand_depth_shift on filtered patches.
            raw = tf.clip_by_value(depth_raw_im, low_th, up_th)
            ref = tf.clip_by_value(depth_ref_im, low_th, up_th)
            if rand_brightness:
                color = tf.image.random_brightness(color_im, 20)
            else:
                color = color_im
            if rand_depth_shift:
                raw, ref = rand_shift_depth([raw, ref], low_th, up_th)

            if has_mask and has_abd:
                im = tf.concat([raw, ref, color, mask, abd], axis=2)
            elif has_mask:
                im = tf.concat([raw, ref, color, mask], axis=2)
            else:
                im = tf.concat([raw, ref, color], axis=2)

            if rand_flip:
                im = tf.image.random_flip_left_right(im)
            if rand_scale:
                pass
            return tf.expand_dims(im, 0)

        concated_im = tf.cond(enqueue_cond, one_img, zero_img)

        ## Pass the 4D batch tensors to a batching op at the end of input data queue
        # shuffle_batch creates a shuffling queue with dequeue op and enqueue QueueRunner
        # min_after_dequeue defines how big a buffer we will randomly sample from
        # bigger means better shuffling but slower start up and more memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        # determines the maximum we will prefetch.
        # capacity = min_after_dequeue + (num_threads + small_safety_margin) * batch_size
        if use_shuffle_batch:
            capacity = min_after_dequeue + (num_threads + 1) * batch_size
            im_batch = tf.train.shuffle_batch(
                [concated_im],
                batch_size=batch_size,
                capacity=capacity,
                enqueue_many=True,
                num_threads=num_threads,
                min_after_dequeue=min_after_dequeue,
                allow_smaller_final_batch=True,
                name="shuffle_batch")
        else:
            im_batch = tf.train.batch(
                [concated_im],
                batch_size=batch_size,
                num_threads=num_threads,
                allow_smaller_final_batch=True,
                enqueue_many=True,
                name="batch")

        # Split concatenated data
        if has_mask and has_abd:
            depth_raw_batch, depth_ref_batch, color_batch, mask_batch, albedo_batch = tf.split(im_batch, n_concat, axis=3)
        elif has_mask:
            depth_raw_batch, depth_ref_batch, color_batch, mask_batch = tf.split(im_batch, n_concat, axis=3)
        else:  # get mask only from ref(after clip, outliers are equal to low_th)
            depth_raw_batch, depth_ref_batch, color_batch = tf.split(im_batch, n_concat, axis=3)
            mask_batch = tf.cast(tf.not_equal(depth_ref_batch, low_th), tf.float32, name='mask_batch')  # 0.0 or 1.0

        # Normalize depth and color maps
        with tf.name_scope('normalize'):
            thres_range = (up_th - low_th) / 2.0
            depth_raw_batch = (depth_raw_batch - low_th) / thres_range
            depth_raw_batch = tf.subtract(depth_raw_batch, 1.0, name='raw_batch')  # [low,up]->[-1,1]
            depth_ref_batch = (depth_ref_batch - low_th) / thres_range
            depth_ref_batch = tf.subtract(depth_ref_batch, 1.0, name='ref_batch')  # [low,up]->[-1,1]
            color_batch = color_batch * mask_batch / 127.0
            color_batch = tf.subtract(color_batch, 1.0, name='aux_batch')  # [0,255]->[-1,1]
            if has_abd:
                albedo_batch = albedo_batch / 127.0  # offline estimated albedo from RGB, [0,255]->[0,2]
            else:
                albedo_batch = None
        # dbg: return and show last diff_mean in batch
        return depth_raw_batch, depth_ref_batch, color_batch, mask_batch, albedo_batch, diff_mean
