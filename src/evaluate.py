import os
import csv
import glob
import time
import shutil
import argparse
import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
slim = tf.contrib.slim

import ops
import utils
import model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dnnet", type=str)
    parser.add_argument("--dtnet", type=str)
    parser.add_argument("--loop", dest="loop", action="store_true")
    parser.add_argument("--gpu", dest="gpu", action="store_true")

    parser.add_argument("--sample_dir", type=str, default="sample/")
    parser.add_argument("--checkpoint_dir", type=str, default="log/")
    parser.add_argument("--csv_path", type=str, default="dataset/test.csv")

    parser.add_argument("--image_size", type=int, default=256),
    parser.add_argument("--low_thres", type=float, default=500.0),
    parser.add_argument("--up_thres", type=float, default=3000.0),

    parser.add_argument("--num_threads", type=int, default=1)

    return parser.parse_args()


def build_model(im_height, im_width, config):
    # batch_size = 1
    depth_in = tf.placeholder(tf.float32, [1, im_height, im_width, 1])
    paddings = tf.constant([[0, 0], [8, 8], [8, 8], [0, 0]])  # avoid edge vanish
    depth_in_pad = tf.pad(depth_in, paddings, mode='SYMMETRIC', name='depth_in_padded')
    color = tf.placeholder(tf.float32, [1, im_height, im_width, 1])
    color_pad = tf.pad(color, paddings, mode='SYMMETRIC', name='color_in_padded')
    is_training = tf.placeholder(tf.bool, name="is_training")

    dnnet = dtnet = None
    if config.dnnet == "None":
        dnnet = None
    elif config.dnnet == "base":
        dnnet = model.base
    elif config.dnnet == "unet":
        dnnet = model.unet
    elif config.dnnet == "convResnet":
        dnnet = model.convResnet
    elif config.dnnet == "UResnet":
        dnnet = model.UResnet
    else:
        raise NotImplementedError("There is no such {} dnnet".format(config.dnnet))
    if config.dtnet == "None":
        dtnet = None
    elif config.dtnet == "hypercolumn":
        dtnet = model.hypercolumn
    else:
        raise NotImplementedError("There is no such {} dtnet".format(config.dtnet))

    if config.gpu:
        device = "/gpu:0"
    else:
        device = "/cpu:0"

    with tf.device(device):
        if dnnet:
            depth_dn, end_pts, weight_vars = dnnet(depth_in_pad, is_training, aux=None, scope="dn_net")
        else:
            depth_dn = None

        if dtnet:
            depth_dt, end_pts, weight_vars = dtnet(depth_dn if depth_dn is not None else depth_in_pad, color_pad)
        else:
            depth_dt = None

        if depth_dn is not None:
            depth_dn = tf.reshape(depth_dn[:, 8:-8, 8:-8, :], [im_height, im_width])
        if dtnet is not None:
            depth_dt = tf.reshape(depth_dt[:, 8:-8, 8:-8, :], [im_height, im_width])

    return {"depth_in": depth_in, "color": color, "is_training": is_training,
            "depth_dn": depth_dn,
            "depth_dt": depth_dt}


def wait_for_new_checkpoint(checkpoint_dir, history):
    while True:
        path = tf.train.latest_checkpoint(checkpoint_dir)
        if not path in history:
            break
        time.sleep(300)

    history.append(path)
    return path


def load_from_checkpoint(sess, path, exclude=None):
    saver = tf.train.Saver()
    saver.restore(sess, path)


def loop_body_patch(it, ckpt_path, depth_in, depth_ref, color, mask, config):
    """
    :param depth_ref: unused yet. offline quantitative evaluation of depth_dt. 
    """
    print(time.ctime())
    print("Load checkpoint: {}".format(ckpt_path))

    h, w = depth_in.shape[:2]
    low_thres = config.low_thres
    up_thres = config.up_thres
    thres_range = (up_thres - low_thres) / 2.0

    params = build_model(h, w, config)
    # ckpt_step = ckpt_path.split("/")[-1]

    sess = tf.Session()
    load_from_checkpoint(sess, ckpt_path)

    depth_dn_im, depth_dt_im = sess.run([params["depth_dn"], params["depth_dt"]],
                                        feed_dict={params["depth_in"]: depth_in.reshape(1, h, w, 1),
                                                   params["color"]: color.reshape(1, h, w, 1),
                                                   params["is_training"]: False})

    depth_dn_im = (((depth_dn_im + 1.0) * thres_range + low_thres) * mask).astype(np.uint16)
    depth_dt_im = (((depth_dt_im + 1.0) * thres_range + low_thres) * mask).astype(np.uint16)

    utils.save_image(depth_dn_im, config.sample_dir, "frame_{}_dn.png".format(it))
    utils.save_image(depth_dt_im, config.sample_dir, "frame_{}_dt.png".format(it))

    tf.reset_default_graph()
    print("saving img {}.".format(it))


def loop_body_whole(it, ckpt_path, raw_arr, gt_arr, rgb_arr, H, W, config):
    """
    forward input raw_array patches seperately, then h_stack and v_stack these patches into whole.
    """
    print(time.ctime())
    print("Load checkpoint: {}".format(ckpt_path))

    h, w = raw_arr[0][0].shape[:2]
    low_thres = config.low_thres
    up_thres = config.up_thres
    thres_range = (up_thres - low_thres) / 2.0

    params = build_model(h, w, config)
    ckpt_step = ckpt_path.split("/")[-1]

    sess = tf.Session()
    load_from_checkpoint(sess, ckpt_path)

    dn_arr = []
    for i, h_list in enumerate(raw_arr):
        dn_h_list = []
        for j in range(len(h_list)):
            depth_dn_patch, depth_dt_patch = sess.run([params["depth_dn"], params["depth_dt"]],
                                                feed_dict={params["depth_in"]: depth_in.reshape(1, h, w, 1),
                                                           params["color"]: color.reshape(1, h, w, 1),
                                                           params["is_training"]: False})
            dn_h_list.append(depth_dn_patch)
        dn_arr.append(dn_h_list)
    dn_im = utils.stack_patch(dn_arr, H, W)
    dn_im = ((dn_im + 1.0) * thres_range + low_thres).astype(np.uint16)
    utils.save_image(dn_im, config.sample_dir, "frame_{}_dn.png".format(it))

    tf.reset_default_graph()
    print("saving img {}.".format(it))


def loop_body_patch_time(sess, name, params, depth_in, color, mask, config):
    """
    Build test graph only once, more efficient when training phase is finished.
    :return: time elapsed for one batch of testing image.
    """
    h, w = depth_in.shape[:2]
    depth_in = depth_in.reshape(1, h, w, 1)
    color = color.reshape(1, h, w, 1)

    low_thres = config.low_thres
    up_thres = config.up_thres
    thres_range = (up_thres - low_thres) / 2.0

    t_start = time.time()
    feed_dict = {params["depth_in"]: depth_in,
                 params["color"]: color,
                 params["is_training"]: False}
    depth_dn_im = depth_dt_im = None
    if (params["depth_dn"] is not None) and (params["depth_dt"] is not None):
        depth_dn_im, depth_dt_im = sess.run([params["depth_dn"], params["depth_dt"]],
                                            feed_dict=feed_dict)
    elif (params["depth_dn"] is not None):
        depth_dn_im = sess.run(params["depth_dn"], feed_dict=feed_dict)
    elif (params["depth_dt"] is not None):
        depth_dt_im = sess.run(params["depth_dt"], feed_dict=feed_dict)

    print("saving img {}.".format(name))
    if depth_dn_im is not None:
        depth_dn_im = (((depth_dn_im + 1.0) * thres_range + low_thres) * mask).astype(np.uint16)
        utils.save_image(depth_dn_im, config.sample_dir, "dn_{}".format(name))
    if depth_dt_im is not None:
        depth_dt_im = (((depth_dt_im + 1.0) * thres_range + low_thres) * mask).astype(np.uint16)
        utils.save_image(depth_dt_im, config.sample_dir, "dt_{}".format(name))
    t_end = time.time()
    return (t_end - t_start)


def loop(data_info, config, split_stack=True, test_time=True):
    up_thres, low_thres = config.up_thres, config.low_thres
    all_ims = []
    for info in data_info:
        depth_in_path, depth_ref_path, color_path, mask_path = info
        name = os.path.basename(depth_in_path)
        raw = Image.open(depth_in_path)
        gt = Image.open(depth_ref_path)
        rgb = Image.open(color_path).convert('L').resize(raw.size)
        mask = Image.open(mask_path)
        assert raw.size == gt.size, 'gt size not match raw size!'

        # Do center crop here
        if not split_stack:
            if config.image_size < min(raw.size):
                raw, gt, rgb, mask = utils.center_crop(raw, gt, rgb, mask, config.image_size)
            elif config.image_size >= max(raw.size):
                raw, gt, rgb, mask = utils.center_pad(raw, gt, rgb, mask, config.image_size)
            else:
                raise NotImplementedError('invalid config.image_size.')

        mask = np.array(mask, dtype=np.float32) / 255.0
        rgb = np.array(rgb, dtype=np.float32) / (127.0 - 1.0) * mask
        thres_range = (up_thres - low_thres) / 2.0
        raw = np.clip(np.array(raw, dtype=np.float32), low_thres, up_thres)
        gt = np.clip(np.array(gt, dtype=np.float32), low_thres, up_thres)
        if config.dnnet == "bilateral":
            raw = cv2.bilateralFilter(raw, 9, 75, 75)
        raw = (raw - low_thres) / thres_range - 1.0
        gt = (gt - low_thres) / thres_range - 1.0

        if split_stack:
            raw_arr, H, W = utils.split_patch(raw, config.image_size)
            gt_arr, _, _ = utils.split_patch(gt, config.image_size)
            rgb_arr, _, _ = utils.split_patch(rgb, config.image_size)
            all_ims.append((name, raw_arr, gt_arr, rgb_arr))
        else:
            all_ims.append((name, raw, gt, rgb, mask))

    ckpt_history = list()
    if test_time and not split_stack:
        path = tf.train.latest_checkpoint(config.checkpoint_dir)
        print("Load checkpoint: {}".format(path))
        params = build_model(config.image_size, config.image_size, config)
        sess = tf.Session()
        load_from_checkpoint(sess, path)

        tt_time = 0.0
        history_len = 3
        t_history = np.zeros(history_len, dtype=np.float32)
        for i, (name, raw, _, rgb, mask) in enumerate(all_ims):
            t_elapsed = loop_body_patch_time(sess, name, params, raw, rgb, mask, config)
            t_history[i % history_len] = t_elapsed
            tt_time += t_elapsed
            avg_time = 1000 * tt_time / (i + 1)  # ms
            mv_avg_time = 1000 * np.mean(t_history)
            print('iter {} | tt_time: {:.4f}s; avg_time: {:.2f}; mv_avg_time: {:.2f}'.format(i+1, tt_time, avg_time, mv_avg_time))
        tf.reset_default_graph()
    else:
        while True:
            # Wait until new checkpoint exist when training phase is not finished.
            path = wait_for_new_checkpoint(config.checkpoint_dir, ckpt_history)
            print("Loading from checkpoint: {}".format(path))

            if split_stack:
                print('evaluating {} imgs'.format(len(all_ims)))
                for i, (name, raw_arr, gt_arr, rgb_arr) in enumerate(all_ims):
                    loop_body_whole(i, path, raw_arr, gt_arr, rgb_arr, H, W, config)
            else:
                for i, (name, raw, gt, rgb, mask) in enumerate(all_ims):
                    loop_body_patch(i, path, raw, gt, rgb, mask, config)

            if not config.loop: break


def main(config):
    print('evaluating csv file: {}'.format(config.csv_path))
    with open(config.csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        data_info = [row for row in reader]

        if not os.path.exists(config.sample_dir):
            os.makedirs(config.sample_dir)

        loop(data_info, config, split_stack=False)


if __name__ == "__main__":
    config = parse_args()
    main(config)
