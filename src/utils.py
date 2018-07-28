import os
import sys
import scipy.misc
import skimage.measure
import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from PIL import Image


def save_image(im, directory, name, depth='uint16'):
    path = os.path.join(directory, name)
    if depth == 'uint16':
        im = im.astype(np.uint16)
        print(np.max(im), np.min(im))
        im = scipy.misc.toimage(im, high=np.max(im), low=np.min(im), mode='I')
        im.save(path)
    else:
        scipy.misc.imsave(path, im)
    return


def compare_psnr(true_im, pred_im):
    return skimage.measure.compare_psnr(true_im, pred_im)


def compare_ssim(true_im, pred_im):
    return skimage.measure.compare_ssim(true_im, pred_im)


def flush_stdout():
    sys.stdout.write("\x1b[1A")
    sys.stdout.write("\x1b[2K")


def add_noise_tf(data, sigma, sess):
    noise = sigma / 255.0 * sess.run(tf.truncated_normal(data.shape))
    return (data + noise)


def add_noise_np(data, sigma):
    noise = sigma * np.random.normal(size=data.shape)
    return (data + noise)


def cut_edge(im, edge_h=20, edge_w=20):
    height, width = im.shape[:2]
    im = im[edge_h:height-edge_h, edge_w:width-edge_w]
    return im, im.shape[0], im.shape[1]


def center_crop(art, ref, aux, mask, patch_size):
    W, H = art.size
    h_min = (H - patch_size) // 2
    w_min = (W - patch_size) // 2
    assert (h_min > 0 and w_min > 0), 'input image {} < crop_size {}' \
        .format((H, W), patch_size)
    art = np.array(art, dtype=np.float32)[h_min:h_min + patch_size, w_min:w_min + patch_size]
    ref = np.array(ref, dtype=np.float32)[h_min:h_min + patch_size, w_min:w_min + patch_size]
    aux = np.array(aux, dtype=np.float32)[h_min:h_min + patch_size, w_min:w_min + patch_size]
    mask = np.array(mask, dtype=np.float32)[h_min:h_min + patch_size, w_min:w_min + patch_size]
    return art, ref, aux, mask


def center_pad(art, ref, aux, mask, patch_size):
    art = np.array(art, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)
    aux = np.array(aux, dtype=np.float32)
    mask = np.array(mask, dtype=np.float32)
    H, W = art.shape[:2]
    height_diff = patch_size - H
    width_diff = patch_size - W
    offset_pad_height = height_diff // 2
    offset_pad_width = width_diff // 2
    assert (offset_pad_height >= 0 and offset_pad_width >= 0), 'no need to pad.'
    pad_width = [(offset_pad_height, height_diff-offset_pad_height),
                 (offset_pad_width, width_diff-offset_pad_width)]
    art = np.pad(art, pad_width+[(0,0)]*(len(art.shape)-2), 'constant', constant_values=0.0)
    ref = np.pad(ref, pad_width+[(0,0)]*(len(ref.shape)-2), 'constant', constant_values=0.0)
    aux = np.pad(aux, pad_width+[(0,0)]*(len(aux.shape)-2), 'constant', constant_values=0.0)
    mask = np.pad(mask, pad_width+[(0,0)]*(len(aux.shape)-2), 'constant', constant_values=0.0)
    return art, ref, aux, mask


def split_patch(im, patch_size):
    """
    Split image to several patches of size `patch_size`, remaining is preserved.
    :return list of ndarrays.
    """
    if not isinstance(im, np.ndarray):
        try: im = np.array(im)
        except: raise ValueError('input im must be np.ndarray!')

    H, W = im.shape[:2]
    patch_h = patch_w = patch_size
    hq, hr = divmod(H, patch_h)
    wq, wr = divmod(W, patch_w)

    h_pad = w_pad = 0
    h_splits, w_splits = hq, wq
    if hr:
        h_pad = patch_h - hr
        h_splits = hq + 1
    if wr:
        w_pad = patch_w - wr
        w_splits = wq + 1
    pad_width = [(0, h_pad), (0, w_pad)] + [(0, 0)] * (len(im.shape)-2)
    im = np.pad(im, pad_width, 'constant', constant_values=0)
    im_array = np.split(im, h_splits, axis=0)
    for i, v in enumerate(im_array):
        im_array[i] = np.split(v, w_splits, axis=1)

    return im_array, H, W


def stack_patch(im_array, H, W):
    """
    Stack split patches to whole image.
    :param im_array: list of ndarrays.
    :return: ndarray of whole image.
    """
    hq = len(im_array)
    im_h_list = [np.hstack(im_array[i]) for i in range(hq)]
    im_pad = np.vstack(im_h_list)
    return im_pad[:H, :W]


def pseudo_color(in_dir, out_dir, up=3000.0, low=0.0, suffix='.png'):
    """
    Convert depth map to pseudo color map.
    """
    im_list = glob.glob1(in_dir, '*.png')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    th_2 = (low + up) / 2.0
    th_1 = (low + th_2) / 2.0
    th_3 = (th_2 + up) / 2.0

    for im_name in im_list:
        im = np.array(Image.open(os.path.join(in_dir, im_name)), dtype=np.float32)
        # print(np.min(im), np.max(im))
        im[im > 2000.0] = 2000.0
        im[im < 520.0] = 0.0
        # im_min = np.min(im)
        # im_max = np.max(im)
        # im = (up - low) * (im - im_min) / (im_max - im_min) + low

        mask_1 = np.less(im, th_1)
        mask_2 = np.less(im, th_2)
        mask_3 = np.less(im, th_3)
        zero_const = np.zeros_like(im)
        one_const = np.ones_like(im)

        red_ch = np.where(mask_2, zero_const, one_const)
        mask = np.logical_and(np.logical_not(mask_2), mask_3)
        red_ch = np.where(mask, (im - th_2) / (th_3 - th_2), red_ch)
        green_ch = np.where(mask_1, (im - low) / (th_1 - low), one_const)
        green_ch = np.where(mask_3, green_ch, one_const - (im - th_3) / (up - th_3))
        blue_ch = np.where(mask_2, one_const, zero_const)
        mask = np.logical_and(np.logical_not(mask_1), mask_2)
        blue_ch = np.where(mask, one_const - (im - th_1) / (th_2 - th_1), blue_ch)

        color = np.stack([red_ch, green_ch, blue_ch], axis=2)
        # print(np.max(color), np.min(color))
        color = (color*255).astype(np.uint8)
        im = Image.fromarray(color)
        im_name = im_name.split('.png')[0] + suffix
        im.save(os.path.join(out_dir, im_name))
    return
