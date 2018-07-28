import os
import time
import numpy as np
import cv2
import sys
from os import path as osp
from glob import glob
from PIL import Image


class Loader():
    def __init__(self, folder, backend='.png'):
        self.folder = folder
        self.backend = backend
        self.filenames = self.load_filenames()

    def load_filenames(self):
        files = glob(osp.join(self.folder, '*{}'.format(self.backend)))
        filenames = [n.split('/')[-1][:-len(self.backend)] for n in files]
        return filenames

    def load_image(self, name):
        im = cv2.imread(osp.join(self.folder, name + self.backend), cv2.IMREAD_UNCHANGED)
        return im

    def show_image(self, img):
        im = Image.fromarray(img).convert('L')
        im.show()

    def save_image(self, name, img):
        cv2.imwrite(osp.join(self.folder, name + self.backend), img)

    def get_filenames(self):
        return self.filenames

def fill_hole(depth_im):
    """Close operation: dilate then erode"""
    kernel = np.ones((12, 12),np.uint16)
    depth_im_de = cv2.morphologyEx(depth_im, cv2.MORPH_CLOSE, kernel)  # dilate_erode
    mask = np.greater(depth_im, 0)
    depth_im_de = np.where(mask, depth_im, depth_im_de)
    #for x in xrange(depth_im.shape[0]):
    #    for y in xrange(depth_im.shape[1]):
    #        if depth_im[x, y]>0:
    #            depth_im_de[x, y] = depth_im[x, y]
    return depth_im_de

def rm_dot(depth_im, mask=None):
    """Open operation: erode then dilate"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    depth_im_ed = cv2.morphologyEx(depth_im, cv2.MORPH_OPEN, kernel)  # erode_dilate
    if not mask:
        mask = np.greater(depth_im, 0)
    depth_im_ed = np.where(mask, depth_im, depth_im_ed)
    return depth_im_ed

def get_mask(depth_im_de, gt):
    d = depth_im_de>0
    g = rm_dot(gt)  # gt may have isolated dots at volume edge
    g = gt>0
    ans = d.astype('uint8') + g.astype('uint8')
    ans = (ans==2)
    return ans.astype('uint8')*255

def get_mask_test(depth_im_de):
    d = depth_im_de>0
    return d.astype('uint8')*255

if __name__ == '__main__':
    print "Dilate erode now..."
    path = os.getcwd()
    folder_name = sys.argv[1]
    test_flag = False
    if len(sys.argv) > 2:
        test_flag = (sys.argv[2] == '-test')

    for group in os.listdir(osp.join(path, folder_name)):
        if not os.path.isdir(osp.join(path, folder_name, group)): continue
        depth = Loader(osp.join(path, folder_name, group, 'depth_map'), backend='.png')
        if not test_flag:
            gt = Loader(osp.join(path, folder_name, group, 'high_quality_depth'), backend='.png')
        if not os.path.isdir(osp.join(path, folder_name, group, 'depth_filled')):
            os.mkdir(osp.join(path, folder_name, group, 'depth_filled'))
        if not os.path.isdir(osp.join(path, folder_name, group, 'mask')):
            os.mkdir(osp.join(path, folder_name, group, 'mask'))
        for i in depth.get_filenames():
            if not os.path.isfile(osp.join(path, folder_name, group, 'depth_map', i + '.png')): continue
            start = time.time()
            depth_im = depth.load_image(i)
            depth_im_de = fill_hole(depth_im)
            if test_flag:
                mask = get_mask_test(depth_im)  # test only get mask from raw, not limited by volume.
            else:
                gt_im = gt.load_image(i)
                mask = get_mask(depth_im_de, gt_im)
            depth.save_image(osp.join('..', 'depth_filled', i), depth_im_de)
            depth.save_image(osp.join('..', 'mask', i), mask)
            print "%s time:%f"%(i, time.time()-start)


