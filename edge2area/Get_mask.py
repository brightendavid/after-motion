# -*- coding: utf-8 -*-
import os

import cv2 as cv
import numpy as np
from PIL import Image


def take_gt_mask(src_path, gt_path, save_path):
    for index, item in enumerate(os.listdir(src_path)):
        _src_path = os.path.join(src_path, item)
        _save_path = os.path.join(save_path, item)
        _gt_path = os.path.join(gt_path, item)
        src = Image.open(_src_path)
        gt = Image.open(_gt_path)
        src = np.array(src)
        gt = np.array(gt)
        result = gtmask_one(src, gt)
        cv.imwrite(_save_path, result)


def gtmask_one(src, gt):
    src= np.where(src<100,0,255)
    gt = np.where(gt > 99, 1, 0)
    gt = cv.merge([gt]).astype('uint8')
    dilate_window = 5
    kernel = np.ones((dilate_window, dilate_window), np.uint8)
    band = cv.dilate(gt, kernel)
    band = np.array(band, dtype='uint8')
    src = np.where(band == 1, src, 0)
    return src

    # cv.imshow("src", src)
    # cv.waitKey(0)


if __name__ == '__main__':
    src = cv.imread(r'./pictures/Tp_D_CRN_M_N_txt00063_txt00017_10835.jpg', 0)
    gt = cv.imread(r'./pictures/Tp_D_CRN_M_N_txt00063_txt00017_10835_gt.png', 0)
    # print(src.shape)
    # print(gt.shape)
    src2 = gtmask_one(src, gt)
    cv.imwrite(r"./pictures/test.bmp",src2)
    # take_gt_mask(src_path='', gt_path='', save_path='')
