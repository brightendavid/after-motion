#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 有读取灰度，读取4通道透明度函数，直接修改透明度，结果必须保存png(可以具有透明度层)
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def imread_four_channel(img):
    """
    img : rgb
    """
    b_channel, g_channel, r_channel = cv.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # alpha通道每个像素点区间为[0,255], 0为完全透明，255是完全不透明
    img_BGRA = cv.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA


def gen_touming_pic(img, gray):
    a, b, _ = img.shape
    for i in range(a):
        for j in range(b):
            if gray[i, j] > 100:
                img[i, j, 3] = 0
    return img


if __name__ == '__main__':
    gray = cv.imread("./1.png", 0)
    gray = np.where(gray < 100, 0, 255)
    img = cv.merge((gray, gray, gray))
    pic2 = imread_four_channel(img)
    pic2 = gen_touming_pic(pic2, gray)
    cv.imwrite("touming_sig.png", pic2)
