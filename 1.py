#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def img_cut(img):
    # 将图像切为圆
    a,b,_ = img.shape
    r = min(a//2,b//2)
    print(img.size)
    for i in range(a):
        for j in range(b):
            if (i-a//2)*(i-a//2)+(j-b//2)*(j-b//2)<r*r:
                img[i,j,3] = 255
    return img

if __name__ == '__main__':

    img = cv.imread(r"./1.png")
    print(img.shape)
    b_channel, g_channel, r_channel = cv.split(img)
    alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)   # alpha通道每个像素点区间为[0,255], 0为完全透明，255是完全不透明
    img_BGRA = cv.merge((b_channel, g_channel, r_channel, alpha_channel))

    print(img_BGRA.shape)
    # cv.imwrite("3.png",img)
    img = img_cut(img_BGRA)
    print(img.shape)
    cv.imwrite("toum.png",img)