#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
def divide_demo(m1, m2):  # 图像减法
    dst = cv.divide(m1, m2)
    cv.imshow('divide_demo', dst)
    cv.waitKey(0)



x = Image.open(r"C:\Users\brighten\Desktop\COD10K高清\1\COD10K-CAM-1-Aquatic-4-Crocodile-115.jpg")

y = Image.open(r"C:\Users\brighten\Desktop\COD10K高清\1\COD10K-CAM-1-Aquatic-4-Crocodile-115篡改.png")


x = cv.cvtColor(np.asarray(x),cv.COLOR_RGB2BGR)

y = cv.cvtColor(np.asarray(y),cv.COLOR_RGB2BGR)
print(x.shape)
print(y.shape)
divide_demo(y, x)
