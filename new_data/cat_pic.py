#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 读取图片，蒙版（黑色代表不选中，白色代表选中）
# photoshop 狗都不用
# 参考博客
# 这个是没有透明度的
# https://blog.csdn.net/qq_29391809/article/details/117394512
import cv2
import os
import numpy as np
img_item=r"C:\Users\brighten\Desktop\splicing\1\COD10K-CAM-1-Aquatic-4-Crocodile-115.jpg"
mask_file=r"C:\Users\brighten\Desktop\splicing\1"
mask_item=r"COD10K-CAM-1-Aquatic-4-Crocodile-115.png"
bg_path=img_item
img = cv2.imread(img_item)
# img = cv2.resize(img, (960, 1280), interpolation=cv2.INTER_LINEAR)
mask = cv2.imread(os.path.join(mask_file, mask_item), cv2.IMREAD_UNCHANGED)
mask=np.expand_dims(mask, axis=2)
bg = cv2.imread(bg_path)
bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

bg_mask = (np.ones(img.shape, dtype='uint8') * 255) - mask # mask求反，用一个255的矩阵减mask
mask = mask.astype("float") # 类型转化：uint8->float
mask = mask / 255 # 求透明度
img = img * mask # 用透明度乘图片
bg = bg * (bg_mask/255) # 背景同样
img = img.astype("uint8")
bg = bg.astype("uint8")
stacked = cv2.addWeighted(img, 1, bg, 1, 0)  # 叠加两图
cv2.imshow('r1', stacked)
cv2.waitKey(0)
