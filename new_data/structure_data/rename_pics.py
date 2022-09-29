#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2 as cv
from PIL import Image


def Rename(src_dir, gt_dir,save_src,save_gt, t):
    for index, item in enumerate(os.listdir(src_dir)):
        # src流
        src_path = os.path.join(src_dir, item)  # src 的路径
        src = Image.open(src_path)
        name = "PS_" + str(t)+".bmp"
        save_path = os.path.join(save_src, name)
        print(save_path)
        src.save(save_path)

        # gt流
        src_path = os.path.join(gt_dir, item)  # src 的路径
        src = Image.open(src_path)
        name = "PS_" + str(t)+".bmp"
        save_path = os.path.join(save_gt, name)
        src.save(save_path)

        t += 1


if __name__ == '__main__':
    src_dir = r"C:\Users\brighten\Desktop\gen_new_data\result"
    gt_dir = r"C:\Users\brighten\Desktop\gen_new_data\result_wenli"
    # save_src=r"C:\Users\brighten\Desktop\gen_new_data\src"
    # save_gt=r"C:\Users\brighten\Desktop\gen_new_data\gt"

    save_src=r"C:\Users\brighten\Desktop\Texture filtering\src"
    save_gt=r"C:\Users\brighten\Desktop\Texture filtering\gt"
    t = 2878
    Rename(src_dir, gt_dir,save_src,save_gt, t)
