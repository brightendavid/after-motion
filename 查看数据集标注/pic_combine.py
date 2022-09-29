#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
对gt和篡改图进行融合，比较标注的位置是否合理
"""
import os
from PIL import Image
import cv2 as cv
import numpy as np


def pic_pic(src1, src2, save_path):
    # tampered pic 和 标注  只要把255 的部分给它
    # 篡改区域  82 三个
    # 篡改边缘  255 三个
    # 非篡改边缘  135  三个
    # 非篡改区域  0 三个
    ones=np.ones_like(src1)*255
    ones2 = np.ones_like(src1) * 50
    src_cuangaiby = np.where(src2 == 255, ones, src1)
    src_cuangaiquyu = np.where(src2 == 50, ones2, src_cuangaiby)
    cv.imwrite(save_path,src1)
    save_path3 = save_path[:-4] + "by" + save_path[-4:]
    cv.imwrite(save_path3,src_cuangaiby)
    save_path2=save_path[:-4]+"quyu"+ save_path[-4:]
    cv.imwrite(save_path2, src_cuangaiquyu)


"""     if type[0] == 'CASIA':
            gt_path = name.split('.')[0] + '_gt' + '.png'
            gt_path = os.path.join(self.casia_gt_path, gt_path)     
        elif type[0] == 'COLUMBIA':
            gt_path = name.split('.')[0] + '_edgemask' + '.bmp'
            gt_path = os.path.join(self.columbia_gt_path, gt_path)
            print(gt_path)
            
        elif type[0] == 'COVERAGE':
            gt_path = name.replace('t', 'forged')
            gt_path = os.path.join(self.coverage_gt_path, gt_path)
"""


def read_dir(src_dir, gt_dir, save_dir):
    for src_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, src_name)
        name = src_name
        if 'casia' in src_dir:
            name = name.split('.')[0] + '_gt' + '.png'
            gt_path = os.path.join(gt_dir, name)
        elif 'Columbia' in src_dir:
            name = name.split('.')[0] + '_edgemask' + '.bmp'
            gt_path = os.path.join(gt_dir, name)
        elif 'coverage' in src_dir:
            name = name.replace('t', 'forged')
            gt_path = os.path.join(gt_dir, name)
        elif 'COD' in src_dir:
            gt_path = os.path.join(gt_dir, name)
        else:
            continue

        save_path=os.path.join(save_dir,src_name)
        try:
            src = cv.imread(src_path, cv.IMREAD_COLOR)
            gt = cv.imread(gt_path, cv.IMREAD_COLOR)
            print(src.shape)
            print(gt.shape)
            pic_pic(src, gt, save_path)
        except Exception as e:
            print("错误", e)
            continue



if __name__ == "__main__":
    # src_dir = r"C:\Users\brighten\Desktop\DATA\public_dataset\casia\src"
    # gt_dir = r"C:\Users\brighten\Desktop\DATA\public_dataset\casia\gt"
    # save_dir = r"C:\Users\brighten\Desktop\save_biaozhu\casia"

    # src_dir=r"C:\Users\brighten\Desktop\DATA\public_dataset\coverage\src"
    # gt_dir = r"C:\Users\brighten\Desktop\DATA\public_dataset\coverage\gt"
    # save_dir = r"C:\Users\brighten\Desktop\save_biaozhu\coverage"

    # src_dir=r"C:\Users\brighten\Desktop\DATA\public_dataset\Columbia\src"
    # gt_dir = r"C:\Users\brighten\Desktop\DATA\public_dataset\Columbia\gt"
    # save_dir = r"C:\Users\brighten\Desktop\save_biaozhu\Columbia"

    src_dir=r"C:\Users\brighten\Desktop\DATA\COD10K_new_tampered\save_png"
    gt_dir=r"C:\Users\brighten\Desktop\DATA\COD10K_new_tampered\double_gt"
    save_dir= r"C:\Users\brighten\Desktop\save_biaozhu\cod10k_new"

    read_dir(src_dir, gt_dir, save_dir)