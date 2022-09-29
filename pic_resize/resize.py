#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image


# 每次使用新的数据集进行生成前必须使用
# 截取图片中的指定区域或在指定区域添加某一图片
def half_resize(src_dir):
    # src_path 下的所有图片全部resize 到一半的大小  且在400 以内  长宽的等比例缩放（在2整除的情况下）
    # src_path=r"C:\Users\brighten\Desktop\截图\real"
    for index, item in enumerate(os.listdir(src_dir)):
        open_path = os.path.join(src_dir, item)
        pic = Image.open(open_path)
        a, b = pic.size
        print("%d  %d " % ((a), (b)))
        while a > 400 or b > 400:
            a = a // 2
            b = b // 2
        pic = pic.resize((a, b))
        pic.save(open_path)


from tqdm import tqdm


def all_resize_to_320_320(src_dir):
    # 完全的破坏长宽之间的比例关系，直接reshape到320*320
    # 全部的图片重新保存到同目录下
    src_list = os.listdir(src_dir)
    for index, item in tqdm(enumerate(src_list)):
        open_path = os.path.join(src_dir, item)
        pic = Image.open(open_path)
        if pic.size !=(2048,1024):
            pic = pic.resize((2048, 1024))
            pic.save(open_path)

def resize_to_gtszie(src_dir,gt_dir,save_dir):
    # 只需要把src的shape 转化为gt的shape即可
    # casia  Tp_D_CND_M_N_ani00018_sec00096_00138.bmp - Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png
    # coverage 1t.bmp - 1forged.bmp
    src_list = os.listdir(src_dir)
    for index, item in tqdm(enumerate(src_list)):
        if "casia" in src_dir:
            gt_name=item[:-4]+"_gt.png"
        elif "coverage" in src_dir:
            gt_name=str(item).replace("t","forged")
        else:
            return
        open_path = os.path.join(src_dir, item)
        gt_path=os.path.join(gt_dir,gt_name)
        gt = Image.open(gt_path)
        save_path=os.path.join(save_dir,item)
        a,b=gt.size
        print(gt.size)
        pic = Image.open(open_path)
        pic = pic.resize((a, b))
        pic.save(save_path)
if __name__ == "__main__":
    # # 这两个是源文件  图片
    # src_dir = r"C:\Users\brighten\Desktop\Image"
    # gt_dir = r"C:\Users\brighten\Desktop\GT_Object"
    # src_dir = r"C:\Users\brighten\Desktop\COD10K-v3\Test\GT_Object"  # 导入的原
    # gt_dir = r"C:\Users\brighten\Desktop\COD10K-v3\Test\Image"
    # gt_dir= r"C:\Users\brighten\Desktop\ceshi\src"
    # src_dir= r"C:\Users\brighten\Desktop\ceshi\mask"
    # src_dir = r"C:\Users\brighten\Desktop\test_cod\Image"
    # gt_dir = r"C:\Users\brighten\Desktop\test_cod\GT_Object"
    gt_dir=r"F:\dataset_watermark\DIV2K_train_HR"
    all_resize_to_320_320(gt_dir)
    # all_resize_to_320_320(src_dir)
    # src_dir=r"C:\Users\brighten\Desktop\cat_casia"
    # gt_dir=r"C:\Users\brighten\Desktop\DATA\public_dataset\casia\gt"
    # save_dir=r"C:\Users\brighten\Desktop\cat_casia_reshape"
    # src_dir=r"C:\Users\brighten\Desktop\cat_coverage"
    # gt_dir=r"C:\Users\brighten\Desktop\DATA\public_dataset\coverage\gt"
    # save_dir=r"C:\Users\brighten\Desktop\cat_coverage_reshape"
    # resize_to_gtszie(src_dir, gt_dir, save_dir)