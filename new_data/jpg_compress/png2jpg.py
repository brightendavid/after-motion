#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import shutil
import os
import numpy as np
from PIL import Image
import cv2 as cv
# 数据集本身带有jpg和png的配比
# 使用原本的test数据集  是没有用过的 ;使用原本的train 数据集 用处不大
# 压缩比的问题
# 其实这个体积的变化有1/3的原因是消去了透明度通道
# quality=70 : 1505.28kb->164kb
# quality=50 : 1505.28kb->124kb
import random


# 先做70的压缩比例 3000   有些图片已经完全不行了

# casia_au_and_casia_jpgcompress  quality = 70  2087张
# coco_cm_jpgcompress quality =60 3000张

def to_jpg(src_dir, save_dir, num):
    # quality = random.randint(50, 70)
    quality = 30
    Path = src_dir
    img_dir = os.listdir(Path)
    img_dir.reverse()
    count = 0
    for img in img_dir:
        # if img.endswith('.png'): # 不单单是将png转为jpg，而是二次的压缩
        PngPath = os.path.join(Path, img)
        # print("path")
        print(PngPath)
        for quality in range(30, 71, 10):
            PNG_JPG(PngPath, save_dir, quality)
        count += 1
        if count == num:
            break


def PNG_JPG(PngPath, save_dir, quality):
    infile = PngPath
    outfile = PngPath.split("\\")[-1][:-4]
    outfile = os.path.join(save_dir, outfile)
    outfile = outfile + "_" + str(quality) + ".jpg"
    # print("out")
    print(outfile)
    img = Image.open(infile)
    # img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=quality)
        else:
            img.convert('RGB').save(outfile, quality=quality)
    except Exception as e:
        print("PNG转换JPG 错误", e)


def find_gt(src_dir, gt_dir1, gt_dir2):
    img_list = os.listdir(src_dir)
    for img in img_list:
        if "compress" not in img:
            name = img.replace("Default", "Gt")
            name = name[:-7]
            print(name)
            name = name + ".bmp"
            print(name)
            gt_path = os.path.join(gt_dir1, name)
            shutil.copy(gt_path, gt_dir2)


def rename(src_path):
    # Tp_D_CND_M_N_ani00018_sec00096_00138_gt_000000126347_000000570741_Au_pla_30397_Au_txt_00073.bmp
    img_list = os.listdir(src_path)
    for img in img_list:
        if "compress" not in img:
            path = os.path.join(src_path, img)
            name = img[:-4] + "_compress" + img[-4:]
            print(name)
            src = Image.open(path)
            save_path = os.path.join(src_path, name)
            src.save(save_path)


def delete_pic(src_path):
    img_list = os.listdir(src_path)
    for img in img_list:
        if "compress" not in img:
            path = os.path.join(src_path, img)
            os.remove(path)


if __name__ == "__main__":
    # png_dir = r"C:\Users\brighten\Desktop\test\tesss"
    # jpg_dir = r"C:\Users\brighten\Desktop\test\new"

    # casia_au_and_casia_template_after_divide 数据集
    png_dir = r"G:\3月最新数据\casia_au_and_casia_template_after_divide\test_src"
    jpg_dir = r"C:\Users\brighten\Desktop\jpg_compress\casia_au_and_casia_jpgcompress\src"
    gt_dir1 = r"G:\3月最新数据\casia_au_and_casia_template_after_divide\test_gt"
    gt_dir2 = r"C:\Users\brighten\Desktop\jpg_compress\casia_au_and_casia_jpgcompress\gt"

    # coco_cm 数据集
    # png_dir=r"G:\3月最新数据\coco_cm\test_src"
    # jpg_dir=r"C:\Users\brighten\Desktop\jpg_compress\coco_cm_jpgcompress\src"
    # gt_dir1=r"G:\3月最新数据\coco_cm\test_gt"
    # gt_dir2 = r"C:\Users\brighten\Desktop\jpg_compress\coco_cm_jpgcompress\gt"
    num = 1000
    to_jpg(png_dir, jpg_dir, num)
    find_gt(jpg_dir, gt_dir1, gt_dir2)
    rename(jpg_dir)
    delete_pic(jpg_dir)
    # rename(gt_dir2)
