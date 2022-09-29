#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 选择文件夹中带有png 后缀的文件
# 源文件夹：C:\Users\brighten\Desktop\gen_new_data\texture_11-28
# 选择后，重新存放在新文件夹内
#
#
# 结构文件夹：C:\Users\brighten\Desktop\gen_new_data\DUTS-TE-Mask
# 要求：重构 为双边缘，有填充的结构图
# 预处理部分

import os
import shutil
import random

import cv2
import numpy as np
from edge2area.automation_full import picture_fulling
from get_gt import *


def get_double_edge(mask):
    # 是原本用于生成双边缘的代码  输入输出为   3通道黑白mask  输出为   单通道灰度的mask 带有双边缘
    # 这个三通道就不合理
    # 输入的是黑白的mask图  Image 形式  3通道的
    if len(mask.shape) != 2:
        # print('the shape of mask is :', mask.shape)
        mask = mask[:, :, 0]
        print("通道数不为1")
    # mask = np.array(mask)[:, :]
    # cv2.imshow("2", mask)
    # cv2.waitKey(0)
    # 给的灰度阈值很宽松
    mask = np.where(mask > 100, 1, 0)

    # print('the shape of mask is :', mask.shape)
    selem = np.ones((3, 3))
    dst_8 = dilation.binary_dilation(mask, selem=selem)
    dst_8 = np.where(dst_8 == True, 1, 0)

    difference_8 = dst_8 - mask
    difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
    difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)

    double_edge_candidate = difference_8_dilation + mask
    double_edge = np.where(double_edge_candidate == 2, 1, 0)
    ground_truth = np.where(double_edge == 1, 205, 0) + np.where(difference_8 == 1, 100, 0) + np.where(mask == 1, 50, 0)
    # ground_truth = np.where(double_edge == 1, 205, 0) + np.where(mask == 1, 50, 0)
    # 所以内侧边缘就是100的灰度值
    return ground_truth


def double_edge(mask_dir):
    src_path = mask_dir
    save_path = mask_dir
    for index, item in enumerate(os.listdir(src_path)):
        _src_path = os.path.join(src_path, item)
        _save_path = os.path.join(save_path, item)
        pred_mask = Image.open(_src_path).convert("L")

        pred_mask = np.asarray(pred_mask)
        result = get_double_edge(pred_mask)

        # result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
        result = Image.fromarray(result.astype(np.uint8))
        result.save(_save_path)
        print("{}/{}".format(index + 1, len(os.listdir(src_path))))
        print(item)


def edge2douedge(edge_dir, save_dir):
    picture_fulling(edge_dir, save_dir)  # 边缘转区域
    # 区域转双边缘


def sort_png(structure_dir, save_dir):
    for index, item in enumerate(os.listdir(structure_dir)):
        file_path = os.path.join(structure_dir, item)
        save_path = os.path.join(save_dir, item)
        if "txt" not in item:
            shutil.copyfile(file_path, save_path)


def img_reshape(img, Weight_out, Height_out):
    """
    最紧邻插值法
    """
    # 获取图像的大小
    Height_in, Weight_in, c = img.shape
    # 创建输出图像
    outimg = np.zeros((Height_out, Weight_out, c), dtype=np.uint8)

    for x in range(Height_out - 1):
        for y in range(Weight_out - 1):
            # 计算输出图像坐标（i,j）坐标使用输入图像中的哪个坐标来填充
            x_out = round(x * (Height_in / Height_out))
            y_out = round(y * (Weight_in / Weight_out))
            # 插值
            outimg[x, y] = img[x_out, y_out]
    return outimg


def dir_reshape(img_dir, save_dir):
    """
    reshape 在填充后，转换为双边缘之前
    """
    for index, item in enumerate(os.listdir(img_dir)):
        print(item)
        src_path = os.path.join(img_dir, item)
        save_path = os.path.join(save_dir, item)
        src = cv2.imread(src_path)
        src_reshape = img_reshape(src, 448, 320)
        cv2.imwrite(save_path, src_reshape)

def pic_cat(img,a,b):
    """
    截取到a,b
    只要大小比a,b 要大 ，全部截取图像到a,b大小
    """
    if img.size[0] > a and img.size[1] > b:
        height_center = img.size[0] // 2
        width_center = img.size[1] // 2
        width_crop = a / 2
        height_crop=b/2

        if width_center > width_crop:
            range_w = random.randint(0, width_center - width_crop)
        else:
            range_w = 0
        if height_center > height_crop:
            range_h = random.randint(0, height_center - height_crop)
        else:
            range_h = 0

        if random.randint(0, 1) == 0:
            img = img.crop(
                (width_center - width_crop + range_w, height_center - height_crop + range_h,
                 width_center + width_crop + range_w,
                 height_center + height_crop + range_h))
        else:
            img = img.crop(
                (width_center - width_crop - range_w, height_center - height_crop - range_h,
                 width_center + width_crop - range_w,
                 height_center + height_crop - range_h))
        return img

def dir_cat(img_dir, save_dir):
    for index, item in enumerate(os.listdir(img_dir)):
        print(item)
        src_path = os.path.join(img_dir, item)
        save_path = os.path.join(save_dir, item)
        src = Image.open(src_path)
        src_reshape = pic_cat(src, 448, 320)
        src_reshape.save(save_path)

if __name__ == '__main__':
    # edge_dir = r"C:\Users\brighten\Desktop\gen_new_data\DUTS-TE-Mask"
    # save_dir = r"C:\Users\brighten\Desktop\gen_new_data\full"
    # edge2douedge(edge_dir, save_dir)
    # dir_reshape(save_dir,save_dir)
    # double_edge(save_dir)
    # sort_path = r"C:\Users\brighten\Desktop\gen_new_data\texture_11-28"
    # sorted_path = r"C:\Users\brighten\Desktop\gen_new_data\texture"
    # sort_png(sort_path, sorted_path)
    # img=Image.open(r".\1m.png").convert("L")
    # t=pic_cat(img,448,320)
    # print(t.size)
    src_dir=r"C:\Users\brighten\Desktop\gen_new_data\texture"
    save_dir=r"C:\Users\brighten\Desktop\gen_new_data\texture_cat"
    dir_cat(src_dir,save_dir=save_dir)

