#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import os
import cv2


# 划线法 考虑弃用
def fulling(img):
    img1 = img.copy()
    img2 = img.copy()

    # 横向
    flag = False
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 255:
                flag = not flag
                while img[i][j] == 255:
                    j += 1
                    if j >= img.shape[1]:
                        break
            if j >= img.shape[1]:
                break
            if flag == True:
                img1[i][j] = 255
        flag = False
    cv2.imshow("1", img1)
    # 纵向
    flag = False
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if img[j][i] == 255:
                flag = not flag
                while img[j][i] == 255:
                    j += 1
                    if j >= img.shape[0]:
                        break
            if j >= img.shape[0]:
                break
            if flag == True:
                img2[j][i] = 255
        flag = False
    cv2.imshow("2", img2)
    img = img1 & img2
    # cv2.imshow("3", img)
    cv2.waitKey(0)
    return img


# [ 0000000010000000001000000 ]
def one_picture(src):
    # breakpoint_connect_path = r'./pictures/output_canong3_canonxt_sub_05.tif'
    # save_path=r''
    # breakpoint_connect_mask = Image.open(breakpoint_connect_path)
    if src.split() == 3:
        src = src.split()[0]
    img = np.asarray(src)
    print(img)
    img = fulling(img)
    cv2.imwrite("pictures/pic.png", img)


def all_picture_full(input, output):
    src_path = input
    save_path = output
    for index, item in enumerate(os.listdir(src_path)):
        _src_path = os.path.join(src_path, item)
        _save_path = os.path.join(save_path, item)
        pred_mask = Image.open(_src_path)
        if pred_mask.split() == 3:  # 若为3rgb图像 则会读取为灰度
            pred_mask = pred_mask.split()[0]
        pred_mask = np.array(pred_mask)

        _mask_after_full = fulling(pred_mask)  # 关键函数

        _mask_after_full = Image.fromarray(_mask_after_full.astype(np.uint8))
        _mask_after_full.save(_save_path)
        print("{}/{}".format(index + 1, len(os.listdir(src_path))))
        print(item)


if __name__ == '__main__':
    # breakpoint_full_path = r'./pictures/output_canong3_canonxt_sub_05.tif'
    # breakpoint_full_mask = Image.open(breakpoint_full_path)
    # one_picture(breakpoint_full_mask)  # 单张图片测试
    src_path = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result2'
    save_path = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result3'
    all_picture_full(src_path, save_path)  # 文件夹内所有图片
