#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
对于jpgcompress 数据集  删除 quality=30，40 中f1_stage1 小于0.1 的图像，说明预测不了，超越了算法极限

"""


import os
import sys
import cv2 as cv
import PIL.Image as Image
import pandas as pd
import numpy as np


def find_picture_name(df, index):
    try:
        if "\\" not in df.loc[index].values[1] and "/" not in df.loc[index].values[1]:
            # 发现存储的文件名称只有图像名称
            return os.path.join(src_dir,df.loc[index].values[1])
    except Exception as e:
        # 说明excel 的格式不对  名称在第一列
        if "\\" not in df.loc[index].values[0] and "/" not in df.loc[index].values[0]:
            # 发现存储的文件名称只有图像名称
            return os.path.join(src_dir,df.loc[index].values[0])
    else:
        sys.exit(0)
    # 给出下标 输出src 文件名
    # 存储的名称包含路径
    return df.loc[index].values[1]
    # 给出 pred图片文件名
    # return df.loc[index].values[2]


def Find_AccSort(df, index):
    x = df.loc[index].values[2] # 【2】的意义是stage1 的f1 score
    x = float(x)
    return x


def read_src(df, savedir, score, flag=False):
    # 读取excel 删除指标过低的图像
    # src.show()

    for index in range(0, len(df) - 1):

        if Find_AccSort(df, index) <= score:
            # f1 stage1 判定  要大于score
            path_dir = find_picture_name(df, index)
            # path_dir=read_dir+path_dir
            # path_dir=str(path_dir)
            # print(path_dir)
            if "jpgcompress" in path_dir:
                path_dir=path_dir.replace("bmp","jpg")
            if os.path.exists(path_dir):
                s = "\\"
                path_dir = path_dir.replace(s, r'//')
                # print(path_dir)
                x = path_dir.rfind("/")
                item = path_dir[x + 1:]
                path = path_dir
                print('path ok: '+path)
            else:
                print('寻找下一张图片!!Path Not find:', path_dir)
                continue
            if flag:
                os.remove(path_dir)




if __name__ == "__main__":
    # cod10k_new 数据集
    # df = pd.read_excel(r'C:\Users\brighten\Desktop\model\8_23\cod10k_new每个样本的指标信息.xlsx')  # excel 路径
    # save_dir = r"C:\Users\brighten\Desktop\act_well\result_casia"  # 图片保存路径
    # src_dir=r"C:\Users\brighten\Desktop\DATA\COD10K_new_tampered\save_png"  # 图像的路径

    # jpg_compress 数据集
    # 这个数据集中的指标过低图像可以直接删除 认定如果指标过低，则说明超越了算法的性能极限
    # C:\Users\brighten\Desktop\DATA\casia_au_and_casia_jpgcompress\src\Tp_D_CND_M_N_ani00018_sec00096_00138_gt_000000047406_000000062574_Au_sec_30687_Au_ind_00054_30_compress.jpg
    # C:\Users\brighten\Desktop\DATA\casia_au_and_casia_jpgcompress\src\Tp_D_CNN_S_N_sec00042_ani00070_10541_gt_000000133596_000000561938_Au_ani_30337_Au_nat_30104_30_compress.bmp
    df = pd.read_excel(r'C:\Users\brighten\Desktop\model\8_23\hard\jpg_compress50.xls')  # excel 路径
    save_dir = r"C:\Users\brighten\Desktop\act_well\result_casia"  # 图片保存路径
    src_dir=r"C:\Users\brighten\Desktop\DATA\casia_au_and_casia_jpgcompress\src"  # 图像的路径

    # 修改上方路径
    read_src(df, save_dir, score=0.1, flag=False)
