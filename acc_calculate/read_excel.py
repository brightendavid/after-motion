#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# time:2021/7/7
# 从excel 找出 src
# 对应 excel的生成代码在calumia_anylize.py,coverage_anylize.py,casia_anylize.py 中
# 对应不同的数据集  命名格式为name_anyllize.py
# 本脚本功能：从数据集中找出本算法中表现不错的图片，与其他算法比较
#
#
# excel:示范 srcName :  C:\Users\brighten\Desktop\元数据集\Columbia Uncompressed Image Splicing Detection Evaluation
# Dataset\4cam_splc\4cam_splc\canong3_canonxt_sub_26.tif predName : output_canong3_canonxt_sub_26.tif
# loss : 0.145315155 precision: 0.99676608 recall : 0.989267398
# f1 :  0.993002582 acc :  0.996181182 acc_sort : 3.975217242

import os
import sys
import cv2 as cv
import PIL.Image as Image
import pandas as pd
import numpy as np


def find_picture_name(df, index):
    # 给出下标 输出src 文件名
    return df.loc[index].values[1]
    # 给出 pred图片文件名
    # return df.loc[index].values[2]


def Find_AccSort(df, index):
    x = df.loc[index].values[-1]
    x = float(x)
    return x


def read_src(df, savedir, score,flag=True):
    # 读取所有图片，保存到指定文件夹，此处读取的是数据集中的篡改图像src
    # src.show()

    for index in range(0, len(df) - 1):

        if Find_AccSort(df, index) >= score:
            # 精确度判定,要大于3
            path_dir = find_picture_name(df, index)
            # path_dir=read_dir+path_dir
            # path_dir=str(path_dir)
            # print(path_dir)
            if os.path.exists(path_dir):
                s = "\\"
                path_dir = path_dir.replace(s, r'//')
                # print(path_dir)
                x = path_dir.rfind("/")
                item = path_dir[x + 1:]
                path = path_dir
                print('path ok')
            else:
                print('寻找下一张图片!!!!Path Not find:', path_dir)
                continue
        else:
            print('!!!!之后的图片准确度不足要求，程序退出')
            sys.exit()

        # path, item = find_src(df, index)
        src = Image.open(path)
        print(src.size)
        # save_dir = os.path.join(savedir,item)
        save_dir = savedir + '\\' + item
        # print(src.size)
        print(save_dir)

        if flag:
            src = cv.cvtColor(np.asarray(src), cv.COLOR_RGB2BGR)  # coverage数据集  很奇怪必须要转换为cv2格式才能够保存，否则报错
            cv.imwrite(save_dir, src)
        # src.save(save_dir)
        # src.show()


if __name__ == "__main__":

    # columbia  160张
    df = pd.read_excel(r'C:\Users\brighten\Desktop\save\casia.xlsx')  # excel 路径
    save_dir = r"C:\Users\brighten\Desktop\act_well\result_casia"  # 图片保存路径

    # coverage  37张
    # df = pd.read_excel(r"C:\Users\brighten\Desktop\save\coverage.xlsx")
    # save_dir = r"C:\Users\brighten\Desktop\act_well\coverage"

    # casia  315张
    # df = pd.read_excel(r"C:\Users\brighten\Desktop\save\casia.xlsx")
    # save_dir = r"C:\Users\brighten\Desktop\act_well\casia"

    # 修改上方路径
    read_src(df, save_dir, score=2.3)
