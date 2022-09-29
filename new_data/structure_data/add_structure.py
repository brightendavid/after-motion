#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
使用双边缘和纹理图像进行制作
带有透明度的结构图
要求图像的大小要大于240


基本的思想是  随机选择纹理或者  颜色相近的纹理进行融合，由cod10k类的人工半透明算法，设置alpha 数值理论上小于122，半透明边缘在于我方
同时，对于gt 进行同样的操纵，可以叠加，现在的gt_mask 是带有填充的双边缘

此时，gt 和src的大小相同，直接进行像素的覆盖

对于纹理分割，需要的gt 为彩色图，需要进行进一步发操作，生成彩色的gt 图像

数据分析：

0 0 0 0 0 0 0 0 0 0 0 0 100 255 255 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 255 255 255 255 100 100 100 0 0 0 0 0 0 0 0 0 0 0 0 100 100 255 255 255 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 255 255 255 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 100 255 255 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 255 255 100 100 100 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 100 255 255 50 50 50 50 50 50 50 50 50 255 100 0 0 0 0 0 0
0 0 0 100 100 255 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 255 255 100 100 100 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 100 100 255 255 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 255 255 100 100 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 100 255 255 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 255 255 100 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 100 255 255 50 50 50 50 50 50 50 255 255 100 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 100 255 255 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 255 255 100 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 100 255 255 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 255 255 255 100 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 255 255 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 255 255 100 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 100 255 255 255 255 255 255 255 255 255 100 100 0 0 0 0 0 0 0 0
"""

import random
import os
import shutil

import numpy as np
import cv2 as cv
from PIL import Image
import traceback
import skimage.morphology as dilation

from new_data.plan1.mask_splicing import MatteMatting

alpha255 = 210 / 255  # 设定透明度=90
alpha100 = 40 / 255


class Touming:
    def __init__(self, img_dir, mask_dir, img_save_dir, gt_save_dir,gt_wenli_dir,gt_wenli_fanshi):

        # 检查输出文件夹
        if os.path.exists(img_save_dir):
            print('输出文件夹已经存在，请手动更换输出文件夹')
        else:
            os.mkdir(img_save_dir)

        if os.path.exists(img_save_dir):
            print('输出文件夹已经存在，请手动更换输出文件夹')
        else:
            os.mkdir(gt_save_dir)
            print('输出文件夹创建成功')

        if os.path.exists(gt_wenli_dir):
            print('输出文件夹已经存在，请手动更换输出文件夹')
        else:
            os.mkdir(gt_wenli_dir)
            print('输出文件夹创建成功')


        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_save_dir = img_save_dir
        self.gt_save_dir = gt_save_dir
        self.gt_wenli_dir=gt_wenli_dir
        self.gt_wenli_fanshi=gt_wenli_fanshi

        self.src_list = []
        self.gt_list = []
        for index, item in enumerate(os.listdir(self.img_dir)):
            # src_path = os.path.join(self.img_dir, item)
            self.src_list.append(item)

        for index, item in enumerate(os.listdir(self.mask_dir)):
            # gt_path = os.path.join(self.mask_dir, item)
            self.gt_list.append(item)



        # print(len(self.gt_list))
        # print(len(self.src_list))

    def gen_origin_gt(self):
        """
        输入图像名称，创建全黑的gt图,作为之后填充gt（双边缘的gt）(直接填充)
        448*320大小   名称为name

        有纹理图像获取  纯黑色的结构gt和代表src均值的纹理滤波均值
        """
        for index, item in enumerate(os.listdir(self.img_dir)):
            name = item
            gt_origin = np.zeros((320, 448))
            name = os.path.join(self.gt_save_dir, name)
            print(name)
            cv.imwrite(name, gt_origin)

            name = os.path.join(self.gt_wenli_dir, item)
            src_path =os.path.join(self.img_dir,item)
            src=cv.imread(src_path)
            gt_wenli=np.ones((320,448,3))
            for i in range(3):
                gt_wenli[:,:,i]=np.mean(src[:, :, i])
            cv.imwrite(name, gt_wenli)

    def dou_gt_full(self, gt_origin_name, gt_add_name,src_name2):
        """
        src  旧有的
        gt 新的
        原图，修改图
        """
        name = os.path.join(self.gt_save_dir, gt_origin_name)
        gt = cv.imread(name, 0)
        gt_add_path = os.path.join(self.mask_dir, gt_add_name)
        gt_add = cv.imread(gt_add_path, 0)
        gt_after = np.where(gt_add != 0, gt_add, gt)
        print(name)
        cv.imwrite(name, gt_after)

        # 获取纹理滤波gt
        name = os.path.join(self.gt_wenli_dir, gt_origin_name)
        gt_wenli = cv.imread(name)
        name2=os.path.join(self.gt_wenli_fanshi,src_name2)
        gt_wenli2=cv.imread(name2)
        gt_add_path = os.path.join(self.mask_dir, gt_add_name)
        gt_add = cv.imread(gt_add_path,1)
        gt_after = np.where(gt_add  == 50, gt_wenli2, gt_wenli) # 82 255      135   0
        gt_after = np.where(gt_add == 255, gt_wenli2, gt_after)
        print(name)
        cv.imwrite(name, gt_after)


    def sort_wenli(self):
        """
        找颜色相近的俩纹理图（可能加入）
        return src1,src2
        """
        pass

    def sort_gt_src(self):
        """
        找一张gt 一张 src 纹理
        """
        for i in range(2):  # 加入循环，使得批处理
            # src_num1 = (random.randint(0, len(self.src_list) - 1))
            for src_num1 in range(len(self.src_list)):
                src_num2 = (random.randint(0, len(self.src_list) - 1))
                if src_num1 == src_num2:
                    src_num2 = (random.randint(0, len(self.src_list) - 1))
                gt_num = (random.randint(0, len(self.gt_list) - 1))

                print(src_num1, src_num2, gt_num)

                src_name1 = self.src_list[src_num1]
                src_name2 = self.src_list[src_num2]

                gt_name = self.gt_list[gt_num]

                print(src_name1)
                print(src_name2)
                print(gt_name)

                self.gen_result(src_name1, src_name2, gt_name)

    def gen_result(self, src_name1, src_name2, gt_name):
        """
        把src2 截取gt形状   贴到   src1    设置
        50部分是截取部分    255,100部分是融合部分
        """
        src_path1 = os.path.join(self.img_save_dir, src_name1)
        src_path2 = os.path.join(self.img_dir, src_name2)
        src1 = cv.imread(src_path1)
        src2 = cv.imread(src_path2)

        gt_path = os.path.join(self.mask_dir, gt_name)
        gt = cv.imread(gt_path)

        src1 = np.where(gt == 50, src2, src1)  # gt为50部分，填充为src2；其他部分   还是src1
        src1 = np.where(gt == 255, src2 * alpha255 + src1 * (1 - alpha255), src1)  # 透明度alpha1
        src1 = np.where(gt == 100, src2 * alpha100 + src1 * (1 - alpha100), src1)  # 透明度 alpha2

        src_save_name = os.path.join(img_save_dir, src_name1)
        cv.imwrite(src_save_name, src1)
        self.dou_gt_full(src_name1, gt_name,src_name2)


def copy_files(rootdir, des_path):
    # 从 root_dir 移动到 des_path 中，移动的文件类型不限制，按照字符顺序先后  移动文件数量为num个
    for item in os.listdir(rootdir):  # 遍历该文件夹中的所有文件
        dirname = os.path.join(rootdir, item)  # 将根目录与文件夹名连接起来，获取文件目录
        shutil.copy(dirname, des_path)  # 移动文件到目标路径

def clear(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)

if __name__ == '__main__':  # 由一般的gt和image生成新的gt和image
    # src  gt  save 文件夹

    img_dir = r'C:\Users\brighten\Desktop\gen_new_data\texture_cat'
    mask_dir = r'C:\Users\brighten\Desktop\gen_new_data\full'
    img_save_dir = r'C:\Users\brighten\Desktop\gen_new_data\result'
    gt_save_dir = r'C:\Users\brighten\Desktop\gen_new_data\result_gt'
    gt_wenli_dir=r'C:\Users\brighten\Desktop\gen_new_data\result_wenli'
    gt_wenli_fanshi=r'C:\Users\brighten\Desktop\gen_new_data\wenli_fanshi'

    clear(img_save_dir)
    clear(gt_save_dir)

    copy_files(img_dir,img_save_dir)
    copy_files(gt_wenli_fanshi, gt_wenli_dir)
    T = Touming(img_dir, mask_dir, img_save_dir, gt_save_dir,gt_wenli_dir,gt_wenli_fanshi)
    T.gen_origin_gt()
    T.sort_gt_src()
