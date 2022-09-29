#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import random
import sys
import traceback

import PIL.Image as Image
import cv2 as cv
import numpy as np
import pandas as pd
import torch

from acc_calculate.acc_functions import my_precision_score, my_acc_score, my_recall_score, my_f1_score, \
    wce_dice_huber_loss

"""
created by haoran
time: 11/8
usage:
1. using band_pred to gen a combine image which shows that src gt band_gt band_pred
2. analyze l1 f1 precision acc and recall to gen a excel file
"""


class Analyze:
    def __init__(self):
        """
        pred_dir
        """
        # 此处是预测图  原图  gt 的存储文件夹 C:\Users\brighten\Desktop\元数据集\Columbia Uncompressed Image Splicing Detection
        # Evaluation Dataset\4cam_splc\4cam_splc
        # calumbia数据集
        self.pred_dir = r'C:\Users\brighten\Desktop\full\columbia\3'
        self.src_dir = r'F:\DATA\public_dataset\Columbia\src'
        self.gt_dir = r'F:\DATA\public_dataset\Columbia\gt'
        self.save_excel_dir = r'C:\Users\brighten\Desktop\calumia.xlsx'
        # self.pred_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\coverage_test\pred_train\result3'
        # self.src_dir = r'C:\Users\brighten\Desktop\元数据集\coverage\COVERAGE\image'
        # self.gt_dir = r'C:\Users\brighten\Desktop\元数据集\coverage\COVERAGE\mask'
        # self.save_excel_dir = r'C:\Users\brighten\Desktop\save\coverage.xlsx'

    def analyze_all(self):
        """

        :return:
        """
        path_dir = self.pred_dir
        # 1. path check
        if os.path.exists(path_dir):
            print('path ok')
        else:
            print('Path Not find:', path_dir)
            sys.exit()

        pred_list = os.listdir(path_dir)
        fileNumber = len(pred_list)
        # print('The number of images are: ', fileNumber)
        rate = 1
        pickNumber = int(fileNumber * rate)
        sample = random.sample(pred_list, pickNumber)
        print("sample",sample)
        f1_score_list = []
        pred_name_list = []
        loss_list = []
        src_name_list = []
        # gt_name_list = []
        # band_gt_name_list = []
        precision_score_list = []
        acc_list = []
        recall_list = []
        accsort_list = []

        # combineArray = np.zeros((320, 4 * 320, 3))

        for index, name in enumerate(sample):
            print(index, '/', len(sample))
            src_path, gt_path = Analyze.calumia_find_src_and_gt(self, name)
            print("src_path, gt_path",src_path, gt_path)
            pred_img = os.path.join(path_dir, name)
            pred_img = Image.open(pred_img)
            # src_img = Image.open(src_path)
            gt_img = Image.open(gt_path)

            if gt_img.split() == 3:  # 若为3rgb图像 则会读取为灰度
                gt_img = gt_img.split()[0]

            if pred_img.split() == 3:  # 若为3rgb图像 则会读取为灰度
                pred_img = pred_img.split()[0]

            pred_img, gt = change(pred_img, gt_img)

            pred_img_tensor = pred_img

            print("pred__")
            print(pred_img.shape)
            print("gt")
            print(gt.shape)

            # loss_tonsor = wce_dice_huber_loss(pred_img_tensor.float(), gt.float())
            #
            # loss = loss_tonsor.item()
            loss=0

            f1_score = my_f1_score(pred_img_tensor, gt)

            acc_score = my_acc_score(pred_img_tensor, gt)
            recall = my_recall_score(pred_img_tensor, gt)
            precision = my_precision_score(pred_img_tensor, gt)

            # output to csv
            f1_score_list.append(f1_score)
            pred_name_list.append(name)
            loss_list.append(loss)
            acc_list.append(acc_score)
            recall_list.append(recall)
            precision_score_list.append(precision)

            acc_sort = acc_score + f1_score + recall + precision
            accsort_list.append(acc_sort)

            src_name = src_path.split('/')[-1]
            src_name_list.append(src_name)

        data = {
            'srcName': src_name_list,
            'predName': pred_name_list,
            'loss': loss_list,
            'precision': precision_score_list,
            'recall': recall_list,
            'f1': f1_score_list,
            'acc': acc_list,
            'acc_sort': accsort_list
        }
        # print(data)
        test = pd.DataFrame(data)

        # 按照 acc_sort 降序排序
        test = test.sort_values(by="acc_sort", ascending=False)

        test.to_excel(self.save_excel_dir)

    def calumia_find_src_and_gt(self, name):
        """
        using pred name to find src and gt
        :return:src_path, gt_path
        输出的名字：output_Sp_Default_34_445004_zebra -->
        输入名字，找出文件位置
        """
        '''
        ###src:canong3_canonxt_sub_01.tif  ###gt:canong3_canonxt_sub_01_edgemask_3.jpg   ###pred:output_canong3_canonxt_sub_01.tif
        '''
        print("name",name)
        pred_name = name
        src_name = pred_name.replace('output_', '')
        gt_name = src_name[:-4]
        gt_name += "_edgemask.bmp"
        # # print(src_name)
        # # print(gt_name)
        src_path = os.path.join(self.src_dir, src_name)
        gt_path = os.path.join(self.gt_dir, gt_name)
        print(src_path)
        print(gt_path)
        #
        # if os.path.exists(gt_path and src_path):
        #     pass
        # else:
        #     print(gt_path or src_path, ' not exists')
        #     traceback.print_exc()
        #     sys.exit()
        return src_path, gt_path


def columbia_gt(src):
    # 读入3通道的彩图  有正常的红绿蓝三色组成  物理意义由上文字所示
    """
            单边缘记作蓝色(0,0,200)
            双边缘记作 (255,0,0)(亮红色):非篡改边缘  (0,255,0)(亮绿色):篡改边缘
            篡改区域:绿色（0，200，0）
            非篡改区域:   红色（200，0，0）
    """
    # b, g, r = cv.split(src)
    mask = np.where(src >= 25, 255, 0)  # 因为是jpg，所以会有像素损失
    mask = np.array(mask)
    return mask


def np_to_tensor(src):
    pred_ndarray = np.array(src)
    if len(pred_ndarray.shape) == 3:
        pred_ndarray = pred_ndarray[:, :, 0]

    pred_ndarray = pred_ndarray / 255

    pred_ndarray4D = pred_ndarray[np.newaxis, np.newaxis, :, :]
    # convert numpy to tensor
    pred_img_tensor = torch.from_numpy(pred_ndarray4D)
    return pred_img_tensor


# def test_one_picture():  # 测试代码
#     gt = cv.imread("../pictures/canong3_canonxt_sub_01_edgemask.jpg")
#     src = cv.imread("../pictures/output_canong3_canonxt_sub_01.tif", cv.THRESH_BINARY)
#
#     mask = columbia_gt(gt)
#     src = np_to_tensor(src)
#     mask = torch.from_numpy(mask)
#     mask = np_to_tensor(mask)
#     print(my_f1_score(src, mask))


def change(pred, inputgt):  # pred为灰度   inputgt为rgb
    # print("pred")
    # print(pred.size)
    inputgt = cv.cvtColor(np.asarray(inputgt), cv.COLOR_RGB2BGR)
    mask = columbia_gt(inputgt)

    src = np_to_tensor(pred)
    mask = torch.from_numpy(mask)
    mask = np_to_tensor(mask)
    # print("src")
    # print(src.shape)
    return src, mask


if __name__ == '__main__':
    analyze = Analyze()
    analyze.analyze_all()
