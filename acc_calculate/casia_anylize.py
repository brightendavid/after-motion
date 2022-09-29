#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
测试区域指标
或者
边缘指标
在 Analyze(is_block=True) 时候，测试区域指标
在Analyze(is_block=False) 时候，测试边缘指标
默认gt 数据集为双边缘数据集 ，默认 is_block =False
"""
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


class Analyze:
    def __init__(self,is_block=False):
        """
        pred_dir
        """
        # 此处是预测图  原图  gt 的存储文件夹 C:\Users\brighten\Desktop\元数据集\Columbia Uncompressed Image Splicing Detection
        # Evaluation Dataset\4cam_splc\4cam_splc
        # coverage 数据集
        # self.pred_dir = r'C:\Users\brighten\Desktop\adaptive_casia_true'
        # self.src_dir = r'C:\Users\brighten\Desktop\public_dataset\casia\src'
        # self.gt_dir = r'C:\Users\brighten\Desktop\DATA\public_dataset\casia\gt'
        # self.save_excel_dir = r'C:\Users\brighten\Desktop\save\adaptive.xlsx'
        self.pred_dir = r'C:\Users\brighten\Desktop\full\casia\3'
        self.src_dir = r'F:\DATA\public_dataset\casia\src'
        self.gt_dir = r'F:\DATA\public_dataset\casia\gt'
        self.save_excel_dir = r'C:\Users\brighten\Desktop\casia.xlsx'
        self.is_block=is_block

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
        print('The number of images are: ', fileNumber)
        rate = 1
        pickNumber = int(fileNumber * rate)
        sample = random.sample(pred_list, pickNumber)

        f1_score_list = []
        pred_name_list = []
        loss_list = []
        src_name_list = []
        precision_score_list = []
        acc_list = []
        recall_list = []
        accsort_list = []

        combineArray = np.zeros((320, 4 * 320, 3))

        for index, name in enumerate(sample):
            print(index, '/', len(sample))
            src_path, gt_path = Analyze.coverage_find_src_and_gt(self, name)
            pred_img = os.path.join(path_dir, name)
            pred_img = Image.open(pred_img)
            gt_img = Image.open(gt_path)
            if pred_img.split() == 3:  # 若为3rgb图像 则会读取为灰度
                pred_img = pred_img.split()[0]
            pred_img = np.array(pred_img)
            try:
                gt_img = np.array(gt_img)  # image类 转 numpy
                gt_img = gt_img[:, :, 0]  # 第1通道
            except IndexError:
                gt_img = np.array(gt_img)
            else:
                pass

            print("shape")
            print(gt_img.shape)
            print(pred_img.shape)
            if gt_img.shape[0] == pred_img.shape[1] and gt_img.shape[1] == pred_img.shape[0] and gt_img.shape[1] != \
                    gt_img.shape[0]:
                gt_img = gt_img.T

            if len(pred_img.shape) ==3:
                pred_img=pred_img[:,:,0]


            print(gt_img.shape)
            print(pred_img.shape)
            if self.is_block:
                gt_img = np.where(gt_img >25,255,0)
            elif self.is_block==False:
                gt_img=np.where(gt_img>99,255,0)
            pred_img, gt = change(pred_img, gt_img)

            pred_img_tensor = pred_img

            print(gt.shape)


            try:
                loss_tonsor = wce_dice_huber_loss(pred_img_tensor.float(), gt.float())
                loss = loss_tonsor.item()
            except ValueError:
                print("尺寸错误")
                continue

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


            acc_sort=acc_score+f1_score+recall+precision
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


        test = pd.DataFrame(data)

        # 按照 acc_sort 降序排序
        test = test.sort_values(by="acc_sort",ascending=False)

        test.to_excel(self.save_excel_dir)

    def coverage_find_src_and_gt(self, name):
        """
        using pred name to find src and gt
        :return:src_path, gt_path
        输出的名字：output_Sp_Default_34_445004_zebra -->
        输入名字，找出文件位置
        """
        '''
        ###src:canong3_canonxt_sub_01.tif  ###gt:canong3_canonxt_sub_01_edgemask_3.jpg   ###pred:output_canong3_canonxt_sub_01.tif
        '''

        pred_name = name
        pred_name = pred_name[:-4]
        pred_name += ".tif"
        src_name = pred_name.replace('output2_', '')
        gt_name = pred_name.replace('output2_', '').replace('Default', 'Gt').replace('jpg', 'bmp').replace('png',
                                                                                                           'bmp').replace(
            'output_poisson', 'Gt')
        gt_name = gt_name[:-4]
        gt_name += "_gt.png"
        print(src_name)
        print(gt_name)
        src_path = os.path.join(self.src_dir, src_name)
        gt_path = os.path.join(self.gt_dir, gt_name)
        print(src_path)
        print(gt_path)

        if os.path.exists(gt_path):
            pass
        else:
            print(gt_path or src_path, ':not exists')
            traceback.print_exc()
            sys.exit()
        return src_path, gt_path


def np_to_tensor(src):
    pred_ndarray = np.array(src)
    pred_ndarray = pred_ndarray / 255

    pred_ndarray4D = pred_ndarray[np.newaxis, np.newaxis, :, :]
    # convert numpy to tensor
    pred_img_tensor = torch.from_numpy(pred_ndarray4D)
    pred_img_tensor = pred_img_tensor.contiguous()
    return pred_img_tensor


def change(pred, inputgt):  # pred为灰度   inputgt为rgb
    # print("pred")
    # print(pred.size)
    src = np_to_tensor(pred)
    mask = torch.from_numpy(inputgt)
    mask = np_to_tensor(mask)
    # print("src")
    # print(src.shape)
    return src, mask

# 写独立的函数会导致问题，注释掉
# def test_one_picture():  # 测试代码  转置测试
#     gt = cv.imread("../pictures/Tp_S_CRN_S_N_art00059_art00059_10508_gt.png", cv.THRESH_BINARY)
#     src = cv.imread("../pictures/output2_Tp_S_CRN_S_N_art00059_art00059_10508.tif", cv.THRESH_BINARY)
#     gt_img = gt
#     pred_img = src
#     if gt_img.shape[0] == pred_img.shape[1] and gt_img.shape[1] == pred_img.shape[0] and gt_img.shape[1] != \
#             gt_img.shape[0]:
#         gt = gt_img.T
#
#     src = np_to_tensor(src)
#     mask = torch.from_numpy(gt)
#     mask = np_to_tensor(mask)
#     print(src.shape)
#     print(mask.shape)
#     print(my_f1_score(src, mask))
#     loss_tonsor = wce_dice_huber_loss(src.float(), mask.float())
#     loss = loss_tonsor.item()
#     print("loss")
#     print(loss)


if __name__ == '__main__':
    analyze = Analyze(is_block=True)
    analyze.analyze_all()