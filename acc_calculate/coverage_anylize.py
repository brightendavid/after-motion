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


class Analyze:
    def __init__(self, is_block=False):
        """
        pred_dir
        """
        # 此处是预测图  原图  gt 的存储文件夹 C:\Users\brighten\Desktop\元数据集\Columbia Uncompressed Image Splicing Detection
        # Evaluation Dataset\4cam_splc\4cam_splc
        # coverage 数据集
        self.pred_dir = r'C:\Users\brighten\Desktop\full\coverage\3'
        # self.src_dir = r'C:\Users\brighten\Desktop\元数据集\coverage\COVERAGE\image'
        self.gt_dir = r'F:\DATA\public_dataset\coverage\gt'
        self.save_excel_dir = r'C:\Users\brighten\Desktop\coverage.xlsx'
        self.is_block = is_block

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
            # print(index, '/', len(sample))
            src_path, gt_path = Analyze.coverage_find_src_and_gt(self, name)
            pred_img = os.path.join(path_dir, name)
            pred_img = Image.open(pred_img)
            # src_img = Image.open(src_path)
            gt_img = Image.open(gt_path)
            if gt_img.split() == 3:  # 若为3rgb图像 则会读取为灰度
                gt_img = gt_img.split()[0]

            if pred_img.split() == 3:  # 若为3rgb图像 则会读取为灰度
                pred_img = pred_img.split()[0]

            pred_img = np.array(pred_img)
            gt_img = np.asarray(gt_img)

            if len(pred_img.shape) == 3:
                pred_img = pred_img[:, :, 0]

            if self.is_block: # 区域指标
                gt_img = np.where(gt_img > 25, 255, 0)
            else:  # 双边缘
                gt_img = np.where(gt_img > 99, 255, 0)
            # gt_img = gt_img[:, :]
            pred_img, gt = change(pred_img, gt_img)

            pred_img_tensor = pred_img

            print(pred_img.shape)
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
        pred_name += ".bmp"
        src_name = pred_name.replace('output2_', '')
        gt_name = pred_name.replace('output2_', '').replace('Default', 'Gt').replace('jpg', 'bmp').replace('png',
                                                                                                           'bmp').replace(
            'output_poisson', 'Gt')
        gt_name = gt_name[:-5]
        gt_name += "forged.bmp"
        print(src_name)
        print(gt_name)
        src_path = ""  # os.path.join(self.src_dir, src_name)
        gt_path = os.path.join(self.gt_dir, gt_name)
        print(src_path)
        print(gt_path)

        if os.path.exists(gt_path):
            pass
        else:
            print(gt_path or src_path, ' not exists')
            traceback.print_exc()
            sys.exit()
        return src_path, gt_path


def np_to_tensor(src):
    pred_ndarray = np.array(src)
    pred_ndarray = pred_ndarray / 255

    pred_ndarray4D = pred_ndarray[np.newaxis, np.newaxis, :, :]
    # convert numpy to tensor
    pred_img_tensor = torch.from_numpy(pred_ndarray4D)
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


if __name__ == '__main__':
    analyze = Analyze(is_block=True)
    analyze.analyze_all()
