#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created by:sjw
Time:2021.7.22
找同名的图片放到同一个文件夹下面
"""
from PIL import Image
import os
import cv2 as cv
import numpy as np

def all_picture_save(src_path, gt_path,save_path):
    for index, item in enumerate(os.listdir(src_path)):
        # 14t.bmp
        name=item
        print(name)
        if "coverage" in gt_path:
            name=name.replace("t","forged")
        elif "casia" in gt_path:
            name=name[:-4]+"_gt.png"
        print(name)
        gt_name=os.path.join(gt_path,name)
        gt=cv.imread(gt_name)
        save_name=os.path.join(save_path,name)
        cv.imwrite(save_name,gt)


def gt_config(gt1,gt2,gt3):
    for index, item in enumerate(os.listdir(gt1)):
        name = item
        print(name)
        path=os.path.join(gt1,name)
        src=cv.imread(path,0)
        src2=np.where(src>99,255,0)
        src3=np.where(src==50,255,src)

        path2=os.path.join(gt2,name)
        path3=os.path.join(gt3,name)
        cv.imwrite(path2,src2)
        cv.imwrite(path3,src3)


if __name__ == '__main__':
    # 输入六元组  为图像篡改检测的所有六个要素  意为保存效果较好的一些图像和它们的配套图像  找图片的索引是  区域预测图像  因为这类图像  的数量最少

    # casia  读取目录
    # src_path = r'C:\Users\brighten\Desktop\public_dataset\casia\src'
    # gt_path = r"C:\Users\brighten\Desktop\public_dataset\casia\gt"  # 双边缘的gt
    # pred_stage1 = r"C:\Users\brighten\Desktop\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\casia_test\pred_train\stage1"
    # pred_stage2 = r"C:\Users\brighten\Desktop\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\casia_test\pred_train\stage2"
    # pred_area = r"C:\Users\brighten\Desktop\act_well\result_casia"  # 名称演示 output2_Tp_D_CNN_S_N_ani00087_ani00088_10102.tif
    # save_path = r'C:\Users\brighten\Desktop\show_result'

    # coverage  读取目录
    # src_path = r'C:\Users\brighten\Desktop\public_dataset\coverage\src'
    # gt_path = r"C:\Users\brighten\Desktop\public_dataset\coverage\gt"  # 双边缘的gt
    # pred_stage1 = r"C:\Users\brighten\Desktop\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\coverage_test\pred_train\stage1"
    # pred_stage2 = r"C:\Users\brighten\Desktop\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\coverage_test\pred_train\stage2"
    # pred_area = r"C:\Users\brighten\Desktop\act_well\result_coverage"  # 名称演示 output2_Tp_D_CNN_S_N_ani00087_ani00088_10102.tif
    # save_path = r'C:\Users\brighten\Desktop\show_result'

    # columbia 读取
    # src_path = r'C:\Users\brighten\Desktop\public_dataset\Columbia\src'
    # gt_path = r"C:\Users\brighten\Desktop\public_dataset\Columbia\gt"  # 双边缘的gt
    # pred_stage1 = r"C:\Users\brighten\Desktop\forgery-edge-detection-main\Mymodel\chr_std\Columbia\stage1"
    # pred_stage2 = r"C:\Users\brighten\Desktop\forgery-edge-detection-main\Mymodel\chr_std\Columbia\stage2"
    # pred_area = r"C:\Users\brighten\Desktop\act_well\result_Columbia"
    # save_path = r'C:\Users\brighten\Desktop\show_result'

    # # cod10k
    # src_path = r'C:\Users\brighten\Desktop\show_result\coverage'  # COD10K_tamper_0.png
    # gt_path = r"C:\Users\brighten\Desktop\DATA\public_dataset\coverage\gt"  # COD10K_Gt_0.bmp

    src_path = r'C:\Users\brighten\Desktop\show_result\casia'  # COD10K_tamper_0.png
    gt_path = r"C:\Users\brighten\Desktop\DATA\public_dataset\casia\gt"  # COD10K_Gt_0.bmp

    # save_path = r'C:\Users\brighten\Desktop\show_result\casia_gt'
    # gt2=r'C:\Users\brighten\Desktop\show_result\casia_gt1'
    # gt3=r'C:\Users\brighten\Desktop\show_result\casia_gt2'
    save_path = r'C:\Users\brighten\Desktop\show_result\coverage_gt'
    gt2=r'C:\Users\brighten\Desktop\show_result\coverage_gt1'
    gt3=r'C:\Users\brighten\Desktop\show_result\coverage_gt2'
    # all_picture_save(src_path, gt_path,save_path)  # 文件夹内所有图片
    gt_config(save_path,gt2,gt3)