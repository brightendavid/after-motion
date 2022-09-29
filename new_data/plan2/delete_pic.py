#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
哪些图片是明显不合格的  思考规律

首先是全黑白的，必然是没有选择好

发现在第一次调用PS动作中出错的，在第二次动作也会出错
"""
import os
import sys
import shutil
import cv2 as cv
from PIL import Image
import numpy as np


def find_bugs(src_dir, gt_dir):
    # 输出贴边的gt
    # 对贴边导致的gt标注不准确的问题进行修正  找出所有贴边的图片
    # 用新的魔棒设置（不连续魔棒）进行重构
    # 在修改完成魔棒属性之后 此方法无用
    print("------- delete wrong pictures ---------")
    gt_list = os.listdir(gt_dir)
    error_list = []
    for gt_list in gt_list:
        gt_name = gt_list
        # src_path = os.path.join(src_dir, src_name)
        gt_path = os.path.join(gt_dir, gt_name)
        # print(gt_name)

        # todo 删除条件  就删除贴边的
        gt = cv.imread(gt_path)
        a = gt.shape[0]
        b = gt.shape[1]
        # print("src_path  :",end=" ")
        # print(src_path)
        # print((np.where(src[:,:,1]==0))[1].shape[0])
        x = np.where(gt == 255)
        if (0 in x[0]) or (a - 1 in x[0]) or (0 in x[1]) or (b - 1 in x[1]):
            error_list.append(gt_name)
    print(error_list)
    print(len(error_list))
    return error_list


def delete_all(delete_dir):
    # 删除全部文件  ！！！！！ 慎用
    # 写这个东西的目的是每次移入目标文件夹20个图片
    # 此时需要重置文件夹
    # 删除以下所有文件夹中的文件  使用os.remove 方法
    print("-----------  delete  modles -------------")

    for dir_name in delete_dir.values():
        # src_list = os.listdir(dir)
        print(dir_name)
        file_list = os.listdir(dir_name)
        for file_name in file_list:
            file_path = os.path.join(dir_name, file_name)
            os.remove(file_path)


def restart(erroe_list):
    # list目录中的图片名，为png格式，  将对应的src 和gt 从源数据集中找出来，copy到 使用目录

    # 旧目录
    gt_yuan_dir = r"C:\Users\brighten\Desktop\COD10K-v3\Train\GT_Object"  # 导入的原
    src_yuan_dir = r"C:\Users\brighten\Desktop\COD10K-v3\Train\Image"

    # 新目录
    src_new_dir = r"C:\Users\brighten\Desktop\test_cod\Image"
    gt_new_dir = r"C:\Users\brighten\Desktop\test_cod\GT_Object"

    if os.path.exists(gt_yuan_dir) and os.path.exists(src_yuan_dir):
        pass
    else:
        print("给出输入的实例检测  src 和 mask  源数据集!!!  必须存在")
        sys.exit()
    # 下面的6个目录随便给
    # 调用ps尽量给绝对路径

    for error_name in erroe_list:
        src_name = os.path.join(src_yuan_dir, error_name)
        src_name = src_name.replace("png", "jpg")
        mask_name = os.path.join(gt_yuan_dir, error_name)

        #  dirname = os.path.join(rootdir, item)  # 将根目录与文件夹名连接起来，获取文件目录
        try:
            shutil.copy(src_name, src_new_dir)  # 移动文件到目标路径
            shutil.copy(mask_name, gt_new_dir)  # 移动文件到目标路径
        except FileNotFoundError:
            continue


def move_files(rootdir, des_path, num=25):
    # 从 root_dir 移动到 des_path 中，移动的文件类型不限制，按照字符顺序先后  移动文件数量为num个
    print("-----move  models ------")
    count = 0
    for item in os.listdir(rootdir):  # 遍历该文件夹中的所有文件
        dirname = os.path.join(rootdir, item)  # 将根目录与文件夹名连接起来，获取文件目录
        print(dirname)
        shutil.move(dirname, des_path)  # 移动文件到目标路径
        count += 1
        if count == num:
            break


def find_posible_errors(src_dir, gt_dir):
    gt_yuan_dir = r"C:\Users\brighten\Desktop\GT_Object"  # 导入的原
    src_yuan_dir = r"C:\Users\brighten\Desktop\Image"
    # 找出可能错误样本  就是篡改部分太大了的图片  根据输出的gt
    print("------- delete wrong pictures ---------")
    gt_list = os.listdir(gt_dir)
    error_list = []
    for gt_name in gt_list:
        # src_path = os.path.join(src_dir, src_name)
        gt_path = os.path.join(gt_dir, gt_name)
        # print(gt_name)
        gt = cv.imread(gt_path)
        a = gt.shape[0]
        b = gt.shape[1]
        if a != 320 or b != 320:
            # error_list.append(gt_name)
            cv.imshow("1", gt)
            cv.waitKey(0)
        # print("src_path  :",end=" ")
        # print(src_path)
        # print((np.where(src[:,:,1]==0))[1].shape[0])
        x = np.where(gt == 0)
        if len(x[1]) < a * b * 3 // 2:  # 黑色未篡改部分应当占据50%以上
            error_list.append(gt_name)
            cv.imshow("1", gt)
            cv.waitKey(0)
    print("共有{}个嫌疑的样本:".format(len(error_list)))
    print(error_list)
    return error_list


def find_unpaired_pics(src_dir, gt_dir):
    # 找出没有配对的pic
    print("-------找出未配对数据 ---------")
    gt_list = os.listdir(gt_dir)
    src_list = os.listdir(src_dir)
    error_list = []
    for src_name in src_list:
        src_path = os.path.join(src_dir, src_name)
        gt_name = src_name
        gt_path = os.path.join(gt_dir, gt_name)
        try:
            src = Image.open(src_path)
            gt = Image.open(gt_path)
        except Exception as e:
            error_list.append(src_name)

    print("共有{}个未配对样本:".format(len(error_list)))
    print(error_list)
    return error_list


def weidu_zhuanhua(src_dir, gt_dir):
    gt_list = os.listdir(gt_dir)
    src_list = os.listdir(src_dir)
    for src_name in src_list:
        src_path = os.path.join(src_dir, src_name)
        gt_name = src_name
        gt_path = os.path.join(gt_dir, gt_name)
        src = Image.open(src_path)
        gt = Image.open(gt_path)
        a, b = src.size
        print(src.size)
        gt.show()
        if a != 320 or b != 320:
            src = src.resize((320, 320))
            src.save(src_path)
            gt = gt.resize((320, 320))
            gt.save(gt_path)
            continue
        if len(src.split()) != 3 or len(gt.split()) != 1:
            print(src_path)
        # src = Image.open(src_path).convert('RGB')
        # gt = Image.open(gt_path).convert('L')
        # src.save(src_path)
        # gt.save(gt_path)


if __name__ == "__main__":
    # src_dir = r'C:\Users\brighten\Desktop\ceshi\save_png'  # 保存的篡改图片  rgb图目录
    # gt_dir = r"C:\Users\brighten\Desktop\ceshi\double_gt"
    dirs = {
        "src_dir": r"C:\Users\brighten\Desktop\ceshi\src",
        "mask_dir": r"C:\Users\brighten\Desktop\ceshi\mask",
        "psd_dir": r"C:\Users\brighten\Desktop\ceshi\save",  # psd 保存路径
        "tempered_dir": r'C:\Users\brighten\Desktop\ceshi\save_png',  # 保存的篡改图片  rgb图目录  输出1！！
        "out_dir": r'C:\Users\brighten\Desktop\ceshi\save_gt',  # 此处是  篡改部分全透明    黑色部分为未篡改部分
        "last_gt": r'C:\Users\brighten\Desktop\ceshi\double_gt',  # 双边缘   # 保存的双边缘gt 是灰度图  输出2！！
    }
    delete_dir = dirs.copy()
    delete_dir.pop("tempered_dir")  # 这两个不能删除
    delete_dir.pop("last_gt")
    # weidu_zhuanhua(src_dir, gt_dir)
    # src_dir = r"C:\Users\brighten\Desktop\DATA\COD10K_new_tampered\save_png"
    # gt_dir = r"C:\Users\brighten\Desktop\DATA\COD10K_new_tampered\double_gt"
    # weidu_zhuanhua(src_dir, gt_dir)
    # error_list = find_bugs(src_dir, gt_dir)
    src_dir = r'C:\Users\brighten\Desktop\ceshi\save_png'
    gt_dir = r'C:\Users\brighten\Desktop\ceshi\double_gt'
    error_list2 = find_posible_errors(src_dir, gt_dir)
    # # queshi_list=find_unpaired_pics(src_dir, gt_dir)
    restart(error_list2)
    delete_all(delete_dir)
