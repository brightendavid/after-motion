#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

from PIL import Image

def find_unpaired_pics(src_dir, gt_dir):
    # 找出没有配对的pic
    print("-------找出未配对数据 ---------")
    gt_list = os.listdir(gt_dir)
    src_list = os.listdir(src_dir)
    error_list = []
    for src_name in src_list:
        src_path = os.path.join(src_dir, src_name)
        if "jpg" in src_name:
            gt_name = src_name.replace("jpg","bmp")
        else:
            gt_name = src_name.replace("png", "bmp")
        gt_path = os.path.join(gt_dir, gt_name)
        try:
            src = Image.open(src_path)
            gt = Image.open(gt_path)
        except Exception as e:
            error_list.append(src_name)

    print("共有{}个未配对样本:".format(len(error_list)))
    print(error_list)
    return error_list
def find_tongming(src_dir):
    src_list = os.listdir(src_dir)
    src_list.sort()
    count=0
    for i in range(1,len(src_list)):
        name =src_list[i]
        src_name=name[:-4]
        if src_name ==src_list[i-1][:-4]:
            print(src_name)
            count+=1
    print("重名，但是后缀不同的有：{}数量".format(count))

if __name__=="__main__":
    src_path=r"C:\Users\brighten\Desktop\DATA\blur_data\train_src"
    gt_path=r"C:\Users\brighten\Desktop\DATA\blur_data\train_gt"
    find_tongming(src_path)
    # find_unpaired_pics(src_path,gt_path)