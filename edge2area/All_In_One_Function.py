# -*- coding: utf-8 -*-
"""
2022/3/3
填充一张图像，在一个function中
"""
import numpy as np
from PIL import Image
import cv2 as cv
import Get_mask
from Breakpoint_connection import breakpoint_connect as bc1
from Breakpoint_connection2 import breakpoint_connect as bc2
from automation_full import fillHole
from get_skeleton import zhangSuen


def Function(file_name):
    pred_mask = Image.open(file_name)
    pred_mask = np.array(pred_mask)
    pred_mask = np.where(pred_mask > 100, 1, 0)  # 二值化  ( x>100 =1 : x<100=0 )
    _pred_mask_erosion = pred_mask
    pred_mask_skeleton = zhangSuen(_pred_mask_erosion)
    pred_mask_skeleton = np.where(pred_mask_skeleton, 255, 0)
    pred_mask_skeleton = Image.fromarray(pred_mask_skeleton.astype(np.uint8))
    pred_after_connect = connect(pred_mask_skeleton)
    pred_after_full = picture_fulling(pred_after_connect)
    return pred_after_full


def Function2(file_name, gt_name):
    # 加入了gt_mask进行修正 和 断点判定连通域
    pred_mask = Image.open(file_name)
    gt = Image.open(gt_name)
    gt = np.array(gt)
    pred_mask = np.array(pred_mask)
    pred_mask = Get_mask.gtmask_one(pred_mask, gt)
    pred_mask = np.where(pred_mask > 100, 1, 0)  # 二值化  ( x>100 =1 : x<100=0 )
    _pred_mask_erosion = pred_mask
    pred_mask_skeleton = zhangSuen(_pred_mask_erosion)
    pred_mask_skeleton = np.where(pred_mask_skeleton, 255, 0)
    pred_mask_skeleton = Image.fromarray(pred_mask_skeleton.astype(np.uint8))
    # pred_mask_skeleton.show()
    pred_after_connect = connect2(pred_mask_skeleton)
    # cv.imwrite("./pictures/pred_after_connect.png", np.array(pred_after_connect))
    pred_after_full = picture_fulling(pred_after_connect)
    return pred_after_full


def connect2(src):
    pred_mask = src
    if pred_mask.split() == 3:  # 若为3rgb图像 则会读取为灰度
        pred_mask = pred_mask.split()[0]
    pred_mask = np.array(pred_mask)
    pred_mask = np.where(pred_mask > 100, 1, 0)
    try:  # 有断点的情况
        _mask_after_connect = bc2(pred_mask)  # 其实输入的就是经过细化之后的结果
        _mask_after_connect = np.where(_mask_after_connect, 255, 0)
        _mask_after_connect = Image.fromarray(_mask_after_connect.astype(np.uint8))
    except IndexError:  # 无断点会报IndexError错
        print("无断点")
        _mask_after_connect = pred_mask  # mask_after_connect为原图
        _mask_after_connect = np.where(_mask_after_connect, 255, 0)
        _mask_after_connect = Image.fromarray(_mask_after_connect.astype(np.uint8))
    else:
        pass
    return _mask_after_connect


def connect(src):
    pred_mask = src
    if pred_mask.split() == 3:  # 若为3rgb图像 则会读取为灰度
        pred_mask = pred_mask.split()[0]
    pred_mask = np.array(pred_mask)
    pred_mask = np.where(pred_mask > 100, 1, 0)
    try:  # 有断点的情况
        _mask_after_connect = bc1(pred_mask)  # 其实输入的就是经过细化之后的结果
        _mask_after_connect = np.where(_mask_after_connect, 255, 0)
        _mask_after_connect = Image.fromarray(_mask_after_connect.astype(np.uint8))
    except IndexError:  # 无断点会报IndexError错
        print("无断点")
        _mask_after_connect = pred_mask  # mask_after_connect为原图
        _mask_after_connect = np.where(_mask_after_connect, 255, 0)
        _mask_after_connect = Image.fromarray(_mask_after_connect.astype(np.uint8))
    else:
        pass
    return _mask_after_connect


def picture_fulling(src):
    pred_mask = src
    pred_mask = np.asarray(pred_mask)
    result = fillHole(pred_mask)
    result = Image.fromarray(result.astype(np.uint8))
    return result


if __name__ == "__main__":
    path = r'./pictures/Tp_D_CRN_M_N_txt00063_txt00017_10835.jpg'
    gt_path = r'./pictures/Tp_D_CRN_M_N_txt00063_txt00017_10835_gt.png'
    full_pic = Function2(path, gt_path)
    # full_pic =Function(path)
    # full_pic.show()  # 能跑
    full_pic.save(r'./pictures/1.png')
