#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 两点连接，做直线
# 没有用
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as dilation
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.draw import line
from calculate_distance import min_dist2
from get_skeleton import zhangSuen


def breakpoint_connect(mask_skeleton):  # 输入的是细化后的图像
    mask_skeleton = torch.Tensor(mask_skeleton).unsqueeze(0).unsqueeze(0)
    kernel = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    _mask = F.conv2d(mask_skeleton, kernel, stride=1, padding=1)
    _mask = _mask.squeeze(0).squeeze(0)
    _mask = np.where((_mask > 0) & (_mask < 2), 1, 0)
    # 卷积之后满足大于0，小于2的点可能是细化之后骨架周围的点，我们要取的是在骨架上的点，所以还会进行下面操作：
    mask_skeleton = mask_skeleton.squeeze(0).squeeze(0)
    mask_skeleton = np.array(mask_skeleton)
    _mask = _mask + mask_skeleton
    breakpoint = np.where(_mask == 2, 1, 0)
    breakpoint_loc = np.where(_mask == 2)  # 断点的坐标点
    breakpoint_loc = np.array(breakpoint_loc)

    mask_skeleton = mask_skeleton + breakpoint  # 断点可视化展示，黄色的是断点，绿色的是骨架
    '''可视化模块'''
    plt.figure("断点判断")
    plt.imshow(mask_skeleton)
    plt.show()

    # 希望每次调用calculate_distance函数，返回最小距离坐标（两点），
    # 在此函数中将两点连接，之后从breakpoint_loc中移除这两点，再次调用calculate_distance，
    # 直到breakpoint_loc只有一个值(或没有值)
    while True:
        if len(breakpoint_loc[0]) <= 1:
            break
        a, b = min_dist2(breakpoint_loc)
        # min_index=min_dist(breakpoint_loc) #按顺序
        # skimage.draw中的line方法，使用规则见包里解析
        start, end = line(breakpoint_loc[0][a], breakpoint_loc[1][a], breakpoint_loc[0][b], breakpoint_loc[1][b])
        mask_skeleton[start, end] = 2

        '''可视化模块'''
        # plt.figure('连线处理演示')
        # plt.imshow(mask_skeleton)
        # plt.show()

        delete_list = []
        delete_list.append(a)
        delete_list.append(b)
        breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)

    '''可视化模块'''
    # plt.figure('连线处理演示')
    # plt.imshow(mask_skeleton)
    # plt.show()
    return mask_skeleton


'''一张图片进行实验'''
# if __name__=='__main__':
#     mask_path=r'C:\Users\97493\Desktop\调试代码\连接断点\预测Mask\coverage\output2_81t.bmp'#mask路径
#     save_path=r'C:\Users\97493\Desktop\调试代码\连接断点\后处理之后的Mask\coverage\output2_81t.bmp'
#     pred_mask = Image.open(mask_path)
#     pred_mask = np.array(pred_mask)
#     pred_mask = np.where(pred_mask > 50, 1, 0)
#     # 先膨胀、再腐蚀
#     selem1 = np.ones((23, 23))
#     _pred_mask_dilation = dilation.binary_dilation(pred_mask, selem1)
#     _pred_mask_dilation = np.where(_pred_mask_dilation == True, 1, 0)
#     _pred_mask_erosion = dilation.binary_erosion(_pred_mask_dilation, selem1)
#     _pred_mask_erosion = np.where(_pred_mask_erosion == True, 1, 0)
#     _pred_mask = _pred_mask_erosion - pred_mask
#     _pred_mask = np.where(_pred_mask == 1, 100, 0)
#     pred_mask=np.where(pred_mask,200,0)
#     #_pred_mask是膨胀腐蚀之后的，pred_mask是原mask
#     _pred_mask = _pred_mask + pred_mask
#     # plt.figure(4)
#     # plt.imshow(_pred_mask)
#     # plt.show()
#     mask_skeleton = zhangSuen(_pred_mask_erosion)  # 细化算法，提取框架
#
#     connected_mask = breakpoint_connect(mask_skeleton)
#     connected_mask=np.where(connected_mask,170,0)
#
#     mask_skeleton = np.where(mask_skeleton == 1, 3, 0)
#     _new_mask=_pred_mask+connected_mask
#     # mask_skeleton=np.where(mask_skeleton==1,3,0)
#     # _new_mask=_new_mask+mask_skeleton
#     plt.figure(4)
#     plt.imshow(_new_mask)
#     plt.show()
#
#     #保存
#     new_mask=Image.fromarray(_new_mask)
#     new_mask=new_mask.convert("L")
#     new_mask.save(save_path)


'''多张图片实验效果'''
if __name__ == '__main__':
    mask_path = r'C:\Users\97493\Desktop\调试代码\连接断点\预测Mask\columb'  # mask路径
    _save_root = r'D:\17.CV\0324_两阶段_0306模型,只监督条带区域，带8张图\后处理之后\columb'

    num = 0
    for index, item in enumerate(os.listdir(mask_path)):
        _mask_path = os.path.join(mask_path, item)
        save_path = os.path.join(_save_root, item)
        pred_mask = Image.open(_mask_path)
        pred_mask = np.array(pred_mask)
        pred_mask = np.where(pred_mask > 50, 1, 0)
        # 先膨胀、再腐蚀
        selem1 = np.ones((23, 23))
        _pred_mask_dilation = dilation.binary_dilation(pred_mask, selem1)
        _pred_mask_dilation = np.where(_pred_mask_dilation == True, 1, 0)
        _pred_mask_erosion = dilation.binary_erosion(_pred_mask_dilation, selem1)
        _pred_mask_erosion = np.where(_pred_mask_erosion == True, 1, 0)
        _pred_mask = _pred_mask_erosion - pred_mask
        _pred_mask = np.where(_pred_mask == 1, 100, 0)
        pred_mask = np.where(pred_mask, 200, 0)
        # _pred_mask是膨胀腐蚀之后的，pred_mask是原mask
        _pred_mask = _pred_mask + pred_mask
        '''可视化模块'''
        # plt.figure("膨胀腐蚀后的结果")
        # plt.imshow(_pred_mask)
        # plt.show()
        mask_skeleton = zhangSuen(_pred_mask_erosion)  # 细化算法，提取框架

        connected_mask = breakpoint_connect(mask_skeleton)
        connected_mask = np.where(connected_mask > 0, 170, 0)

        # mask_skeleton = np.where(mask_skeleton == 1, 3, 0)
        _new_mask = _pred_mask + connected_mask
        # mask_skeleton=np.where(mask_skeleton==1,3,0)
        # _new_mask=_new_mask+mask_skeleton
        '''可视化模块'''
        # plt.figure("膨胀腐蚀+断点连接")
        # plt.imshow(_new_mask)
        # plt.show()

        # 保存
        new_mask = Image.fromarray(_new_mask)
        new_mask = new_mask.convert("L")
        new_mask.save(save_path)
        num = num + 1
        print('（{}/{}）'.format(num, len(os.listdir(mask_path))))
    print('done!')
