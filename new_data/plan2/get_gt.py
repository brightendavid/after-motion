#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# 直接读取PSD 文件是不行的，出现的复制图层是一个小图，不能体现位置信息 不行
# from psd_tools import PSDImage
#
# from os import path
#
# psd = PSDImage.open(r'C:\Users\brighten\Desktop\ceshi\save\COD10K-CAM-1-Aquatic-2-ClownFish-11.psd')
#
#
# def extractLayerImge(layer):
#     layer_image = layer.composite()
#
#     layer_image.save(r'C:\Users\brighten\Desktop\后处理工作\new_data\plan2\pic_test_file\%s.tif' % layer.name)
#
# for layer in psd.descendants():
#     print('descendants ', layer)
#     if layer.name=="图层 1":
#         extractLayerImge(layer)

# 图像比较方法也是不行的
# import os
# import numpy as np
# from PIL import Image
# import cv2 as cv
# png_dir = r'C:\Users\brighten\Desktop\ceshi\save_png'
# src_dir = r"C:\Users\brighten\Desktop\ceshi\src_png"
# src_list = os.listdir(src_dir)
# for src_name in src_list:
#     png_name = src_name.replace("jpg", "png")
#     png_name=os.path.join(png_dir,png_name)
#     src = os.path.join(src_dir, src_name)
#     png=cv.imread(png_name,cv.IMREAD_GRAYSCALE)
#     src_yuan =cv.imread(src,cv.IMREAD_GRAYSCALE)
#     # cv.imshow("2",png)
#     # cv.imshow("3",src_yuan)
#     mask=np.where(abs(png-src_yuan)>5,255,0)
#     mask =mask.astype(np.uint8)
#     cv.imshow("1",mask)
#     cv.waitKey(0)


# 还是操作蒙版比较好
# 在保存tempered pic 之后操作  会破坏PSD 的结构
import os

import cv2
from win32com.client import Dispatch, GetActiveObject, GetObject
from PIL import Image
import numpy as np
from new_data.plan2.psd2png import pas_to_png

import skimage.morphology as dilation


def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:  # jpg为-3/py为-2
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
    return Filelist


# 打开文件夹下 的所有PSD 文件   并通过脚本操作
def save_as_gt(operation_dir):
    # org_img_folder = r'C:\Users\brighten\Desktop\test\src\\'
    org_img_folder = operation_dir
    # 检索文件
    imglist = getFileList(org_img_folder, [], 'psd')

    print('本次执行检索到 ' + str(len(imglist)) + ' 个psd文件\n')

    # print(len(imglist))

    if len(imglist):
        try:
            app = GetActiveObject("Photoshop.Application")
        except:
            app = Dispatch("Photoshop.Application")

        # fileName = r"cod_high_resolution_Terrestrial-45-Spider-2513.png"
        # fileName=os.path.join(org_img_folder, fileName)
        # docRef = app.Open(fileName)

        for imgpath in imglist:
            imgname = os.path.splitext(os.path.basename(imgpath))[0]
            print(imgpath)
            docRef = app.Open(imgpath)
            # 脚本变更为  先删除背景(是纯白色 的无效背景)   再   把剩下的两个图层当作src 和mask 进行操作(使用的为ps action  注意调用)   最后保存为png 即可
            try:
                app.DoAction('get_gt', '默认动作')  # 关键操作   要PS实现录入动作  油漆桶 要带不连续属性
                # 所有错误 全部跳过 进行下一张图片运算
            except Exception as e:
                os.remove(imgpath)


def get_double_edge(mask):
    # 是原本用于生成双边缘的代码  输入输出为   3通道黑白mask  输出为   单通道灰度的mask 带有双边缘
    # 这个三通道就不合理
    # 输入的是黑白的mask图  Image 形式  3通道的
    if len(mask.shape) != 2:
        # print('the shape of mask is :', mask.shape)
        mask = mask[:, :, 0]
        print("通道数不为1")
    # mask = np.array(mask)[:, :]
    # cv2.imshow("2", mask)
    # cv2.waitKey(0)
    # 给的灰度阈值很宽松
    mask = np.where(mask >200, 1, 0)

    # print('the shape of mask is :', mask.shape)
    selem = np.ones((3, 3))
    dst_8 = dilation.binary_dilation(mask, selem=selem)
    dst_8 = np.where(dst_8 == True, 1, 0)

    difference_8 = dst_8 - mask
    difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
    difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)

    double_edge_candidate = difference_8_dilation + mask
    double_edge = np.where(double_edge_candidate == 2, 1, 0)
    ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(mask == 1, 50, 0)
    # 所以内侧边缘就是100的灰度值
    return ground_truth


def gen_double_edge(mask1_dir, out_dir):
    mask1_list = os.listdir(mask1_dir)
    print("------ 从现在开始从原有的mask1 生成双边缘的mask -----")
    for src_name in mask1_list:
        # print("src_name:", end=" ")
        print(src_name)
        mask_name = src_name
        src_path = os.path.join(mask1_dir, mask_name)
        # src = Image.open(src_path)
        src = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        # print(src.shape)
        src1 = cv2.imread(src_path, cv2.COLOR_RGB2BGR)
        # cv2.imshow("1", src1)
        # cv2.waitKey(0)

        apaph = src[:, :, -1]
        src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
        mask = np.ones((src.shape[0], src.shape[1])) * 255

        # alaph 数值计算公式
        mask = mask * (255 - apaph) / 255 + apaph * src1 / 255

        # cv2.imshow("masK", mask)
        # cv2.imwrite("./mask.png", mask)  # 带灰度的，通道数1

        save_path = os.path.join(out_dir, mask_name)
        two_edge_gt = get_double_edge(mask)
        cv2.imwrite(save_path, two_edge_gt)


if __name__ == "__main__":
    org_img_folder = r'C:\Users\brighten\Desktop\ceshi\save\\'  # 此处是PSD源文件
    # 操作此目录下 的所有psd 文件
    save_as_gt(org_img_folder)  # 此处做了一个PS动作流
    psd_dir = r"C:\Users\brighten\Desktop\ceshi\save"  # 此处是PSD源文件 （修改后）
    out_dir = r'C:\Users\brighten\Desktop\ceshi\save_gt'  # 此处是  篡改部分全透明    黑色部分为未篡改部分
    # # 输入psd的保存路径  png 的输出路径   psd转png  结果在out_dir
    pas_to_png(psd_dir, out_dir)  # 此处是  篡改部分全透明    黑色部分为未篡改部分
    last_gt = r'C:\Users\brighten\Desktop\ceshi\double_gt'
    gen_double_edge(out_dir, last_gt)