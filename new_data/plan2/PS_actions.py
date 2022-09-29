#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# This script demonstrates how to use the action manager to execute a
# previously defined action liek the default actions that ships with Photoshop
# Or one that you've created yourself. The name of the action comes from
# Photoshop's Actions Palette


# 功能是把黑色背景的物体复制一份,  调用的是ps的脚本  保存时，需要人为操作   覆盖原图   需要保存
# 输入为图片
# 只能操作一次  注意保存
import os
import random

from win32com.client import Dispatch, GetActiveObject, GetObject


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
def copy_move(operation_dir):
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
            # print(imgpath)
            docRef = app.Open(imgpath)
            # 脚本变更为  先删除背景(是纯白色 的无效背景)   再   把剩下的两个图层当作src 和mask 进行操作(使用的为ps action  注意调用)   最后保存为png 即可
            try:
                count=(random.randint(0, 9))
                # app.DoAction('蒙版', '默认动作')  # 原版本  的蒙版   单纯进行了移动操纵，没其他的任何操作 检测结果太好了  认为不能胜任困难样本 的定位
                # 核心脚本 获取新数据集   其他的代码全部是配套   对于图像的模式化要求很高
                # 做一个随机数
                if count % 3 == 2:
                    app.DoAction('蒙版4', '默认动作')  # 关键操作   要PS实现录入动作  两个的差别在于向上和向下移动  3是向下移动
                elif count % 3 == 1:
                    app.DoAction('蒙版2', '默认动作')  # 关键操作   要PS实现录入动作    两个脚本现在全部有形状变化的功能  2是向上移动
                else:
                    app.DoAction('蒙版3', '默认动作')  # 这是放大  三者比例相同    放大向右移动
                count += 1
            except Exception as e:
                os.remove(imgpath)


if __name__ == "__main__":
    org_img_folder = r'C:\Users\brighten\Desktop\ceshi\save\\'
    # 操作此目录下 的所有psd 文件
    copy_move(org_img_folder)