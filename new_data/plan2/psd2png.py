#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Import local modules
import os

from psd_tools import PSDImage
from glob import glob


def pas_to_png(psd_dir,outdir):
    psd_list = os.listdir(psd_dir)
    for i in range(len(psd_list)):
        name = os.path.join(psd_dir, psd_list[i])
        psd = PSDImage.open(name)
        psd[2].visible = False  # 对应的把不需要的图层隐藏不可见
        save_name = psd_list[i][:-4] + ".png"
        save_name = os.path.join(outdir, save_name)
        psd.composite(True).save(save_name)  # True这个参数一定要有，上边的隐藏语句才有效，不然还是合并了全部图层可见。 psd 转png 才会把透明像素自动转为白色
        # 而其他方式保存会导致透明度丢失

if __name__== "__main__":
    psd_dir = r"C:\Users\brighten\Desktop\ceshi\save"
    out_dir = r'C:\Users\brighten\Desktop\ceshi\save_png'
    # 输入psd的保存路径  png 的输出路径   psd转png  结果在outdir
    # pas_to_png(psd_dir,out_dir)
    pas_to_png(r"C:\Users\brighten\Desktop\test\new",r"C:\Users\brighten\Desktop\test")