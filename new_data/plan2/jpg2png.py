#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 本来打算拿这个对原图操作的，但是实际上没什么用
# 这个脚本在plan2 中没有采用
import os
import numpy as np
from PIL import Image
import cv2 as cv

src_dir = r"C:\Users\brighten\Desktop\ceshi\src"
save_png_dir=r"C:\Users\brighten\Desktop\ceshi\src_png"
src_list = os.listdir(src_dir)
for src_name in src_list:
    src = os.path.join(src_dir, src_name)
    pic=Image.open(src)
    save_name=os.path.join(save_png_dir,src_name)
    save_name=save_name.replace("jpg","png")
    pic.save(save_name)