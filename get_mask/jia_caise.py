#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2 as cv

import numpy as np
from PIL import Image
output2=Image.open(r"C:\Users\brighten\Desktop\test_data\公开数据集\虎子_gt.png")
y = cv.cvtColor(np.asarray(output2),cv.COLOR_RGB2BGR)
output2_cai = cv.applyColorMap(y, cv.COLORMAP_JET)
cv.imwrite("7.png",output2_cai)
cv.waitKey(0)