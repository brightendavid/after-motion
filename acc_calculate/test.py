#!/usr/bin/env python
# -*- coding:utf-8 -*-
import base64

import cv2
from PIL import Image
import numpy as np
if __name__ == "__main__":
    f = Image.open(r'C:\Users\brighten\Desktop\ceshi\double_gt\COD10K-CAM-3-Flying-61-Katydid-4038.png').convert('L')
    print(f.size)

    img = cv2.cvtColor(np.asarray(f), cv2.COLOR_GRAY2BGR)
    print(img.shape)
    cv2.imshow("1",img)
    cv2.waitKey(0)
    # f=f.convert('rgb')