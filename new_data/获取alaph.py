import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import cv2

# 输入必须是带有透明度通道的图像
# 打印出所有alpha 值
# cv2.IMREAD_UNCHANGED
img = cv2.imread(r"C:\Users\brighten\Desktop\1.png", cv2.cv2.IMREAD_UNCHANGED)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        print(img[i,j,-1],end=" ")
        # print(img[i, j], end=" ")
    print("")
print(img.shape)
