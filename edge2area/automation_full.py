# -*- coding: utf-8 -*-
import os
from PIL import Image
import cv2
import numpy as np


# 自动泛洪法
# 有一定的准确度问题

def fillHole(im_in):
    im_floodfill = im_in.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image  转化
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv

    return im_out


'''src=cv2.imread("./pictures/output_canong3_canonxt_sub_08.tif")
s=fillHole(src)
cv2.imshow("1",s)
cv2.waitKey(0)'''


def picture_fulling(input, output):
    src_path = input
    save_path = output
    for index, item in enumerate(os.listdir(src_path)):
        _src_path = os.path.join(src_path, item)
        _save_path = os.path.join(save_path, item)
        pred_mask = Image.open(_src_path).convert('L')

        pred_mask = np.asarray(pred_mask)
        result = fillHole(pred_mask)

        # result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
        result = Image.fromarray(result.astype(np.uint8))
        result.save(_save_path)
        print("{}/{}".format(index + 1, len(os.listdir(src_path))))
        print(item)


if __name__ == "__main__":
    src_path = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result2'
    save_path = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result3'
    picture_fulling(src_path, save_path)