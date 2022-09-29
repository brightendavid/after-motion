# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np


# 步骤 3 人为给定种子点法
# 存在争议 现改为使用automation_full.py 代码，自动填充空白区域

def get_x_y(path, n):
    im = Image.open(path)
    plt.imshow(im, cmap=plt.get_cmap("gray"))
    pos = plt.ginput(n)
    return pos


def regionGrow(gray, seeds, thresh, p):
    seedMark = np.zeros(gray.shape)
    if p == 8:
        connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    elif p == 4:
        connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    while len(seeds) != 0:

        pt = seeds.pop(0)
        for i in range(p):
            tmpX = int(pt[0] + connection[i][0])
            tmpY = int(pt[1] + connection[i][1])

            if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                continue

            if abs(int(gray[tmpX, tmpY]) - int(gray[pt])) < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = 255  # 不保留原有边缘
                seeds.append((tmpX, tmpY))
    seedMark = seedMark + gray  # 准确度更高  修改
    return seedMark
    # return gray


def one_picture():
    path = r"./pictures/output_canong3_canonxt_sub_05.tif"
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    seeds = get_x_y(path=path, n=3)
    print("strat:")
    new_seeds = []
    for seed in seeds:
        print(seed)
        new_seeds.append((int(seed[1]), int(seed[0])))

    result = regionGrow(gray, new_seeds, thresh=3, p=4)

    result = Image.fromarray(result.astype(np.uint8))
    result.show()
    result.save("./1.png")


def fulling(src_path, save_path):
    for index, item in enumerate(os.listdir(src_path)):
        _src_path = os.path.join(src_path, item)
        _save_path = os.path.join(save_path, item)
        pred_mask = Image.open(_src_path)
        if pred_mask.split() == 3:
            pred_mask = pred_mask.split()[0]

        gray = pred_mask
        gray = np.asarray(gray)
        x = np.where(gray == 255)
        x = np.array(x)

        _, y = x.shape
        if y <= 10:
            continue
        print(y)
        seeds = get_x_y(path=_src_path, n=5)
        print("strat:")
        new_seeds = []
        for seed in seeds:
            print(seed)
            new_seeds.append((int(seed[1]), int(seed[0])))
        result = regionGrow(gray, new_seeds, thresh=3, p=4)

        # result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
        result = Image.fromarray(result.astype(np.uint8))
        result.save(_save_path)
        print("{}/{}".format(index + 1, len(os.listdir(src_path))))
        print(item)


if __name__ == '__main__':
    # src_path = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\casia_test\pred_train\result2'
    # save_path = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\casia_test\pred_train\result3'
    # fulling(src_path, save_path)
    one_picture()
