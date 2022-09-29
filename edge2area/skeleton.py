# 文件夹内逐个图像  做一个中间结果   膨胀腐蚀之后做一个细化     保存在  save_path 文件夹  中
# 细化算法 可以考虑做边缘滤波,但是估计不行
# 这个代码运行非常慢，尤其是遇到复杂的结构的图像
# 得到了消除小断点的结构骨架
# 这是步骤 1
import os

import numpy as np
import skimage.morphology as dilation
from PIL import Image

from get_skeleton import zhangSuen


def one_picture():
    _mask_path = r'./pictures/output2_Tp_D_CNN_S_N_ani00087_ani00088_10102.tif'
    pred_mask = Image.open(_mask_path)
    pred_mask = np.array(pred_mask)
    pred_mask = np.where(pred_mask > 100, 1, 0)  # 二值化  ( x>100 =1 : x<100=0 )
    # selem1 = np.ones((20, 20)) # 加入膨胀腐蚀会导致图像的大范围变形，一定不能加
    # _pred_mask_dilation = dilation.binary_dilation(pred_mask, selem1) # 膨胀
    # _pred_mask_dilation = np.where(_pred_mask_dilation == True, 1, 0)
    # _pred_mask_erosion = dilation.binary_erosion(_pred_mask_dilation, selem1) # 腐蚀
    # _pred_mask_erosion = np.where(_pred_mask_erosion == True, 1, 0)
    _pred_mask_erosion = pred_mask
    pred_mask_skeleton = zhangSuen(_pred_mask_erosion)
    pred_mask_skeleton = np.where(pred_mask_skeleton, 255, 0)
    # 8bit(0~255)，需要指定格式转换成uint8否则保存的图像全黑
    pred_mask_skeleton = Image.fromarray(pred_mask_skeleton.astype(np.uint8))
    pred_mask_skeleton.save('./膨胀腐蚀.png')


def skeletoning(input, output):  # 封装
    mask_path = input
    save_path = output
    for index, item in enumerate(os.listdir(mask_path)):
        _mask_path = os.path.join(mask_path, item)
        _save_path = os.path.join(save_path, item)
        try:
            pred_mask = Image.open(_mask_path)
            pred_mask = np.array(pred_mask)
            pred_mask = np.where(pred_mask > 100, 1, 0)  # 二值化  ( x>100 =1 : x<100=0 )
            selem1 = np.ones((20, 20))
            # _pred_mask_dilation = dilation.binary_dilation(pred_mask, selem1) # 膨胀
            # _pred_mask_dilation = np.where(_pred_mask_dilation == True, 1, 0)
            # _pred_mask_erosion = dilation.binary_erosion(_pred_mask_dilation, selem1) # 腐蚀
            # _pred_mask_erosion = np.where(_pred_mask_erosion == True, 1, 0)
            _pred_mask_erosion = pred_mask
            pred_mask_skeleton = zhangSuen(_pred_mask_erosion)
            pred_mask_skeleton = np.where(pred_mask_skeleton, 255, 0)
            # 8bit(0~255)，需要指定格式转换成uint8否则保存的图像全黑
            pred_mask_skeleton = Image.fromarray(pred_mask_skeleton.astype(np.uint8))
            pred_mask_skeleton.save(_save_path)
        except Exception:
            pass
        print("{}/{}".format(index + 1, len(os.listdir(mask_path))))


if __name__ == '__main__':
    mask_path = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\pred_train'  # 算法结果，未处理
    save_path = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result1'  # 中间结果  已处理保存
    # skeletoning(mask_path, save_path)
    one_picture()
