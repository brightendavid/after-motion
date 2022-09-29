#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by haoran
time:8-24
cod10k 类
输入的是经过resize之后的图，resize的方法为双三次插值
mask---> 篡改后的mask --->双边缘


2021/8/1
sjw
# 输入必须为mask   和    src     根据mask    生成了双边缘的gt  和splicing后的结果
要把 crop的函数改为  边缘的透明度为100  在复制的时候，加入对于透明度的计算即可
"""
import os
import numpy as np
import cv2 as cv
from PIL import Image
import traceback
import skimage.morphology as dilation
from new_data.plan1.mask_splicing import MatteMatting


class COD10K:
    """
    这个类在图像的名称中还需要修改
    要求：底纹和img的大小相同
    先将底纹  缩放到  某个大小，
    输入mask 和 img  ,从img 中取  mask 形状，复制到  纹理底纹中   可以叠加

    """
    def __init__(self, img_path, mask_path, img_save_path,gt_save_path):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_save_root_path = img_save_path
        self.img_tamper_save_path_ = os.path.join(img_save_path, 'src')
        self.img_tamper_save_path = os.path.join(self.img_tamper_save_path_,
                                                 'COD10K_tamper_' + img_path.split('\\')[-1])
        self.img_tamper_save_path = self.img_tamper_save_path.replace("jpg", "png")
        self.img_poisson_save_path_ = os.path.join(img_save_path, 'tamper_poisson_result_320_cod10k')
        self.img_poisson_save_path = os.path.join(self.img_poisson_save_path_,
                                                  'tamper_poisson_' + img_path.split('\\')[-1])

        self.img_gt_save_path_ = os.path.join(img_save_path, 'gt')
        self.img_gt_save_path = os.path.join(self.img_gt_save_path_, 'COD10K_tamper_gt_' + mask_path.split('\\')[-1])

        self.tamper_num = 1
        self.bk_shape = (320, 320)
        # 检查输出文件夹
        if os.path.exists(self.img_save_root_path):
            print('输出文件夹已经存在，请手动更换输出文件夹')
        else:
            os.mkdir(self.img_save_root_path)
            os.mkdir(self.img_tamper_save_path_)
            os.mkdir(self.img_poisson_save_path_)
            os.mkdir(self.img_gt_save_path_)
            print('输出文件夹创建成功')

        # 阈值
        self.area_percent_threshold = 0.6
        self.bbox_threshold = 0.1

        pass

    def required_condition(self, area_percent, bbox):
        """
        :param area_percent:
        :param bbox:
        :return:
        """
        if area_percent > self.area_percent_threshold:
            print('面积超出阈值')
            return 'area_over_threshold'
        else:
            return 'area_ok'
        pass

    def gen_data_pair(self, img_path, mask_path):
        tamper_img_list = []
        mask_list = []
        for t_img in os.listdir(img_path):
            tamper_img_list.append(os.path.join(img_path, t_img))
            mask_list.append(os.path.join(mask_path, t_img))
        return tamper_img_list, mask_list
        pass

    def check_edge(self):
        pass

    # 这个函数已经没有用了
    def get_double_edge(self):
        mask = Image.open(self.mask_path)
        mask = np.array(mask)[:, :, 0]
        mask = np.where(mask == 255, 1, 0)

        print('the shape of mask is :', mask.shape)
        selem = np.ones((3, 3))
        dst_8 = dilation.binary_dilation(mask, selem=selem)
        dst_8 = np.where(dst_8 == True, 1, 0)

        difference_8 = dst_8 - mask
        difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
        difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)

        double_edge_candidate = difference_8_dilation + mask
        double_edge = np.where(double_edge_candidate == 2, 1, 0)
        ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(mask == 1,
                                                                                                           50, 0)
        # 所以内侧边缘就是100的灰度值
        return ground_truth





    def crop(self, img, target_shape=(320, 320)):
        img_shape = img.shape
        height = img_shape[0]
        width = img_shape[1]
        random_height_range = height - target_shape[0]
        random_width_range = width - target_shape[1]

        if random_width_range < 0 or random_height_range < 0:
            print('臣妾暂时还做不到!!!')
            traceback.print_exc()
            return 'error'

        random_height = np.random.randint(0, random_height_range)
        random_width = np.random.randint(0, random_width_range)

        return img[random_height:random_height + target_shape[0], random_width:random_width + target_shape[1]]

    def gen_tamper_result(self):
        """
        输入一对图片 src+gt
        :return:
        """
        # read image
        background = Image.open(self.img_path)
        background = np.array(background)
        mask = Image.open(self.mask_path).convert('RGB')
        mask = np.array(mask)
        mask = np.where(mask[:, :, 0] == 255, 1, 0)

        # 找到mask 的矩形区域
        oringal_background = background.copy()
        a = mask
        a = np.where(a != 0)
        bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])  # 在mask 周围圈出一个方形的框
        cut_mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]  # cut_mask 为  mask 标注位置的框
        # cut_area = oringal_background[bbox[0]:bbox[2], bbox[1]:bbox[3]]  # cut_area 为原图标注位置的框

        # 计算物体所占区域面积
        # object_area_percent = cut_mask.size / (self.bk_shape[0] * self.bk_shape[1])

        # 以左上角的点作为参考点，计算可以paste的区域
        background_shape = background.shape
        object_area_shape = cut_mask.shape
        # 要求在      合法范围内!!!   crop  拼接的图像
        paste_area = [background_shape[0] - object_area_shape[0], background_shape[1] - object_area_shape[1]]
        print('the permit paste area is :', paste_area)
        row1 = np.random.randint(0, paste_area[0])
        col1 = np.random.randint(0, paste_area[1])

        # 在background上获取mask的区域
        # temp_background = background.copy()
        mm = MatteMatting(self.img_path, self.mask_path)
        temp_background = mm.save_image(mask_flip=True)
        # print(temp_background.shape)
        cut_area = temp_background[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        # print("shape")
        # print(cut_area.shape)
        # cv.imwrite('1.png', temp_background)
        random_area = False
        if random_area:
            cut_area = temp_background[row1:row1 + object_area_shape[0], col1:col1 + object_area_shape[1], :]  # 4通道
            # 原图 中要被取代的部分
            cut_area[:, :, 0] = cut_area[:, :, 0] * cut_mask
            # 只有在cut_mask  之内的部分要被复制   求出的 是精确的物体分割  框内其他部分为0像素
            cut_area[:, :, 1] = cut_area[:, :, 1] * cut_mask
            cut_area[:, :, 2] = cut_area[:, :, 2] * cut_mask
            cut_area[:, :, 3] = cut_area[:, :, 3] * cut_mask
        else:
            # cut_area = temp_background[row1:row1 + object_area_shape[0], col1:col1 + object_area_shape[1], :]
            cut_area[:, :, 0] = cut_area[:, :, 0] * cut_mask
            cut_area[:, :, 1] = cut_area[:, :, 1] * cut_mask
            cut_area[:, :, 2] = cut_area[:, :, 2] * cut_mask
            cut_area[:, :, 3] = cut_area[:, :, 3] * cut_mask
        # cut_area 保留了4通道  包含了透明度层  周围一圈全部是透明的   图像的周围是半透明的

        for i in range(5):  # 只替换一次 选到了合适的区域就退出  选出黏贴的位置   定位在图像的左上角
            # 求出的是篡改位置的左上角坐标
            # 保存为row2   col2
            row2 = np.random.randint(0, paste_area[0])
            col2 = np.random.randint(0, paste_area[1])
            if abs(row1 - row2) + abs(col1 - col2) < 50:
                print('随机选到的区域太近，最好重新选择')
            else:
                break

        # # 判断object和bg的大小是否符合要求
        # if paste_area[0] < 5 or paste_area[1] < 5:
        #     print('提醒：允许的粘贴区域太小')
        # if paste_area[0] < 1 or paste_area[1] < 1:
        #     print('无允许粘贴的区域')
        #     return False, False, False
        # 随机在background上贴上该mask的区域，并且保证与原区域有一定的像素偏移,然后生成新的mask图

        tamper_image = []
        tamper_mask = []
        tamper_gt = []
        tamper_poisson = []
        for times in range(self.tamper_num):
            # todo  其次    这个换为有透明度的
            bk_mask = np.zeros((background_shape[0], background_shape[1]), dtype='uint8')
            bk_area = np.zeros((background_shape[0], background_shape[1], 4), dtype='uint8')
            bk_mask[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1]] = cut_mask
            bk_area[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1], :] = cut_area

            # bk_area 只有待粘贴的部分为黑  ，其他和background相同

            # todo  首先  background 是原图不能变

            # cv.imshow("3", background[:, :, :])
            # cv.imshow("mask", bk_area[:, :, :])
            # cv.imwrite("mask.png", bk_area[:, :, :-1])

            cv.waitKey(0)
            ks = np.ones((background_shape[0], background_shape[1], 3), dtype=np.float32)
            ks2 = np.ones((background_shape[0], background_shape[1], 3), dtype=np.float32)
            # result = np.ones((background_shape[0], background_shape[1], 3), dtype=np.float32)
            a = np.squeeze(bk_area[:, :, -1]) / 255
            print(a.shape)
            ks2[:, :, 0] = a[:, :]
            ks2[:, :, 1] = a[:, :]
            ks2[:, :, 2] = a[:, :]

            result = background * (ks - ks2) + bk_area[:, :, :-1] * ks2

            tamper_image.append(result)
            tamper_mask.append(bk_mask)
        # 调用save_method保存

        for index, item in enumerate(tamper_image):

            mask = tamper_mask[index]
            print('the shape of mask is :', mask.shape)
            selem = np.ones((3, 3))
            dst_8 = dilation.binary_dilation(mask, selem=selem)
            dst_8 = np.where(dst_8 == True, 1, 0)
            difference_8 = dst_8 - mask

            difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
            difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
            double_edge_candidate = difference_8_dilation + tamper_mask[index]
            double_edge = np.where(double_edge_candidate == 2, 1, 0)
            ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
                tamper_mask[index] == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
            tamper_gt.append(ground_truth)

            try:
                mask = Image.fromarray(tamper_mask[index])
            except Exception as e:
                print('mask to Image error', e)
            # 在这里的时候，mask foreground background 尺寸都是一致的了，poisson融合时，offset置为0
            # foreground = Image.fromarray(bk_area)

            # background = Image.fromarray(oringal_background)
            # mask = bk_mask
            try:
                # 保存
                for index, t_img in enumerate(tamper_image):
                    t_img = Image.fromarray((np.uint8(t_img))).convert('RGB')
                    t_img.save(self.img_tamper_save_path)
                    # t_img.save(os.path.join(self.img_tamper_save_path,t_img.split('\\')[-1]))

                for index, t_gt in enumerate(tamper_gt):
                    t_img = Image.fromarray(np.uint8(t_gt)).convert('RGB')
                    t_img.save(self.img_gt_save_path)
                    pass
            except Exception:
                traceback.print_exc()
