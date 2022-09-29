#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
直接使用mask抠图
和原本的数据集制作方式没有什么区别
机器抠图是完美的，完全会按照mask给出的部分把图片取出来
但是人是不一样的，人的抠图会有一些的损失，而在PS中，会自动的把使用套索工具中跨越像素的部分填充为半透明

考虑把图像的边缘加一个透明度，模仿人的行为


可以生成带有边缘透明度的一个ｐｎｇ　图像
"""
# 参考博客
# https://blog.csdn.net/qq_29391809/article/details/106036745
import cv2
from PIL import Image
import numpy as np
import os


# def edge_demo(image):
#     blurred = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯降噪，适度
#     gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
#     # 求梯度
#     xgrd = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
#     ygrd = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
#
#     egde_output = cv2.Canny(xgrd, ygrd, 50, 150)  # 50低阈值，150高阈值
#     # egde_output = cv.Canny(gray,50,150)   #都可使用
#     #cv2.imshow('canny_edge', egde_output)
#     # 输出彩色图
#     dst = cv2.bitwise_and(image, image, mask=egde_output)
#     #cv2.imshow('color edge', dst)
#     #cv2.waitKey(0)
#     return egde_output


class UnsupportedFormat(Exception):
    def __init__(self, input_type):
        self.t = input_type

    def __str__(self):
        return "不支持'{}'模式的转换，请使用为图片地址(path)、PIL.Image(pil)或OpenCV(cv2)模式".format(self.t)


class MatteMatting():
    def __init__(self, original_graph, mask_graph, input_type='path'):
        """
        将输入的图片经过蒙版转化为透明图构造函数
        :param original_graph:输入的图片地址、PIL格式、CV2格式
        :param mask_graph:蒙版的图片地址、PIL格式、CV2格式
        :param input_type:输入的类型，有path：图片地址、pil：pil类型、cv2类型
        """
        if input_type == 'path':
            self.img1 = cv2.imread(original_graph)
            self.img2 = cv2.imread(mask_graph)
        elif input_type == 'pil':
            self.img1 = self.__image_to_opencv(original_graph)
            self.img2 = self.__image_to_opencv(mask_graph)
        elif input_type == 'cv2':
            self.img1 = original_graph
            self.img2 = mask_graph
        else:
            raise UnsupportedFormat(input_type)
        self.mask_edge = self.edge_demo(self.img2)  # 只有mask 的边缘为255  标记为白  最原始的img2   为白色标注

    def touming(self, image):
        # cv2 类型的透明度
        b_channel, g_channel, r_channel = cv2.split(image)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 全部为255

        col = self.img2.shape[0]
        row = self.img2.shape[1]
        for i in range(col):
            for j in range(row):

                if self.mask_edge[i][j] == 255:
                    alpha_channel[i][j] = 100
                elif self.img2[i][j][0] == 255:
                    alpha_channel[i][j] = 0
        # 调用模式
        img_BGRA = cv2.merge((r_channel, g_channel,b_channel, alpha_channel))

        # 直接使用模式
        # img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        return img_BGRA

    @staticmethod
    def __transparent_back(self, img):
        """
        Image类型图像
        :param img: 传入图片地址
        :return: 返回替换白色后的透明图
        """
        img = img.convert('RGBA')
        L, H = img.size
        color_0 = (255, 255, 255, 255)  # 要替换的颜色  把掩膜之外的全部变透明像素  alapha=255 全不透明    alpha=0全透明
        for h in range(H):
            for l in range(L):
                dot = (l, h)
                color_1 = img.getpixel(dot)

                if color_1 == color_0:
                    color_1 = color_1[:-1] + (0,)
                    img.putpixel(dot, color_1)
                elif self.mask_edge[h, l] == 255:  # 边缘像素
                    color_1 = color_1[:-1] + (100,)
                    img.putpixel(dot, color_1)

        return img

    def save_image(self, mask_flip=False):
        """
        用于保存透明图
        :param path: 保存位置
        :param mask_flip: 蒙版翻转，将蒙版的黑白颜色翻转;True翻转;False不使用翻转
        """
        if mask_flip:
            self.img2 = cv2.bitwise_not(self.img2)  # 黑白翻转

        image = cv2.add(self.img1, self.img2)
        # cv2.imshow("1",image)  # 掩膜之后的图像
        # cv2.imshow("2",self.img1) # 原图
        # cv2.imshow("3", self.img2) # mask
        # cv2.waitKey(0)
        # print(image.shape)
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV转换成PIL.Image格式
        img = self.touming(image)  # self.touming 返回的是cv2 格式

        # print(img.size)
        #  img.save(path)
        # img.show()
        return img

    @staticmethod
    def __image_to_opencv(image):
        """
        PIL.Image转换成OpenCV格式
        """
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return img

    def edge_demo(self, image):
        result= cv2.Canny(image,30,40)
        # cv2.imshow("result",result)
        cv2.waitKey(0)
        cv2.imwrite("../pic/edge.png", result)
        return result


if __name__ == "__main__":
    path = r"C:\Users\brighten\Desktop\splicing\2"
    img_item = r"COD10K-CAM-1-Aquatic-2-ClownFish-11.jpg"
    mask_item = r"COD10K-CAM-1-Aquatic-2-ClownFish-11.png"
    img_file = os.path.join(path, img_item)
    mask_file = os.path.join(path, mask_item)
    save_path = os.path.join(path, "output.png")
    mask = cv2.imread(mask_file)

    mm = MatteMatting(img_file, mask_file)
    img = mm.save_image(mask_flip=True)  # mask_flip是指蒙版翻转，即把白色的变成黑色的，黑色的变成白色的
    print(img.shape)  # (600, 800, 4) # 3个通道的cv2格式
    cv2.imwrite(save_path, img)