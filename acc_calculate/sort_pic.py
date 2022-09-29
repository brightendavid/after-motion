#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Created by:sjw
Time:2021.7.22
填写结果相关的文件夹目录
输出结果到指定文件夹下
用于结果展示时的图片定位方便
"""
from PIL import Image
import os


def all_picture_save(data, src_path, gt_path, pred_stage1, pred_stage2, pred_area, save_path):
    for index, item in enumerate(os.listdir(pred_stage1)):
        # pred_area  名称演示 output2_Tp_D_CNN_S_N_ani00087_ani00088_10102.tif
        # 把前面的5个item全部保存到save_path 中
        pic_name = item
        # 初始定义  防止报错
        open_src_path = ""
        open_area_path = ""
        open_gt_path = ""
        open_pred_stage1 = ""
        open_pred_stage2 = ""

        # 不同数据集的命名格式不一样
        if data == "casia":
            pic_name = pic_name.replace("output2_", "")
            # pic_name=pic_name[:-4]
            # open格式casia
            open_src_path = os.path.join(src_path, pic_name)
            open_area_path = os.path.join(pred_area, "output2_" + pic_name)
            open_gt_path = os.path.join(gt_path, pic_name[:-4] + "_gt.png")
            open_pred_stage1 = os.path.join(pred_stage1, "output1_" + pic_name)
            open_pred_stage2 = os.path.join(pred_stage2, "output2_" + pic_name)
        elif data == "columbia":
            pic_name = pic_name.replace("output_", "")  # canong3_canonxt_sub_01.tif
            # open格式columbia
            open_src_path = os.path.join(src_path, pic_name)  # canong3_02_sub_01.tif
            open_area_path = os.path.join(pred_area, "output_" + pic_name)  # output_canong3_canonxt_sub_01.tif
            open_gt_path = os.path.join(gt_path, pic_name[:-4] + "_edgemask.bmp")  # canong3_02_sub_01_edgemask.jpg
            open_pred_stage1 = os.path.join(pred_stage1, pic_name)  # canong3_canonxt_sub_01.tif  数量较少
            open_pred_stage2 = os.path.join(pred_stage2, pic_name)  # canong3_canonxt_sub_01.tif
        elif data == "coverage":
            pic_name = pic_name.replace("output2_", "")  # 1t.bmp
            # open格式columbia
            open_src_path = os.path.join(src_path, pic_name)  # 1t.bmp
            open_area_path = os.path.join(pred_area, "output2_" + pic_name)  # output2_1t.bmp
            open_gt_path = os.path.join(gt_path, pic_name.replace("t", "forged"))  # 1forged.bmp
            open_pred_stage1 = os.path.join(pred_stage1, "output1_" + pic_name)  # output1_1t.bmp
            open_pred_stage2 = os.path.join(pred_stage2, "output2_" + pic_name)  # output2_1t.bmp
        elif data == "cod":
            # open格式cod10k
            # open_area_path = os.path.join(pred_area, "output2_" + pic_name)  # output2_1t.bmp
            # output1_COD10K_tamper_4.png # 输入scr path
            pic_name = pic_name.replace("output1_", "")
            open_src_path = os.path.join(src_path, pic_name)  # COD10K_tamper_0.png
            open_gt_path = os.path.join(gt_path, pic_name.replace("tamper", "Gt"))  # COD10K_Gt_0.bmp
            open_gt_path = open_gt_path.replace("png", "bmp")
            open_pred_stage1 = os.path.join(pred_stage1, "output1_" + pic_name)  # output1_COD10K_tamper_4.png
            open_pred_stage2 = os.path.join(pred_stage2, "output2_" + pic_name)  # output2_COD10K_tamper_4.png
        elif data == "texture":
            # open格式cod10k
            # open_area_path = os.path.join(pred_area, "output2_" + pic_name)  # output2_1t.bmp
            # output1_COD10K_tamper_4.png # 输入scr path
            pic_name = pic_name.replace("output1_", "")
            open_src_path = os.path.join(src_path, pic_name)  # Bark(Gt_5754_103566_donut)(158_60_42_34_1).png
            open_gt_path = os.path.join(gt_path, pic_name.replace("tamper",
                                                                  "Gt"))  # Bark(Gt_5754_103566_donut)(158_60_42_34_1).bmp
            open_gt_path = open_gt_path.replace("png", "bmp")
            open_pred_stage1 = os.path.join(pred_stage1,
                                            "output1_" + pic_name)  # output1_Gt_2964_49673_toaster_Fabric_Fabric.jpg
            open_pred_stage2 = os.path.join(pred_stage2,
                                            "output2_" + pic_name)  # output2_Gt_2964_49673_toaster_Fabric_Fabric.jpg

        # 打开图片
        src = Image.open(open_src_path)
        if data == "casia" or data == "coverage" or data == "columbia":
            area = Image.open(open_area_path)
        gt = Image.open(open_gt_path)
        stage1 = Image.open(open_pred_stage1)
        stage2 = Image.open(open_pred_stage2)

        # save 格式
        save_src_path = os.path.join(save_path, pic_name + "_src.png")
        save_area_path = os.path.join(save_path, pic_name + "_area.png")
        save_gt_path = os.path.join(save_path, pic_name + "_gt.png")
        save_pred_stage1 = os.path.join(save_path, pic_name + "_stage1.png")
        save_pred_stage2 = os.path.join(save_path, pic_name + "_stage2.png")

        # save图片
        if data == "casia" or data == "coverage" or data == "columbia":
            area.save(save_area_path)
        gt.save(save_gt_path)
        src.save(save_src_path)
        stage1.save(save_pred_stage1)
        stage2.save(save_pred_stage2)

        print("{}/{}".format(index + 1, len(os.listdir(pred_area))))


if __name__ == '__main__':
    # 输入六元组  为图像篡改检测的所有六个要素  意为保存效果较好的一些图像和它们的配套图像  找图片的索引是  区域预测图像  因为这类图像  的数量最少

    # casia  读取目录
    # src_path = r'C:\Users\brighten\Desktop\public_dataset\casia\src'
    # gt_path = r"C:\Users\brighten\Desktop\public_dataset\casia\gt"  # 双边缘的gt
    # pred_stage1 = r"C:\Users\brighten\Desktop\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\casia_test\pred_train\stage1"
    # pred_stage2 = r"C:\Users\brighten\Desktop\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\casia_test\pred_train\stage2"
    # pred_area = r"C:\Users\brighten\Desktop\act_well\result_casia"  # 名称演示 output2_Tp_D_CNN_S_N_ani00087_ani00088_10102.tif
    # save_path = r'C:\Users\brighten\Desktop\show_result'

    # coverage  读取目录
    # src_path = r'C:\Users\brighten\Desktop\public_dataset\coverage\src'
    # gt_path = r"C:\Users\brighten\Desktop\public_dataset\coverage\gt"  # 双边缘的gt
    # pred_stage1 = r"C:\Users\brighten\Desktop\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\coverage_test\pred_train\stage1"
    # pred_stage2 = r"C:\Users\brighten\Desktop\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\coverage_test\pred_train\stage2"
    # pred_area = r"C:\Users\brighten\Desktop\act_well\result_coverage"  # 名称演示 output2_Tp_D_CNN_S_N_ani00087_ani00088_10102.tif
    # save_path = r'C:\Users\brighten\Desktop\show_result'

    # columbia 读取
    # src_path = r'C:\Users\brighten\Desktop\public_dataset\Columbia\src'
    # gt_path = r"C:\Users\brighten\Desktop\public_dataset\Columbia\gt"  # 双边缘的gt
    # pred_stage1 = r"C:\Users\brighten\Desktop\forgery-edge-detection-main\Mymodel\chr_std\Columbia\stage1"
    # pred_stage2 = r"C:\Users\brighten\Desktop\forgery-edge-detection-main\Mymodel\chr_std\Columbia\stage2"
    # pred_area = r"C:\Users\brighten\Desktop\act_well\result_Columbia"
    # save_path = r'C:\Users\brighten\Desktop\show_result'

    # # cod10k
    src_path = r'G:\3月最新数据\cod10k\train_src'  # COD10K_tamper_0.png
    gt_path = r"G:\3月最新数据\cod10k\train_gt"  # COD10K_Gt_0.bmp
    # output2_COD10K_tamper_4.png
    # output1_COD10K_tamper_4.png
    pred_stage1 = r"G:\实验结果\0324_两阶段_0306模型,只监督条带区域，带8张图\0324_两阶段_0306模型,只监督条带区域，带8张图\COD10K_test\pred_train\stage1"
    pred_stage2 = r"G:\实验结果\0324_两阶段_0306模型,只监督条带区域，带8张图\0324_两阶段_0306模型,只监督条带区域，带8张图\COD10K_test\pred_train\stage2"
    pred_area = r"C:\Users\brighten\Desktop\forgery-edge-detection-main\images"
    save_path = r'C:\Users\brighten\Desktop\show_result'

    # texture  # 命名格式混乱  无法定位
    # src_path = r'G:\3月最新数据\periodic_texture\divide\train_src'
    # gt_path = r"G:\3月最新数据\periodic_texture\divide\train_gt"
    # pred_stage1 = r"G:\实验结果\0324_两阶段_0306模型,只监督条带区域，带8张图\0324_两阶段_0306模型,只监督条带区域，带8张图\texture_sp_test\pred_train\stage1"
    # pred_stage2 = r"G:\实验结果\0324_两阶段_0306模型,只监督条带区域，带8张图\0324_两阶段_0306模型,只监督条带区域，带8张图\texture_sp_test\pred_train\stage2"
    # pred_area = r"C:\Users\brighten\Desktop\forgery-edge-detection-main\images"
    # save_path = r'C:\Users\brighten\Desktop\show_result'

    data = ""
    if "casia" in src_path:
        data = "casia"
    elif "Columbia" in src_path:
        data = "columbia"
    elif "coverage" in src_path:
        data = "coverage"
    elif "cod" in src_path:
        data = "cod"
    elif "texture" in src_path:
        data = "texture"
    else:
        import sys

        print("error!!非支持的数据集测试，只能支持三大公开数据集")
        sys.exit(1)

    print(data)
    save_path = os.path.join(save_path, data)
    all_picture_save(data, src_path, gt_path, pred_stage1, pred_stage2, pred_area, save_path)  # 文件夹内所有图片
