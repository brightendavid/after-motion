#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
文件夹里面少放点图   小电脑吃不消
处理的速度很慢，但是至少比人力要快不少,快几百倍的感觉
最终的结果在  save_png 和    doule_gt 中

消去某些测试失败的，或者说做一个脚本把重合度过高的删除   delete_bugs(tempered_dir, last_gt) 语句
在所有的图片生成完毕之后做删除明显错误样本

所有的 魔棒使用不连续属性的魔棒    真实的图片  不连续部分分布广泛

可以选取所有像素相似的不连续部分

"""
import os
import sys

from delete_pic import delete_all, move_files
from PS_actions import copy_move
from add_layer import add_layers
from get_gt import save_as_gt, gen_double_edge
from psd2png import pas_to_png

if __name__ == "__main__":
    is_move = True
    is_only_one=True
    # 测试时，选择is_move=False
    # 实际使用只需要给出下面两个目录即可
    # gt_yuan_dir = r"C:\Users\brighten\Desktop\GT_Object"  # 导入的原
    # src_yuan_dir = r"C:\Users\brighten\Desktop\Image"
    src_yuan_dir = r"C:\Users\brighten\Desktop\test_cod\Image"
    gt_yuan_dir = r"C:\Users\brighten\Desktop\test_cod\GT_Object"
    if os.path.exists(gt_yuan_dir) and os.path.exists(src_yuan_dir) or is_move == False:
        pass
    else:
        print("给出输入的实例检测  src 和 mask  源数据集!!!  必须存在")
        sys.exit()
    # 下面的6个目录随便给
    # 调用ps尽量给绝对路径，ps无法识别相对路径  使用的目录汇总
    dirs = {
        "src_dir": r"C:\Users\brighten\Desktop\ceshi\src",
        "mask_dir": r"C:\Users\brighten\Desktop\ceshi\mask",
        "psd_dir": r"C:\Users\brighten\Desktop\ceshi\save",  # psd 保存路径
        "tempered_dir": r'C:\Users\brighten\Desktop\ceshi\save_png',  # 保存的篡改图片  rgb图目录  输出1！！
        "out_dir": r'C:\Users\brighten\Desktop\ceshi\save_gt',  # 此处是  篡改部分全透明    黑色部分为未篡改部分
        "last_gt": r'C:\Users\brighten\Desktop\ceshi\double_gt',  # 双边缘   # 保存的双边缘gt 是灰度图  输出2！！
    }
    delete_dir = dirs.copy()
    delete_dir.pop("tempered_dir")  # 这两个不能删除
    delete_dir.pop("last_gt")

    org_img_folder = dirs["psd_dir"] + "\\"

    # 检查文件夹合法性
    for dir_key, dir_name in dirs.items():
        if os.path.exists(dir_name):
            print('{} 输出文件夹：{}  已经存在，继续执行'.format(dir_key, dir_name))
        else:
            os.mkdir(dir_name)
            print('{} 输出文件夹：{}  创建成功'.format(dir_key, dir_name))

    num = 75
    # 每次处理的图片数量 在使用了优化脚本的现在，num无论设置多少都没有问题，现在是一次只打开一个psd文件的任务
    # 现在已经不需要中间文件夹，可以直接从源文件夹中读取所有图片，一次处理完成

    # 从COD10K数据集中移入图片
    for i in range(25):
        if is_move:
            move_files(gt_yuan_dir, dirs["mask_dir"], num)
            move_files(src_yuan_dir, dirs["src_dir"], num)

        src_list = os.listdir(dirs["src_dir"])
        for src_name in src_list:
            mask_name = src_name.replace("jpg", "png")
            src = os.path.join(dirs["src_dir"], src_name)
            mask = os.path.join(dirs["mask_dir"], mask_name)
            # src = r"C:\Users\brighten\Desktop\COD10K高清\4\COD10K-CAM-2-Terrestrial-45-Spider-2513.jpg"
            # mask = r"C:\Users\brighten\Desktop\COD10K高清\4\COD10K-CAM-2-Terrestrial-45-Spider-2513_mask.png"
            try:
                add_layers(src, mask, dirs["psd_dir"])
                # 输入的是图片的路径   加图层     基本没有错误  出现错误是图片本身的问题  跳过  IDAT  block 错误 几百张图片里面会有一张出错
            except Exception as e:
                continue
            # org_img_folder = r'C:\Users\brighten\Desktop\test\src\\'

        # 操作此目录下 的所有psd 文件
        copy_move(org_img_folder)  # 这个东西很容易出问题  保存为PSD 格式
        # 输入psd的保存路径  png 的输出路径   psd转png  结果在  psd_dir
        pas_to_png(dirs["psd_dir"], dirs["tempered_dir"])  # 从PSD 到png  输出的是  tempered picture  !!! 一定要在save_gt之前

        save_as_gt(org_img_folder)  # 输出  mask   只有黑   白部分是透明色  和上面的用的同一个PSD文件
        # 输入psd的保存路径  png 的输出路径   psd转png  结果在out_dir
        pas_to_png(dirs["psd_dir"], dirs["out_dir"])  # 此处是  篡改部分全透明    黑色部分为未篡改部分  还是这个函数 但是输入的PSD 变化过了

        gen_double_edge(dirs["out_dir"], dirs["last_gt"])  # 由普通篡改mask到双边缘mask
        if is_only_one:
            break
        if is_move:
            pass
            # x = input("是否删除所有源文件，保留结果文件(y/n)?")
            # delete_all(delete_dir)  # 若不是本机运行，里面的目录名称改
        else:
            break
# 机械检测 运行delete_pic.py 中的
#     src_dir = r'C:\Users\brighten\Desktop\ceshi\save_png'
#     gt_dir = r'C:\Users\brighten\Desktop\ceshi\double_gt'
#     error_list2 = find_posible_errors(src_dir, gt_dir)
#     # # queshi_list=find_unpaired_pics(src_dir, gt_dir)
#     restart(error_list2)
#     delete_all(delete_dir)
