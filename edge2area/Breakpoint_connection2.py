# -*- coding: utf-8 -*-
"""找断点和连接断点
此方法只采用细化算法提取骨架，
然后判断断点进行连接，因为很多断点在边缘位置，
所以加入边缘断点的补全"""
# 这是步骤2！！！
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.draw import line
import cv2 as cv
import Connect_area
from calculate_distance import min_dist2
from calculate_distance import too_close_point

'''发现自己判断断点算法的漏洞，比如说一个2*2区域的有两个连接，其他三个区域并无链接不会判定为断点 (已解决)
'''


def pad_0(pred_mask):
    # 填充一圈0用于判断断点是否在边缘，在判断完之后消除填充痕迹 图像周围一圈0
    _pred_mask = np.pad(pred_mask, (1, 1))
    return _pred_mask


def breakpoint_connect(mask_skeleton):  # 输入的是细化后的图像 result1结果
    # 事先计算连通域
    connect_domain = Connect_area.Seed_Filling(mask_skeleton*255, Connect_area.NEIGHBOR_HOODS_8)
    _, connect_points = Connect_area.reorganize(connect_domain)
    # print(connect_points)
    cv.imwrite("./pictures/cn.png", connect_domain * 50)

    _row, _col = mask_skeleton.shape
    # print("shape11")
    # print(mask_skeleton.shape)
    # 去除边界点函数
    for i in range(_row):  # 将图像边界上的标注点全部消去 原因在于膨胀腐蚀操作导致了错误标记 这个在允许的误差之内
        for j in range(_col):
            if i == 0 or j == 0 or i == _row - 1 or j == _col - 1:
                mask_skeleton[i][j] = 0

    # mask_skeleton = pad_0(mask_skeleton) # 删除周围填充0的操作 此时图像的大小不便
    row, col = mask_skeleton.shape  # 与row  col相等

    mask_skeleton = torch.Tensor(mask_skeleton).unsqueeze(0).unsqueeze(0)
    kernel = [[1, 1, 1],
              [1, 0, 1],
              [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    _mask = F.conv2d(mask_skeleton, kernel, stride=1, padding=1)
    # _mask = F.conv2d(mask_skeleton, kernel, stride=1, padding=0)
    _mask = _mask.squeeze(0).squeeze(0)
    mask_skeleton = mask_skeleton.squeeze(0).squeeze(0)
    mask_skeleton = np.array(mask_skeleton)
    # print("shpae22")
    # print(mask_skeleton.shape)
    # 解决2*2区域的bug，先要把周围的都清理掉才能
    # _mask==2且在mask_skeleton上
    mask_equal_2 = np.where((_mask == 2), 1, 0)
    # _mask表示8邻域之内的点的个数， mask_equal_2表示8邻域之内有2 个 点，mask_equal_2同样是一个mask图
    # mask_skeleton 表示标注点矩阵 1表示是边缘
    # mask_equal_2 表示8邻域内有2点
    mask_equal_2 = mask_equal_2 + mask_skeleton
    mask_equal_2 = np.where(mask_equal_2 == 2)  # 既是标注点又是邻域内有2点的点 mask_equal_2   的点标注为1
    mask_equal_2 = np.array(mask_equal_2)  # 返回索引 满足mask_equal_2的索引
    # (2,?)为行 列的坐标

    for i in range(len(mask_equal_2[0])):
        # 历遍所有的mask_equal_2 点集
        flag = 0
        row_1 = mask_equal_2[0][i]
        col_1 = mask_equal_2[1][i]
        '''
            000
            100
            000
        '''
        if mask_skeleton[row_1, col_1 - 1] == 1:
            if mask_skeleton[row_1 - 1, col_1 - 1] == 1 or mask_skeleton[row_1 + 1, col_1 - 1] == 1:
                flag = 1

        '''
              000
              000
              010
        '''
        if mask_skeleton[row_1 + 1, col_1] == 1:
            if mask_skeleton[row_1 + 1, col_1 - 1] == 1 or mask_skeleton[row_1 + 1, col_1 + 1] == 1:
                flag = 1

        '''
              000
              001
              000
        '''
        if mask_skeleton[row_1, col_1 + 1] == 1:
            if mask_skeleton[row_1 + 1, col_1 + 1] == 1 or mask_skeleton[row_1 - 1, col_1 + 1] == 1:
                flag = 1

        '''
              010
              000
              000
        '''
        if mask_skeleton[row_1 - 1, col_1] == 1:
            if mask_skeleton[row_1 - 1, col_1 + 1] == 1 or mask_skeleton[row_1 - 1, col_1 - 1] == 1:
                flag = 1

        if flag == 1:
            _mask[row_1][col_1] = 1
            # 要求两个点并排，不交叉，不斜角，形式为，只有此时flag==1
            # 110
            # 010
            # 000

    _mask = np.where(_mask == 1, 1, 0)  # 取1   此处包括了   邻域为1 和部分的邻域为2的点 全部置为1 _mask 物理意义变化
    # 卷积之后满足大于0，小于2的点可能是细化之后骨架周围的点，我们要取的是在骨架上的点，所以还会进行下面操作：

    _mask = _mask + mask_skeleton
    breakpoint = np.where(_mask == 2, 1, 0)  # 表示断点  物理意义就是 邻域为1 和部分的邻域为2的点
    '''圈出断点区域，对区域进行判断更为方便'''
    breakpoint_loc = np.where(breakpoint != 0)  # 下标，所有的breakpoint的横纵坐标
    breakpoint_loc = np.array(breakpoint_loc)
    # breakpoint矩形区域：[bbox[0]:bbox[1],bbox[2]:bbox[3]]
    if breakpoint_loc.shape[1] != 0:  # 存在没有断点的情况,需要return，否则会报错
        bbox = np.min(breakpoint_loc[0]), np.max(breakpoint_loc[0]), np.min(breakpoint_loc[1]), np.max(
            breakpoint_loc[1])
    else:
        return

    '''删'''
    '''可视化模块'''
    # mask_skeleton_1 = mask_skeleton + breakpoint  # 断点可视化展示，黄色的是断点，绿色的是骨架
    '''plt.figure("断点判断")
    plt.imshow(mask_skeleton_1)
    plt.show()'''

    '''在边界上有多个点（大于等于2），要考虑是否有误差点（判断误差点距离小于等于30）'''
    left = np.where(breakpoint_loc[1] == 1)
    left = np.array(left)  # left.size!=0
    if len(left[0]) >= 2:
        breakpoint_loc, breakpoint = too_close_point(left, breakpoint_loc, breakpoint, mask_skeleton)
    right = np.where(breakpoint_loc[1] == col - 2)
    right = np.array(right)
    if len(right[0]) >= 2:
        breakpoint_loc, breakpoint = too_close_point(right, breakpoint_loc, breakpoint, mask_skeleton)
    up = np.where(breakpoint_loc[0] == 1)
    up = np.array(up)
    if len(up[0]) >= 2:
        breakpoint_loc, breakpoint = too_close_point(up, breakpoint_loc, breakpoint, mask_skeleton)
    down = np.where(breakpoint_loc[0] == row - 2)
    down = np.array(down)
    if len(down[0]) >= 2:
        breakpoint_loc, breakpoint = too_close_point(down, breakpoint_loc, breakpoint, mask_skeleton)

    '''删'''
    '''可视化模块'''
    # mask_skeleton_2 = mask_skeleton + breakpoint  # 断点可视化展示，黄色的是断点，绿色的是骨架
    '''plt.figure("断点判断")
    plt.imshow(mask_skeleton_2)
    plt.show()'''

    # 左边界存在断点，经过pad的图边界是1到row-1或col-1，所以判断条件是1
    # breakpoint矩形区域：[bbox[0]:bbox[1],bbox[2]:bbox[3]]
    if bbox[2] == 1:  # 左边界有断点
        '''左边界上有断点分以下几种情况：1.左边界有两个断点；2.左边界有一个断点，上边界有一个断点；3.左边界有一个断点，下边界有一个断点；
           4.左边界一个断点，右边界一个断点；5.左边界一个断点，无其他断点在边界上；6.左边界断点数大于二'''
        left = np.where(breakpoint_loc[1] == 1)  # 左边界的断点的下标
        # 返回的是下标索引，所以np.where()返回的是元组类型，并不是numpy，还需要转换成numpy格式才能取数
        left = np.array(left)
        if len(left[0]) == 2:  # 加限制条件大于20像素
            a = left[0, 0]
            b = left[0, 1]
            mask_skeleton[breakpoint_loc[0, a]:breakpoint_loc[0, b], 1] = 1  # 连接两个断点 ，从断点列表中delete这里两个断点
            # 从breakpoint_loc中移除两点
            delete_list = [a, b]
            breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
        # breakpoint矩形区域：[bbox[0]: bbox[1], bbox[2]: bbox[3]]
        elif len(left[0]) == 1:  # 左边界有1个断点
            if bbox[0] == 1:  # 左边界
                up = np.where(breakpoint_loc[0] == 1)
                up = np.array(up)
                a = left[0, 0]
                b = up[0, 0]
                mask_skeleton[1:breakpoint_loc[0, a], 1 + 1] = 1
                mask_skeleton[1 + 1, 1:breakpoint_loc[1, b]] = 1
                delete_list = [a, b]
                breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
            elif bbox[1] == row - 2:  # 下边界
                down = np.where(breakpoint_loc[0] == row - 2)
                down = np.array(down)
                a = left[0, 0]
                b = down[0, 0]
                mask_skeleton[breakpoint_loc[0, a]:row - 2, 1] = 1
                mask_skeleton[row - 2 - 1, 1:breakpoint_loc[1, b]] = 1
                delete_list = [a, b]
                breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
            elif bbox[3] == col - 2:  # 左右边界都有断点情况
                ave = 0  # 判定是上半部还是下半部
                right = np.where(breakpoint_loc[1] == col - 2)
                right = np.array(right)
                a = left[0, 0]
                b = right[0, 0]
                ave = (breakpoint_loc[0, a] + breakpoint_loc[0, b]) / 2
                if ave < (row / 2):  # 在上下图形的上半部分
                    # print("生效1")
                    mask_skeleton[2, :] = 1  # 填充上边
                    mask_skeleton[1:breakpoint_loc[0, a], 1 + 1] = 1  # 填充左边
                    mask_skeleton[1:breakpoint_loc[0, b], col - 2 - 1] = 1  # 填充右边
                else:
                    # print("生效2")
                    mask_skeleton[row - 2 - 1, :] = 1  # 填充下边
                    mask_skeleton[breakpoint_loc[0, a]:row - 2, 1 + 1] = 1  # 填充左边
                    mask_skeleton[breakpoint_loc[0, b]:row - 2, col - 2 - 1] = 1  # 填充右边
            else:  # 左边只有一个断点，且没有其他断点在边界的情况，从断点集中舍弃该点
                pass
                '''delete_list = []
                delete_list.append(left[0, 0])
                breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)'''
    '''print("break")
    print(breakpoint_loc)'''
    '''
    [[338 342 343 345 350 353]
    [219 231 239 248 255 259]]
    '''
    if bbox[3] == col - 2:
        '''右边同样有以下几种情况'''
        right = np.where(breakpoint_loc[1] == col - 2)
        # 返回的是下标索引，所以np.where()返回的是元组类型，并不是numpy，还需要转换成numpy格式才能取数
        right = np.array(right)
        if len(right[0]) == 2:
            a = right[0, 0]
            b = right[0, 1]
            mask_skeleton[breakpoint_loc[0, a]:breakpoint_loc[0, b], col - 2 - 1] = 1
            # 从breakpoint_loc中移除两点
            delete_list = [a, b]
            breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
        elif len(right[0]) == 1:
            up = np.where(breakpoint_loc[0] == 1)
            up = np.array(up)
            down = np.where(breakpoint_loc[0] == row - 2)
            down = np.array(down)
            if up.size > 0:
                a = right[0, 0]
                b = up[0, 0]
                mask_skeleton[1:breakpoint_loc[0, a], col - 2 - 1] = 1
                mask_skeleton[1 + 1, breakpoint_loc[1, b]:col - 2] = 1
                delete_list = [a, b]
                breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
            elif down.size > 0:
                a = right[0, 0]
                b = down[0, 0]
                mask_skeleton[breakpoint_loc[0, a]:row - 2, col - 2 - 1] = 1
                mask_skeleton[row - 2 - 1, breakpoint_loc[1, b]:col - 2] = 1
                delete_list = [a, b]
                breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
            else:  # 右边只有一个断点，且没有其他断点在边界的情况，从断点集中舍弃该点，有待考虑
                pass
                '''delete_list = []
                delete_list.append(right[0, 0])
                breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)'''
    '''
    print("right")
    print(right)
    print(right[0, 0])#[[0]]
    '''
    '''上下边界都有断点情况：'''

    # 图像有意义的像素坐标范围 [1:row-2][1:col-2]  因为周围的一圈像素全部被替换为0
    up = np.where(breakpoint_loc[0] == 1)
    up = np.array(up)
    down = np.where(breakpoint_loc[0] == row - 2)
    down = np.array(down)
    if up.size == 1 and down.size == 1:  # 连接到最左或最右
        a = up[0, 0]
        b = down[0, 0]
        ave = (breakpoint_loc[1, a] + breakpoint_loc[1, b]) / 2
        if ave < (col / 2):  # 在上下图形的上半部分
            mask_skeleton[:, 1 + 1] = 1
            mask_skeleton[1 + 1, 1:breakpoint_loc[1, a]] = 1
            mask_skeleton[row - 2 - 1, 1:breakpoint_loc[1, a]] = 1
        else:
            mask_skeleton[:, col - 2 - 1] = 1
            mask_skeleton[1 + 1, breakpoint_loc[1, a]:col - 2 - 1] = 1
            mask_skeleton[row - 2 - 1, breakpoint_loc[1, b]:col - 2 - 1] = 1
        delete_list = [a, b]
        breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
    up = np.where(breakpoint_loc[0] == 1)
    up = np.array(up)
    down = np.where(breakpoint_loc[0] == row - 2)
    down = np.array(down)
    if up.size == 2:  # 上方连接
        # print("test 生效上方")
        a = up[0, 0]
        b = up[0, 1]
        mask_skeleton[1 + 1, breakpoint_loc[1, a]:breakpoint_loc[1, b]] = 1
        delete_list = [a, b]
        breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
    up = np.where(breakpoint_loc[0] == 1)
    up = np.array(up)
    down = np.where(breakpoint_loc[0] == row - 2)
    down = np.array(down)
    if down.size == 2:  # 下方连线
        # print("test  生效下方")
        a = down[0, 0]
        b = down[0, 1]
        mask_skeleton[row - 2 - 1, breakpoint_loc[1, a]:breakpoint_loc[1, b]] = 1
        delete_list = [a, b]
        breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
    # 希望每次调用calculate_distance函数，返回最小距离坐标（两点），
    # 在此函数中将两点连接，之后从breakpoint_loc中移除这两点，再次调用calculate_distance，
    # 直到breakpoint_loc只有一个值(或没有值)
    looptime = 0
    while True:  # 找出最近邻断点 连线 删除断点
        if len(breakpoint_loc[0]) <= 1:
            break
        elif looptime > 10:
            break
        looptime += 1
        a, b = min_dist3(breakpoint_loc, connect_domain)  # 在这个点集中距离最近的两点，提取为a,b，是下标
        # min_index=min_dist(breakpoint_loc) #按顺序
        # skimage.draw中的line方法，使用规则见包里解析
        if a != 99 and b != 99:
            start, end = line(breakpoint_loc[0][a], breakpoint_loc[1][a], breakpoint_loc[0][b], breakpoint_loc[1][b])
            # line 函数划线  断点连线方法
            # line方法返回的是点集[1,2,3,4],[1,2,3,4] 两个 4元的list
            mask_skeleton[start, end] = 2
            delete_list = [a, b]
            breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
    return mask_skeleton


def min_dist3(breakpoint_loc, connect_domain):
    # 每次取最小距离的两个点，返回这两个点的序列
    min_dist = 6600  # 阈值，由coverage预测mask最大图片对角线求得
    a = 99
    b = 99
    # 每次调用该函数前都会判断len(breakpoint_loc)是否小于等于1
    for i in range(0, len(breakpoint_loc[0])):
        for j in range(0, len(breakpoint_loc[0])):
            # print(connect_domain[breakpoint_loc[0][i], breakpoint_loc[1][i]])
            # print(connect_domain[breakpoint_loc[0][j], breakpoint_loc[1][j]])
            if connect_domain[breakpoint_loc[0][i], breakpoint_loc[1][i]] != connect_domain[
                breakpoint_loc[0][j], breakpoint_loc[1][j]]:
                dist = math.hypot(breakpoint_loc[0][i] - breakpoint_loc[0][j],
                                  breakpoint_loc[1][i] - breakpoint_loc[1][j])
                if (dist != 0) and (min_dist >= dist):
                    min_dist = dist
                    a = i
                    b = j
                else:
                    pass
            # else:
            #     continue
    return a, b


def connecting(input, output):  # 原本的main  函数化
    src_path = input
    save_path = output
    for index, item in enumerate(os.listdir(src_path)):
        _src_path = os.path.join(src_path, item)
        _save_path = os.path.join(save_path, item)
        pred_mask = Image.open(_src_path)
        if pred_mask.split() == 3:  # 若为3rgb图像 则会读取为灰度
            pred_mask = pred_mask.split()[0]
        pred_mask = np.array(pred_mask)
        pred_mask = np.where(pred_mask > 100, 1, 0)
        try:  # 有断点的情况
            _mask_after_connect = breakpoint_connect(pred_mask)  # 其实输入的就是经过细化之后的结果
            _mask_after_connect = np.where(_mask_after_connect, 255, 0)
            _mask_after_connect = Image.fromarray(_mask_after_connect.astype(np.uint8))
            _mask_after_connect.save(_save_path)
            print("{}/{}".format(index + 1, len(os.listdir(src_path))))
            print(item)
        except IndexError:  # 无断点会报IndexError错
            print("无断点")
            _mask_after_connect = pred_mask  # mask_after_connect为原图
            _mask_after_connect = np.where(_mask_after_connect, 255, 0)
            _mask_after_connect = Image.fromarray(_mask_after_connect.astype(np.uint8))
            _mask_after_connect.save(_save_path)
            print("{}/{}".format(index + 1, len(os.listdir(src_path))))
            print(item)
        else:
            pass


if __name__ == '__main__':
    # 文件夹内所有图像
    src_path = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result1'
    save_path = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result2'
    connecting(src_path, save_path)
