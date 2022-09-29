# 计算任意两点之间的距离，每次返回距离最小的两个点坐标,最近邻点
# 提供min_dist，min_dist2，too_close_point函数
import math
import numpy as np


def min_dist(breakpoint_loc):
    # breakpoint_loc是一个二维numpy矩阵，
    # 函数的具体功能是：每次找到第一列元素组成的坐标和距离最近的坐标
    # 返回最近坐标的index
    min_dist = 660  # 阈值，由coverage预测mask最大图片对角线求得
    min_index = 99
    # 每次调用该函数前都会判断len(breakpoint_loc)是否小于等于1
    for i in range(1, len(breakpoint_loc[0])):
        '''
        实例：
        import numpy as np
        import math
        p1=np.array([0,0])
        p2=np.array([1000,2000])
        p3=p2-p1
        p4=math.hypot(p3[0],p3[1])
        print(p4)
        '''
        dist = math.hypot(breakpoint_loc[0][0] - breakpoint_loc[0][i], breakpoint_loc[1][0] - breakpoint_loc[1][i])
        if min_dist > dist:
            min_dist = dist
            min_index = i
        else:
            pass
    return min_index


def min_dist2(breakpoint_loc):
    # 每次取最小距离的两个点，返回这两个点的序列
    min_dist = 6600  # 阈值，由coverage预测mask最大图片对角线求得
    a = 99
    b = 99
    # 每次调用该函数前都会判断len(breakpoint_loc)是否小于等于1
    for i in range(0, len(breakpoint_loc[0])):
        for j in range(0, len(breakpoint_loc[0])):
            dist = math.hypot(breakpoint_loc[0][i] - breakpoint_loc[0][j], breakpoint_loc[1][i] - breakpoint_loc[1][j])
            if (dist != 0) and (min_dist >= dist):
                min_dist = dist
                a = i
                b = j
            else:
                pass
    return a, b


def too_close_point(oriention, breakpoint_loc, breakpoint, mask_skeleton):
    """重写，不用距离来判断，用两点之间是否都为1"""
    for i in range(len(oriention[0])):
        for j in range(i + 1, len(oriention[0])):
            a = oriention[0, i]
            b = oriention[0, j]
            # 接下来判断mask_skeleton在mask[breakpoint[0,a],breakpoint[1,a]]到mask[breakpoint[0,b],breakpoint[1,b]]上的点都为1
            if mask_skeleton[breakpoint[0, a]:breakpoint[0, b], breakpoint[1, a]:breakpoint[1, b]]:
                breakpoint[breakpoint_loc[0, b], breakpoint_loc[1, b]] = 0
                delete_list = [a, b]
                breakpoint_loc = np.delete(breakpoint_loc, delete_list, axis=1)
            else:
                break

    return breakpoint_loc, breakpoint
