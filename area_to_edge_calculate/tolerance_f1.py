"""
@author :haoran
time:0316
tolerance 指标
"""
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from tqdm import tqdm
import skimage.morphology as dilation
import os
import pandas as pd


class ToleranceMetrics:
    def __init__(self):
        pass

    def __gen_band(self, gt, dilate_window=5):
        """
        :param gt: PIL type
        :param dilate_window:
        :return:
        """

        _gt = gt.copy()

        # input required
        if len(_gt.split()) == 3:
            _gt = _gt.split()[0]
        else:
            pass

        _gt = np.array(_gt, dtype='uint8')

        if max(_gt.reshape(-1)) == 255:
            _gt = np.where((_gt == 255) | (_gt == 100), 1, 0)
            _gt = np.array(_gt, dtype='uint8')
        else:
            pass

        _gt = cv.merge([_gt])
        kernel = np.ones((dilate_window, dilate_window), np.uint8)
        _band = cv.dilate(_gt, kernel)
        _band = np.array(_band, dtype='uint8')
        # _band = np.where(_band == 1, 255, 0)
        _band = Image.fromarray(np.array(_band, dtype='uint8'))
        if len(_band.split()) == 3:
            _band = np.array(_band)[:, :, 0]
        else:
            _band = np.array(_band)
        return _band

    def area2edge(self, orignal_mask):
        """
        :param orignal_mask: 输入的是 01 mask图
        :return: 255 100 50 mask 图
        """
        # print('We are in mask_to_outeedge function:')
        try:
            mask = orignal_mask
            # print('the shape of mask is :', mask.shape)
            selem = np.ones((3, 3))
            dst_8 = dilation.binary_dilation(mask, selem=selem)
            dst_8 = np.where(dst_8 == True, 1, 0)
            difference_8 = dst_8 - orignal_mask

            difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
            difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
            double_edge_candidate = difference_8_dilation + mask
            double_edge = np.where(double_edge_candidate == 2, 1, 0)
            ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
                mask == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
            ground_truth = np.where(ground_truth == 305, 255, ground_truth)
            ground_truth = np.array(ground_truth, dtype='uint8')
            return ground_truth

        except Exception as e:
            print(e)

    def pred2mask(self, pred):
        """
        将预测图转化为二值图
        :param pred: numpy
        :return:
        """
        if max(pred.reshape(-1)) > 1:
            pred = pred / 255
        else:
            pass
        pred = np.where(pred > 0.5, 1, 0)

        return pred

    def t0f1(self, pred_path, gt_path, edge_mode='area'):
        """
        :param pred:
        :param edge_mode: area or edge
        :return:
        """
        try:
            pred = Image.open(pred_path)
            gt = Image.open(gt_path)
            if len(pred.split()) == 3:
                pred = pred.split()[0]

            if len(gt.split()) == 3:
                gt = gt.split()[0]
            pred = np.array(pred)
            gt = np.array(gt)
        except Exception as e:
            print(e)
            return 'read img error'
        for i in range(2):
            if edge_mode == 'area':
                _ = self.pred2mask(pred)
                std_edge = self.area2edge(_)
                dou_pred = np.where((std_edge == 255) | (std_edge == 100), 1, 0)
            elif edge_mode == 'edge':
                # 二值化  使得预测概率较小的预测点消除  留下概率较大的边缘
                dou_pred = np.where(pred > 127, 1, 0)

            else:
                print('Input edge_mode error!')

            dou_gt = np.where((gt == 255) | (gt == 100), 1, 0)  # 255 和100 是篡改边缘和非篡改边缘

            # plt.subplot(121)
            # plt.imshow(dou_gt, cmap='gray')
            # plt.subplot(122)
            # plt.imshow(dou_pred, cmap='gray')
            # plt.show()
            # plt.figure('diff')
            # plt.imshow(dou_gt-dou_gt)
            # plt.show()

            try:
                # result = confusion_matrix(y_true=dou_gt.reshape(-1), y_pred=dou_pred.reshape(-1))
                # print(result)
                # TP = result[1][1]
                # FP = result[1][0]
                # FN = result[0][1]
                # TN = result[0][0]
                # my_precision = TP / (TP + FP)
                # my_recall = TP / (TP + FN)
                # print(my_recall)
                # plt.subplot(121)
                # plt.imshow(dou_pred, cmap='gray')
                # plt.subplot(122)
                # plt.imshow(dou_pred*255-dou_gt*255)
                # plt.show()

                f1 = f1_score(y_pred=dou_pred.reshape(-1), y_true=dou_gt.reshape(-1), zero_division=0)
                precision = precision_score(y_pred=dou_pred.reshape(-1), y_true=dou_gt.reshape(-1), zero_division=0)
                recall = recall_score(y_pred=dou_pred.reshape(-1), y_true=dou_gt.reshape(-1), zero_division=0)
                # f1 = -1
                # precision = -1
                # recall = -1
                acc = accuracy_score(y_pred=dou_pred.reshape(-1), y_true=dou_gt.reshape(-1))
            except Exception as e:
                _ = Image.fromarray(pred)
                _ = _.resize((gt.shape[1], gt.shape[0]))
                pred = np.array(_)
                print('Scale the image')
                continue

        return {'f1': f1, 'precision': precision, 'recall': recall, 'acc': acc}

    def tx_f1_precision_recall(self, pred_path, gt_path, edge_mode='area', tolerance=1):
        """
        :param pred:
        :param edge_mode: area or edge
        :return:
        """
        try:
            pred = Image.open(pred_path)
            gt = Image.open(gt_path)
            if len(pred.split()) == 3:
                pred = pred.split()[0]
            if len(gt.split()) == 3:
                gt = gt.split()[0]
            pred = np.array(pred)
            gt = np.array(gt)
        except Exception as e:
            print(e)
            return 'read img error'
        for i in range(2):
            if edge_mode == 'area':
                mask = self.pred2mask(pred)
                std_edge = self.area2edge(mask)
                dou_pred = np.where((std_edge == 255) | (std_edge == 100), 1, 0)
            elif edge_mode == 'edge':
                dou_pred = np.where(pred > 127, 1, 0)
            else:
                print('Input edge_mode error!')

            dou_gt = np.where((gt == 255) | (gt == 100), 1, 0)

            # plt.subplot(131)
            # plt.title('gt')
            # plt.imshow(dou_gt, cmap='gray')
            #
            # plt.subplot(132)
            # plt.title('pred_edge')
            # plt.imshow(dou_pred, cmap='gray')
            #
            # plt.subplot(133)
            # plt.title('pred')
            #
            # plt.imshow(pred,cmap='gray')
            # plt.show()
            try:
                result_or = confusion_matrix(y_true=dou_gt.reshape(-1), y_pred=dou_pred.reshape(-1))
                result_band = confusion_matrix(
                    y_true=self.__gen_band(Image.fromarray(dou_gt.astype('uint8')),
                                           dilate_window=tolerance + 2).reshape(-1),
                    y_pred=dou_pred.reshape(-1))
                TP_or = result_or[1][1]
                TP = result_band[1][1]
                FP_or = result_or[1][0]
                FN_or = result_or[0][1]
                # TP = result[1][1]
                # FP = result[1][0]
                # FN = result[0][1]
                # TN = result[0][0]
                t_precision = 0 if TP_or + FN_or == 0 else 1 if TP / (TP_or + FP_or) > 1 else TP / (TP_or + FP_or)
                t_recall = 0 if TP_or + FN_or == 0 else 1 if TP / (TP_or + FN_or) > 1 else TP / (TP_or + FN_or)
                # print('tolerance precision :', t_precision)
                # print('tolerance recall :', t_recall)
                t_f1 = 0 if (t_recall + t_precision) == 0 else (2 * t_recall * t_precision) / (t_recall + t_precision)
                # print('tolerance f1:', t_f1)
            except Exception as e:

                _ = Image.fromarray(pred)
                _ = _.resize((gt.shape[1], gt.shape[0]))
                pred = np.array(_)
                continue
        return {'f1': t_f1, 'precision': t_precision, 'recall': t_recall}


if __name__ == '__main__':
    edge_mode="edge" # "edge" # area 区域  ； edge条带
    # stage 1 不在计算范围之内
    # 若文件夹内存在stage1名称文件夹  则忽略
    # 使用其他算法测试时，将310行的edge_mode改为area  ，此时将会调用区域转边缘算法将预测  区域 转为 边缘
    # 使用本算法时候，调回edge_mode =edge
    # pred  list 所有算法的结果按照文件夹存放   此处是文件夹的文件夹  再下面一级才是图片文件
    # 新添加文件加入，
    # 输出是文件夹  输出是excel  不需要pth 权重文件
    pred = [r'C:\Users\brighten\Desktop\act_well\RRU',
            # 本算法浩然电脑coverage图片   0
            r"C:\Users\brighten\Desktop\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\casia_test\pred_train",
            # 本算法casia数据集图片  1
            r"C:\Users\brighten\Desktop\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\columb",
            # 本算法 columbia  （2）
            r"C:\Users\brighten\Desktop\指标计算casia",
            # 对比方法casia数据集图片  3
            r"C:\Users\brighten\Desktop\指标计算clumbia",
            # 对比方法columbia数据集图片4
            r"C:\Users\brighten\Desktop\act_well",
            r"C:\Users\brighten\Desktop\new_code_add_15\chr_std\casia",
            r"C:\Users\brighten\Desktop\new_code_add_15\chr_std\coverage",
            r"C:\Users\brighten\Desktop\cat",
            r"C:\Users\brighten\Desktop\new_code_add_15\chr_std\casia",
            r"C:\Users\brighten\Desktop\1",
            r"I:\重要文件\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\coverage_test\pred_train",
            r"I:\重要文件\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\casia_test\pred_train",
            r"I:\重要文件\0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束\columb\pred_train"
            # '/home/liu/chenhaoran/test/0323_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束_stress_blur/coverage_test/pred_train',
            # '/home/liu/chenhaoran/test/0323_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束_stress_blur/casia_test/pred_train'
            ]
    # 存放对应数据集的gt 是gt图片的父目录
    # 这样放就是有一点不好，容易把数字填错了
    # 三个公开数据集的gt文件夹
    gt = [r'C:\Users\brighten\Desktop\DATA\public_dataset\coverage\gt',  # 0
          r"C:\Users\brighten\Desktop\public_dataset\Columbia\Columbia\gt",  # 1
          r"C:\Users\brighten\Desktop\DATA\public_dataset\casia\gt",  # 2
          r"F:\DATA\public_dataset\coverage\gt",
          r"F:\DATA\public_dataset\casia\gt",
          r"F:\DATA\public_dataset\Columbia\gt"
          # '/home/liu/chenhaoran/Tamper_Data/3月最新数据/public_dataset/columbia/dou_gt500',
          # '/home/liu/chenhaoran/Tamper_Data/3月最新数据/public_dataset/coverage/gt'
          ]
    tm = ToleranceMetrics()
    pred_dir = os.path.join(pred[-1])
    gt_dir = os.path.join(gt[-1])
    blur_idx_list = os.listdir(pred_dir)
    gt_list = os.listdir(gt_dir)
    _ = []
    for i in blur_idx_list:
        if 'stage1' in i:
            pass
        else:
            _.append(i)
    blur_idx_list = _
    print('the test img file is :', blur_idx_list)
    count = 0
    f1_avg = []
    precision_avg = []
    recall_avg = []
    acc_avg = []
    error_list = []
    add_list = []

    # average_zhibiao[f1_avg]+=1
    # print(average_zhibiao[f1_avg])
    for blur_idx, blur_item in enumerate(tqdm(blur_idx_list)):
        add_name = []
        add_tolerance = []
        add_f1 = []
        add_precision = []
        add_acc = []
        add_recall = []
        pred_list = os.listdir(os.path.join(pred_dir, blur_item))
        # blur_factor = int(float(blur_item.split('_')[-1]))
        blur_factor = 0
        # if blur_factor in [20,19,18,17,16,15,14,13,0,4,6,7,11,12,19]:
        #     continue

        if blur_factor not in [0]:
            continue
        for idx, item in enumerate(tqdm(pred_list)):
            if idx > 200:
                # 为了节约时间，每个数据集/算法  只计算前100张图片 。认定具有和全部数据集几乎相同的指标代表性
                break
            _pred_path = os.path.join(os.path.join(pred_dir, blur_item), item)
            # _gt_path = os.path.join(gt_dir, item.split('.')[0] + '_edgemask_3' + '.jpg')  # columbia
            if "coverage" in gt_dir:
                _gt_path = os.path.join(gt_dir, item.replace('output2_', ''))  # coverage
                # _gt_path = os.path.join(gt_dir, item.replace('t', 'forged'))  # coverage
                _gt_path=_gt_path[:-5]+"forged.bmp"
            # _gt_path=_gt_path.replace("forgedif","bmp")
            elif "casia" in gt_dir:
                _gt_path = os.path.join(gt_dir, item.split('.')[0] + '_gt.png') # casia
                _gt_path = _gt_path.replace("output2_", "")
            elif "olumbia" in gt_dir :
                _gt_path = os.path.join(gt_dir, item.split('.')[0] + '_edgemask.bmp') # casia
                _gt_path = _gt_path.replace("output_", "")
            # 自己的算法在output后面加1或2 表示第一或者第2阶段
            if os.path.exists(_pred_path) and os.path.exists(_gt_path):
                pass
            else:
                print('No this file:', _pred_path, _gt_path)
                continue

            try:
                # 此处的edge_mode=area参数是其他区域检测方法使用
                # 此处的edge_mode=edge参数是本算法  边缘检测方法使用
                result = tm.t0f1(pred_path=_pred_path, gt_path=_gt_path, edge_mode=edge_mode)
                add_name.append(item)
                add_tolerance.append(0)
                add_f1.append(result['f1'])
                add_acc.append(result['acc'])
                add_precision.append(result['precision'])
                add_recall.append((result['recall']))
                #
                for t in range(1, 5):
                    result = tm.tx_f1_precision_recall(pred_path=_pred_path, gt_path=_gt_path, tolerance=t,
                                                       edge_mode=edge_mode)

                    add_name.append(item)
                    add_tolerance.append(t)
                    add_f1.append(result['f1'])

                    add_acc.append(-1)
                    add_precision.append(result['precision'])
                    add_recall.append((result['recall']))
            except Exception as e:
                print(e)
                error_list.append(item)
                continue

        data = {
            'name': add_name,
            'tolerance': add_tolerance,
            'f1': add_f1,
            'precision': add_precision,
            'recall': add_recall,
            'acc': add_acc
        }
        test = pd.DataFrame(data)
        test.to_excel(
            r'C:\Users\brighten\Desktop\fine_true%s.xlsx' % (
                blur_item))
        add_name.clear()
        add_tolerance.clear()
        add_f1.clear()
        add_precision.clear()
        add_acc.clear()
        add_recall.clear()

    print(error_list)
