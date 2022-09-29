from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision.transforms import transforms
import cv2 as cv
from PIL import Image
import torch
import os

'''
coverage和casia路径MixData里面改
'''



class TamperDataset(Dataset):
    def __init__(self,using_data=None):

        self.test_src_list,self.test_gt_list=MixData(using_data=using_data).gen_dataset()

    def __getitem__(self, index):
        src_path=self.test_src_list[index]
        gt_path=self.test_gt_list[index]
        img=Image.open(src_path).convert('RGB')
        gt=Image.open(gt_path).convert('L')
        if len(gt.split()) == 3:
            gt = gt.split()[0]
        elif len(gt.split()) == 1:
            pass
        if img.size!=gt.size:
            print(img.size,gt.size)
            sample={'if_not':True}
            return sample
        gt_band = self.__gen_band(gt)  # 一阶段条带
        gt_dou_edge = self.__to_dou_edge(gt)  # 二阶段边缘
        img=transforms.ToTensor()(img)
        gt_band=transforms.ToTensor()(gt_band)
        gt_dou_edge=transforms.ToTensor()(gt_dou_edge)

        sample = {'tamper_image': img, 'gt_band': gt_band, 'gt_double_edge': gt_dou_edge,'if_not':False,
                  'path': {'src': src_path, 'gt': gt_path}}
        return sample


    def __len__(self):
        length=len(self.test_src_list)
        return length

    def __gen_band(self,gt,dilate_window=5):
        _gt=gt.copy()
        if len(_gt.split())==3:
            _gt=_gt.split()[0]
        else:
            pass
        _gt=np.array(_gt,dtype='uint8')
        if max(_gt.reshape(-1))==255:
            _gt=np.where((_gt==255)|(_gt==100),1,0)
            _gt=np.array(_gt,dtype='uint8')
        else:
            pass
        _gt=cv.merge([_gt])#对拆封的通道合并
        kernel=np.ones((dilate_window,dilate_window),np.uint8)
        _band=cv.dilate(_gt,kernel)
        _band=np.array(_band,dtype='uint8')
        _band=np.where(_band==1,255,0)
        _band=Image.fromarray(np.array(_band,dtype='uint8'))
        if len(_band.split())==3:
            _band=np.array(_band)[:,:,0]
        else:
            _band=np.array(_band)
        return _band

    def __to_dou_edge(self,dou_em):
        #转化100为255为边缘
        _dou_em = dou_em.copy()
        if len(_dou_em.split()) == 3:
            _dou_em = _dou_em.split()[0]
        else:
            pass
        _dou_em=np.array(_dou_em)
        _dou_em=np.where(_dou_em==100,255,_dou_em)
        _dou_em=np.where(_dou_em==50,0,_dou_em)
        _dou_em = Image.fromarray(np.array(_dou_em, dtype='uint8'))
        if len(_dou_em.split()) == 3:
            _band = np.array(_dou_em)[:, :, 0]
        else:
            _dou_em= np.array(_dou_em)
        return _dou_em

class MixData:
    def __init__(self,using_data=None):
        #本机地址
        if using_data['coverage']:
            self.src_dir = r'C:\Users\97493\Desktop\调试代码\训练模拟数据集\public_dataset\coverage\src'
            self.gt_dir = r'C:\Users\97493\Desktop\调试代码\训练模拟数据集\public_dataset\coverage\gt'
        elif using_data['casia']:
            self.src_dir = r'C:\Users\97493\Desktop\调试代码\训练模拟数据集\public_dataset\casia\src'
            self.gt_dir = r'C:\Users\97493\Desktop\调试代码\训练模拟数据集\public_dataset\casia\gt'

        #在其他机器上
        # if using_data['casia']:
        #     self.src_dir = r'C:\Users\97493\Desktop\调试代码\训练模拟数据集\public_dataset\coverage\src'
        #     self.gt_dir = r'C:\Users\97493\Desktop\调试代码\训练模拟数据集\public_dataset\coverage\gt'
        # elif using_data['coverage']:
        #     self.src_dir = r'C:\Users\97493\Desktop\调试代码\训练模拟数据集\public_dataset\casia\gt'
        #     self.gt_dir = r'C:\Users\97493\Desktop\调试代码\训练模拟数据集\public_dataset\casia\gt'

    def gen_dataset(self):
        src_list=[]
        gt_list=[]
        for index,item in enumerate(os.listdir(self.src_dir)):
            img_src_path=os.path.join(self.src_dir,item)
            '''
            在这里更改src和gt的不同地址
            '''
            if 'coverage' in self.src_dir:
                _item=item.replace('t','forged')   #coverage
            elif 'casia' in self.src_dir:
                _item=item.split('.')[0]+'_gt'+'.png'
            img_gt_path=os.path.join(self.gt_dir,_item)
            if os.path.exists(img_gt_path):
                src_list.append(img_src_path)
                gt_list.append(img_gt_path)
            else:
                print(img_src_path,'未匹配')
        return src_list,gt_list


