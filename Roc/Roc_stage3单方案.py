import matplotlib.pyplot as plt
import argparse
import os,sys
sys.path.append('./')
from model.unet_two_stage_model_0719_new8map import UNetStage1 as Net1      #stage1/stage1.5都是一样的
from model.unet_two_stage_model_0719_new8map import UNetStage2 as Net2_1    #我们的stage2网络
from model.two_stage_model import UNetStage2 as Net2_2   #chr原来的stage2网络
from model.one_unet_v3 import UNetStage as Net
from model.one_unet_v3 import Cat_Unet as Cat
from functions import *
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from Roc.dataloader_random import TamperDataset
from sklearn.metrics import explained_variance_score


parser=argparse.ArgumentParser(description='Roc绘制')
parser.add_argument('--resume_1',default=[
    '',
    '',
    ''],help='')
parser.add_argument('--stage3',default='1',help='1代表并联方案stage3不加原图，2代表并联方案stage3加原图或梯度图，3代表并联方案stage3加原图加特征')
args, unknown = parser.parse_known_args()

def test(dataParser,item):

    save_dir = '/home/liu/gzf'  # Roc图保存地址
    name = '_什么什么方案'  # 什么什么方案 修改Roc图名称

    fpr_1,tpr_1,auc_1=one_work_1(dataParser)
    plt.plot(fpr_1,tpr_1, 'b', label='the lastest method, AUC=%0.2f'%auc_1 )
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('ROC for '+str(item))
    # plt.show()
    save_name='Roc_'+item+name+'.png'
    plt.savefig(os.path.join(save_dir,save_name))


@torch.no_grad()
def one_work_1(dataParser):
    model1 = Net()
    model2 = Net()
    model3 = Cat()
    test_pred=[]
    test_label=[]
    length=len(dataParser)
    if torch.cuda.is_available():
        model1.cuda()
        if os.path.isfile(args.resume_1[0]) and os.path.isfile(args.resume_2[0]) and os.path.isfile(args.resume_3[0]):
            checkpoint1 = torch.load(args.resume_1[0])
            model1.load_state_dict(checkpoint1['state_dict'])

            checkpoint2 = torch.load(args.resume_2[0])
            model2.load_state_dict(checkpoint2['state_dict'])

            checkpoint3 = torch.load(args.resume_3[0])
            model3.load_state_dict(checkpoint3['state_dict'])
            print("=> 已成功加载checkpoint")
        else:
            print("=> !!!!!!! checkpoint found at '{}'".format(args.resume))
    else:
        model1.cpu()
        if os.path.isfile(args.resume_1[0]) and os.path.isfile(args.resume_2[0]) and os.path.isfile(args.resume_3[0]):
            checkpoint1 = torch.load(args.resume_1[0],map_location='cpu')
            model1.load_state_dict(checkpoint1['state_dict'])

            checkpoint2 = torch.load(args.resume_2[0],map_location='cpu')
            model2.load_state_dict(checkpoint2['state_dict'])

            checkpoint3 = torch.load(args.resume_3[0],map_location='cpu')
            model3.load_state_dict(checkpoint3['state_dict'])
            print("=> 已成功加载checkpoint")
        else:
            print("=> !!!!!!! checkpoint found at '{}'".format(args.resume))
    model1.eval()
    model2.eval()
    model3.eval()
    for index,item in enumerate(dataParser):
        if item['if_not']:
            continue
        else:
            pass
        print('[{}/{}]'.format(index,length))
        if torch.cuda.is_available():
            images=item['tamper_image'].cuda()
            labels_band=item['gt_band'].cuda()
            labels_dou_edge=item['gt_double_edge'].cuda()
        else:
            images = item['tamper_image']
            labels_band = item['gt_band']
            labels_dou_edge = item['gt_double_edge']

        #不加原图
        if args.stage3==1:
            one_stage_outputs = model1(images)
            two_stage_outputs = model2(images)
            three_stage_outputs = model3(one_stage_outputs[0], two_stage_outputs[0])
        elif args.stage3==2:
            one_stage_outputs = model1(images)
            two_stage_outputs = model2(images)
            three_stage_outputs = model3(one_stage_outputs[0], two_stage_outputs[0], images)
        elif args.stage3 == 2:
            one_stage_outputs = model1(images)
            two_stage_outputs = model2(images)
            three_stage_outputs = model3(one_stage_outputs[0], two_stage_outputs[0],
                                         one_stage_outputs[1], one_stage_outputs[2], one_stage_outputs[3],
                                         two_stage_outputs[1], two_stage_outputs[2], two_stage_outputs[3])

        y = three_stage_outputs[0].reshape(-1)
        l = labels_dou_edge.reshape(-1)
        y = np.array(y.cpu().detach())
        # y = np.where(y > 0.5, 1, 0).astype('int')
        l = np.array(l.cpu().detach()).astype('int')
        y=y.tolist()
        l=l.tolist()
        test_pred.extend(y)  #注意append和extend区别
        test_label.extend(l)
        if index==2000:
            break

    _fpr, _tpr, _the = metrics.roc_curve(test_label,test_pred,pos_label=1)
    auc=metrics.auc(_fpr,_tpr)
    torch.cuda.empty_cache()
    return _fpr,_tpr,auc


if __name__=='__main__':
    using_data = {
        'coverage': False,
        'casia': False,
        }
    for index,item in enumerate(using_data):
        using_data[item]=True
        testData = TamperDataset(using_data=using_data)
        testDataLoader=torch.utils.data.DataLoader(testData,batch_size=1,num_workers=0)
        test_avg=test(dataParser=testDataLoader,item=item)
        using_data[item] = False
        print(item+'已完成')