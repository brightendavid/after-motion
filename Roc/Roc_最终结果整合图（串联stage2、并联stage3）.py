import matplotlib.pyplot as plt
import argparse
import os,sys
sys.path.append('./')
from model.unet_two_stage_model_0719_new8map import UNetStage1 as Net1      #stage1/stage1.5都是一样的
from model.unet_two_stage_model_0719_new8map import UNetStage2 as Net2_1    #我们的stage2网络
from model.two_stage_model import UNetStage2 as Net2_2   #chr原来的stage2网络
from functions import *
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from Roc.dataloader_random import TamperDataset
from sklearn.metrics import explained_variance_score

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'


parser=argparse.ArgumentParser(description='Roc绘制')
parser.add_argument('--resume_1',default=[
    r'D:\21.路径\串行最好结果（stage1.5前10层纹理triplet loss、stage2强调5倍stage1.5）\stage1.5_权重17_最优.pth',
    r'D:\21.路径\串行最好结果（stage1.5前10层纹理triplet loss、stage2强调5倍stage1.5）\stage2_weight5.pth'],help='最优模型地址（stage2改过）')
parser.add_argument('--resume_2',default=[
    r'D:\21.路径\浩然最好的结果\stage1.pth',
    r'D:\21.路径\浩然最好的结果\stage2.pth'],help='chr（stage2未改过）')
args, unknown = parser.parse_known_args()

def test(dataParser,item):

    save_dir='/home/liu/gzf'    #Roc图保存地址
    name='_什么什么方案'     #什么什么方案 修改Roc图名称

    fpr_1,tpr_1,auc_1=one_work_1(dataParser)
    fpr_2,tpr_2,auc_2=one_work_2(dataParser)

    plt.plot(fpr_1,tpr_1, 'b', label='最新方案, AUC=%0.2f'%auc_1 )
    plt.plot(fpr_2,tpr_2,'g', label='chr, AUC=%0.2f'%auc_2)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('ROC for '+str(item))
    plt.show()
    # save_name='Roc_'+item+name+'.png'
    # plt.savefig(os.path.join(save_dir,save_name))


@torch.no_grad()
def one_work_1(dataParser):
    model1 = Net1()
    model2 = Net2_1()
    test_pred=[]
    test_label=[]
    length=len(dataParser)
    if torch.cuda.is_available():
        model1.cuda()
        model2.cuda()
        if os.path.isfile(args.resume_1[0]) and os.path.isfile(args.resume_1[1]):
            checkpoint1 = torch.load(args.resume_1[0])
            checkpoint2 = torch.load(args.resume_1[1])
            model1.load_state_dict(checkpoint1['state_dict'])
            model2.load_state_dict(checkpoint2['state_dict'])
            print("=> 已成功加载checkpoint")
        else:
            print("=> !!!!!!! checkpoint found at '{}'".format(args.resume))
    else:
        model1.cpu()
        model2.cpu()
        if os.path.isfile(args.resume_1[0]) and os.path.isfile(args.resume_1[1]):
            checkpoint1 = torch.load(args.resume_1[0],map_location='cpu')
            checkpoint2 = torch.load(args.resume_1[1],map_location='cpu')
            model1.load_state_dict(checkpoint1['state_dict'])
            model2.load_state_dict(checkpoint2['state_dict'])
            print("=> 已成功加载checkpoint")
        else:
            print("=> !!!!!!! checkpoint found at '{}'".format(args.resume))
    model1.eval()
    model2.eval()
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

        one_stage_outputs_1 = model1(images)
        rgb_pred_rgb = torch.cat((one_stage_outputs_1[0], images), 1)
        two_stage_outputs_1 = model2(rgb_pred_rgb, one_stage_outputs_1[1], one_stage_outputs_1[2], one_stage_outputs_1[3])
        y = two_stage_outputs_1[0].reshape(-1)
        l = labels_dou_edge.reshape(-1)
        y = np.array(y.cpu().detach())
        # y = np.where(y > 0.5, 1, 0).astype('int')
        l = np.array(l.cpu().detach()).astype('int')
        y=y.tolist()
        l=l.tolist()
        test_pred.extend(y)
        test_label.extend(l)
        # if index==2:
        #     break

    _fpr, _tpr, _the = metrics.roc_curve(test_label,test_pred,pos_label=1)

    auc=metrics.auc(_fpr,_tpr)
    torch.cuda.empty_cache()
    return _fpr,_tpr,auc

@torch.no_grad()
def one_work_2(dataParser):
    model1 = Net1()
    model2 = Net2_2()
    test_pred = []
    test_label = []
    length=len(dataParser)
    if torch.cuda.is_available():
        model1.cuda()
        model2.cuda()
        if os.path.isfile(args.resume_2[0]) and os.path.isfile(args.resume_2[1]):
            checkpoint1 = torch.load(args.resume_2[0])
            checkpoint2 = torch.load(args.resume_2[1])
            model1.load_state_dict(checkpoint1['state_dict'])
            model2.load_state_dict(checkpoint2['state_dict'])
            print("=> 已成功加载checkpoint")
        else:
            print("=> !!!!!!! checkpoint found at '{}'".format(args.resume))
    else:
        model1.cpu()
        model2.cpu()
        if os.path.isfile(args.resume_2[0]) and os.path.isfile(args.resume_2[1]):
            checkpoint1 = torch.load(args.resume_2[0], map_location='cpu')
            checkpoint2 = torch.load(args.resume_2[1], map_location='cpu')
            model1.load_state_dict(checkpoint1['state_dict'])
            model2.load_state_dict(checkpoint2['state_dict'])
            print("=> 已成功加载checkpoint")
        else:
            print("=> !!!!!!! checkpoint found at '{}'".format(args.resume))
    model1.eval()
    model2.eval()
    for index, item in enumerate(dataParser):
        if item['if_not']:
            continue
        else:
            pass
        print('[{}/{}]'.format(index,length))
        if torch.cuda.is_available():
            images = item['tamper_image'].cuda()
            labels_band = item['gt_band'].cuda()
            labels_dou_edge = item['gt_double_edge'].cuda()
        else:
            images = item['tamper_image']
            labels_band = item['gt_band']
            labels_dou_edge = item['gt_double_edge']

        one_stage_outputs_1 = model1(images)
        rgb_pred_rgb = torch.cat((one_stage_outputs_1[0], images), 1)
        two_stage_outputs_1 = model2(rgb_pred_rgb, one_stage_outputs_1[1], one_stage_outputs_1[2],
                                     one_stage_outputs_1[3])
        y = two_stage_outputs_1[0].reshape(-1)
        l = labels_dou_edge.reshape(-1)
        y = np.array(y.cpu().detach())
        # y = np.where(y > 0.5, 1, 0).astype('int')
        l = np.array(l.cpu().detach()).astype('int')
        y = y.tolist()
        l = l.tolist()
        test_pred.extend(y)
        test_label.extend(l)
        # if index==2:
        #     break
    _fpr, _tpr, _the = metrics.roc_curve(test_label,test_pred,pos_label=1)
    auc = metrics.auc(_fpr, _tpr)
    torch.cuda.empty_cache()
    return _fpr, _tpr, auc

if __name__=='__main__':
    using_data = {
        'casia': False,
        'coverage': False,
    }
    for index,item in enumerate(using_data):
        using_data[item]=True
        testData = TamperDataset(using_data=using_data)
        testDataLoader=torch.utils.data.DataLoader(testData,batch_size=1,num_workers=0)
        test_avg=test(dataParser=testDataLoader,item=item)
        using_data[item] = False
        print(item+'已完成')