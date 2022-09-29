# 填充算法  未写pass
# 两中填充算法 逐像素判定
# 和  种子像素填充
# 作为主函数使用，调用skeleton.py --> Breakpoint_connection.py --> full.py 结果在full.py 的save_dir 中
import Breakpoint_connection2
import edge2area.skeleton as skeleton
import edge2area.Breakpoint_connection as Breakpoint_connection
import edge2area.full as full
import edge2area.automation_full as automation_full
import os

from Get_mask import take_gt_mask


def old_method():
    # 读取目录
    src_dir = r'C:\Users\brighten\Desktop\yuan\Columbia_stage3'
    result1_dir = r'C:\Users\brighten\Desktop\full\Columbia\1'
    result2_dir = r'C:\Users\brighten\Desktop\full\Columbia\2'
    save_dir = r'C:\Users\brighten\Desktop\full\Columbia\3'
    if not os.path.exists(result1_dir):
        os.mkdir(result1_dir)
    if not os.path.exists(result2_dir):
        os.mkdir(result2_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    skeleton.skeletoning(src_dir, result1_dir)  # 较慢，当心
    Breakpoint_connection.connecting(result1_dir, result2_dir)
    # full.fulling(result2_dir, save_dir) # 手动代码
    automation_full.picture_fulling(result2_dir, save_dir)  # 自动代码


def new_method():
    # 读取目录
    src_dir = r'C:\Users\brighten\Desktop\yuan\Columbia_stage3'
    gt_path= r""
    result0_dir = r'C:\Users\brighten\Desktop\full\Columbia\1'
    result1_dir = r'C:\Users\brighten\Desktop\full\Columbia\1'
    result2_dir = r'C:\Users\brighten\Desktop\full\Columbia\2'
    save_dir = r'C:\Users\brighten\Desktop\full\Columbia\3'
    if not os.path.exists(result1_dir):
        os.mkdir(result1_dir)
    if not os.path.exists(result2_dir):
        os.mkdir(result2_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    take_gt_mask(src_dir, gt_path, result0_dir)
    skeleton.skeletoning(src_dir, result1_dir)  # 较慢，当心
    Breakpoint_connection2.connecting(result1_dir, result2_dir)
    automation_full.picture_fulling(result2_dir, save_dir)  # 自动代码


if __name__ == '__main__':
    pass

# columb:
# src_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\pred_train'
# result1_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result1'
# result2_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result2'
# save_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result4'

# casia stage2
# src_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\casia_test\pred_train\stage2'
# result1_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\casia_test\pred_train\result1'
# result2_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\casia_test\pred_train\result2'
# save_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\casia_test\pred_train\result3'

# coverage 2
# src_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\coverage_test\pred_train\stage2'
# result1_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\coverage_test\pred_train\result1'
# result2_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\coverage_test\pred_train\result2'
# save_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\coverage_test\pred_train\result3'
