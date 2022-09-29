# columb
self.pred_dir = r'C:\Users\brighten\Desktop\0324_两阶段_0306模型,只监督条带区域，带8张图\columb\result4'
self.src_dir = r'C:\Users\brighten\Desktop\元数据集\Columbia Uncompressed Image Splicing Detection Evaluation Dataset\4cam_auth\4cam_auth'
self.gt_dir = r'C:\Users\brighten\Desktop\元数据集\Columbia Uncompressed Image Splicing Detection Evaluation Dataset\4cam_auth\4cam_auth\edgemask'


## columb命名 实例

### src:
canong3_canonxt_sub_01.tif

### gt:
canong3_canonxt_sub_01_edgemask_3.jpg

### pred:
output_canong3_canonxt_sub_01.tif


# 运行顺序

使用所有代码之前修改路径

先运行_anylize后缀代码，生成对应数据集的excel文件，包含原图路径和精确度三个指标

_anylize 输入为pred和gt的路径

src_dir是为了保证数据的完整性写的，修改后面的data 可删

excel 文件先进行求和 和 排序 ,求和与排序部分代码未写.

也就是拉表格的事情，求和排序，可以考虑加入此代码(已完成)

再使用read_excel.py 读取excel 文件中标注的图片路径文字读取图片保持到对应文件夹



# 现在工作

完成mantranet 算法测试(over)

cfa算法 (matlab代码)(不做)换掉

ELA算法(over)

传统噪声算法(over)