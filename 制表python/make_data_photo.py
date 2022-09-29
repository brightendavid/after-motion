#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
colours=["#62BEE1","#fcce0e","#e77d24","#277379","#bf232b"]
current_palette = sns.color_palette("RdBu")
# 分类色
size = 4
t = np.random.random(6)
# casia
a = np.array([0.603252892, 0.32184586, 0.366311235, 0.943413967])
b = np.array([0.748235623, 0.001412467, 0.002762942, 0.922527287])
c = np.array([0.071260466, 0.593152739, 0.10623978, 0.480644112])

d = np.array([0.460100883, 0.7272168, 0.483484081, 0.884933187])
e = np.array([0.22351361, 0.058168871, 0.048820644, 0.904351599])
f = np.array([0.912481655, 0.773882027, 0.814623881, 0.979152729])

total_width, n = 0.72, 5
width = total_width / n
x = np.arange(size)

x = x - (total_width - width) / 2
print("sssssss")
print(x)

plt.bar(x, a, width=width, label="ManTra-Net", color=colours[0])
plt.bar(x + width, b, width=width, label="ELA", color=colours[1])
plt.bar(x + width * 2, c, width=width, label="DWT", color=colours[2])
# plt.bar(x + width * 3, d, width=width, label="RRU", color=sns.color_palette("RdBu")[3])
plt.bar(x + width * 3, e, width=width, label="HLED", color=colours[3])
plt.bar(x + width * 4, f, width=width, label="OURS", color=colours[4])

plt.ylabel("ratio")
plt.title("CASIA")
labels = ['Precision ','Recall ','F1','Acc']
z = np.arange(len(labels))
plt.xticks(z, labels)
z = np.arange(len(labels)-1)
plt.yticks(z)
plt.legend()
plt.savefig("CASIA.png")
plt.show()



# coveage
a = np.array([0.582567778,	0.1633604,	0.227596705	,0.882903921
])
b = np.array([0.536564896,	8.26523E-05,	0.00016485,	0.874765167
])
c = np.array([0.138714521	,0.639255872,0.221723732	,0.440097912
])

d = np.array([0.228486935,	0.455494063	,0.276932454,	0.74125558
])
e = np.array([0.345168986,	0.135282519	,0.139565661	,0.8580378
])
f = np.array([0.893902173,	0.800839133	,0.825248885,	0.966393899])

total_width, n = 0.72, 5
width = total_width / n
x = np.arange(size)

x = x - (total_width - width) / 2
print("sssssss")
print(x)

plt.bar(x, a, width=width, label="ManTra-Net", color=colours[0])
plt.bar(x + width, b, width=width, label="ELA", color=colours[1])
plt.bar(x + width * 2, c, width=width, label="DWT", color=colours[2])
# plt.bar(x + width * 3, d, width=width, label="RRU", color=sns.color_palette("RdBu")[3])
plt.bar(x + width * 3, e, width=width, label="HLED", color=colours[3])
plt.bar(x + width * 4, f, width=width, label="OURS", color=colours[4])

plt.ylabel("ratio")
plt.title("COVERAGE")
labels = ['Precision ','Recall ','F1','Acc']
z = np.arange(len(labels))
plt.xticks(z, labels)
z = np.arange(len(labels)-1)
plt.yticks(z)
plt.legend()
plt.savefig("COVERAGE.png")
plt.show()




# columbia
a = np.array([0.744370764	,0.22154641	,0.304290528	,0.748120337
])
b = np.array([0.689916012,	9.24401E-05,	0.000184749	,0.740888864

])
c = np.array([0.356272423,	0.35248243,	0.315089339,	0.646890059

])

d = np.array([0.499821254	,0.538478604,	0.492631373	,0.707821224

])
e = np.array([0.349990312	,0.039161528,	0.06479272,	0.727088026

])
f = np.array([0.993692295	,0.984108522,	0.98886671,	0.993619307
])


total_width, n = 0.72, 5
width = total_width / n
x = np.arange(size)

x = x - (total_width - width) / 2
print("sssssss")
print(x)

plt.bar(x, a, width=width, label="ManTra-Net", color=colours[0])
plt.bar(x + width, b, width=width, label="ELA", color=colours[1])
plt.bar(x + width * 2, c, width=width, label="DWT", color=colours[2])
# plt.bar(x + width * 3, d, width=width, label="RRU", color=sns.color_palette("RdBu")[3])
plt.bar(x + width * 3, e, width=width, label="HLED", color=colours[3])
plt.bar(x + width * 4, f, width=width, label="OURS", color=colours[4])

plt.ylabel("ratio")
plt.title("Columbia")
labels = ['Precision ','Recall ','F1','Acc']
z = np.arange(len(labels))
plt.xticks(z, labels)
z = np.arange(len(labels)-1)
plt.yticks(z)
plt.legend()
plt.savefig("Columbia.png")
plt.show()
