import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *


def Distance(x,y):
    return np.sqrt(sum((x-y)**2))  

def run_K_means(data_x,cluster_center,k):
	# E步计算数据分类
	pred_y = []
	sum_distance = 0
	for x in data_x:
		dis_ls = [Distance(x,center) for center in cluster_center]
		pred_y.append(dis_ls.index(min(dis_ls)))
		sum_distance += min(dis_ls)
	pred_y = np.array(pred_y)

	# M步重新计算中心点
	cluster_center = np.array([list(np.mean(data_x[pred_y==i],axis = 0)) for i in range(k)])
	return cluster_center,pred_y,sum_distance

k = int(input("k:"))
kmeans_time = int(input("kmeans_time:"))
iterTime = int(input("iterTime:"))
n_samples = int(input("n_samples:"))
# 生成数据
data_x,data_y = make_blobs(n_features=2,n_samples=n_samples,centers=k,cluster_std = 2,center_box=(-20.0, 20.0))
min_sum_distance = 10**1000
for i in range(kmeans_time):
	# 初始化中心点
	cluster_center = data_x[np.random.randint(len(data_x),size = k)]
	count = 0
	plt.ion()                     # 开启一个画图的窗口
	while count < iterTime:
		plt.clf()                 # 清除之前画的图
		cluster_center,pred_y,sum_distance = run_K_means(data_x,cluster_center,k)
		plt.scatter([i[0] for i in data_x],[i[1] for i in data_x],c = pred_y)
		plt.scatter([i[0] for i in cluster_center],[i[1] for i in cluster_center],c = 'r',s = 25,marker = "v")
		plt.title("%s times kmeans %s times iteration"%(i+1,count+1))
		plt.pause(0.5)              # 暂停一秒
		plt.ioff()                 # 关闭画图的窗口
		count += 1
	if sum_distance < min_sum_distance:
		min_sum_distance = sum_distance
		result_center = cluster_center
		result_y = pred_y
print(result_center)
plt.clf()
plt.scatter([i[0] for i in data_x],[i[1] for i in data_x],c = result_y)
plt.scatter([i[0] for i in result_center],[i[1] for i in result_center],c = 'r',s = 25,marker = "v")
plt.title("final iteration")
plt.show()
