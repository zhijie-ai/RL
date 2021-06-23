#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/5/10 16:17                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pickle

f = open('../data/analysis/loss_random_0.3_10_0.001_1024_filtered_0.8.pkl', 'rb')#数据集过滤了空值，0.8用户
f = open('../data/analysis/loss_random_0.3_10_0.001_1024_filtered_all.pkl', 'rb')# 过滤了空值，全部用户
f = open('../data/analysis/loss_random_0.3_10_0.001_1024_not_filtered_0.8.pkl', 'rb')# 不过滤空值，0.8用户
f = open('../data/analysis/loss_random_0.3_10_0.001_1024_not_filtered_all.pkl', 'rb')# 不过滤空值，全部用户
#
f = open('../data/analysis/loss_greedy_0.3_10_0.001_1024_filtered_0.8.pkl', 'rb')#数据集过滤了空值，0.8用户
f = open('../data/analysis/loss_greedy_0.3_10_0.001_1024_filtered_all.pkl', 'rb')# 过滤了空值，全部用户
f = open('../data/analysis/loss_greedy_0.3_10_0.001_1024_not_filtered_0.8.pkl', 'rb')# 不过滤空值，0.8用户
f = open('../data/analysis/loss_greedy_0.3_10_0.001_1024_not_filtered_all.pkl', 'rb')# 不过滤空值，全部用户
#
f = open('../data/analysis/loss_comb_0.3_10_0.001_1024_filtered_0.8.pkl', 'rb') # 数据集过滤了空值，0.8用户
f = open('../data/analysis/loss_comb_0.3_10_0.001_1024_filtered_all.pkl', 'rb')# 过滤了空值，全部用户
f = open('../data/analysis/loss_comb_0.3_10_0.001_1024_not_filtered_0.8.pkl', 'rb')# 不过滤空值，0.8用户
f = open('../data/analysis/loss_comb_0.3_10_0.001_1024_not_filtered_all.pkl', 'rb')# 不过滤空值，全部用户
loss = pickle.load(f)
f.close()
print(loss[0])
print(loss[1])
print(loss[2])


num=10
ran = 15
def plot(data_,name):
    for ind,data in enumerate(data_):
        data = [np.mean(data[ind-num:ind+num]) for ind ,val in enumerate(data) if ind%ran==num]
        plt.plot(range(len(data)),data,label=ind)
    plt.legend()
    plt.grid(True)

    plt.savefig('jpg8/{}.png'.format(name))
    # plt.show()

if __name__ == '__main__':
    plot(loss,'comb_not_filtered_all')




