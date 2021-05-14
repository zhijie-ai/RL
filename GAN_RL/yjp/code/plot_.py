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

f = open('../data/analysis/loss_random.pkl','rb')
loss = pickle.load(f)
f.close()
print(len(loss[0]),max(loss[0]))
print(loss[0])
print(loss[8])
print(loss[9])


num=10
ran = 50
def plot(data,label,num=10,ran=50):
    plt.figure()
    for d,n in zip(data,label):
        plt.plot(range(len(d)),d,label=n)
        plt.legend()
        plt.grid(True)

    # plt.savefig('jpg/{}'.format(name))
    plt.show()

if __name__ == '__main__':
    plot([loss[0]],['loss_{}'.format(0)],num=1,ran=5)




