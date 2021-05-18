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

f = open('../data/analysis/analysis_PW_data_model.pkl', 'rb')
loss = pickle.load(f)
p1 = pickle.load(f)
p2 = pickle.load(f)
f.close()


num=10
ran = 15
def plot(data):
    data = [np.mean(data[ind-num:ind+num]) for ind ,val in enumerate(data) if ind%ran==num]
    plt.plot(range(len(data)),data)
    plt.legend()
    plt.grid(True)

    # plt.savefig('jpg/{}'.format(name))
    plt.show()

if __name__ == '__main__':
    plot(loss)




