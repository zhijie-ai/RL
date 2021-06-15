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

f = open('../data/analysis/gan_rl_PW_50.pkl', 'rb')
pw_loss = pickle.load(f)
pw_p1 = pickle.load(f)
pw_p2 = pickle.load(f)
f.close()


num=10
ran = 50
def plot(data,label,name,num=10,ran=50):
    plt.figure()
    for d,n in zip(data,label):
        d = [np.mean(d[ind-num:ind+num]) for ind ,val in enumerate(d) if ind%ran==num]
        plt.plot(range(len(d)),d,label=n)
        plt.legend()
        plt.grid(True)

    plt.show()

    # plt.savefig('jpg7/{}'.format(name))


if __name__ == '__main__':

    plot([pw_loss,pw_loss],['loss-lstm','loss-pw'],'loss',num=10)
    plot([pw_p1,pw_p1],['p1-lstm','p1-pw'],'p1',num=10)
    plot([pw_p2,pw_p2],['p2-lstm','p2-pw'],'p2',num=10)



