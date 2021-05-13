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

f = open('../data/analysis/analysis2_LSTM.pkl','rb')
lstm_loss_ = pickle.load(f)
lstm_p1_ = pickle.load(f)
lstm_p2_ = pickle.load(f)
f.close()
f = open('../data/analysis/analysis_PW.pkl', 'rb')
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

    # plt.savefig('jpg/{}'.format(name))
    plt.show()

if __name__ == '__main__':
    plot([lstm_loss_,pw_loss],['loss-lstm_','loss-pw'],'loss_',num=1)
    plot([lstm_p1_,pw_p1],['p1-lstm_','p1-pw'],'p1_',num=1)
    plot([lstm_p2_,pw_p2],['p2-lstm_','p2-pw'],'p2_',num=1)



