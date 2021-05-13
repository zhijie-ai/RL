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

f = open('../data/analysis/analysis2_LSTM_50.pkl', 'rb')
lstm_loss_ = pickle.load(f)
lstm_p1_ = pickle.load(f)
lstm_p2_ = pickle.load(f)
f.close()
f = open('../data/analysis/analysis2_PW_50.pkl', 'rb')
pw_loss_ = pickle.load(f)
pw_p1_ = pickle.load(f)
pw_p2_ = pickle.load(f)
f.close()
f = open('../data/analysis/analysis2_PW_50.pkl', 'rb')
lstm_loss = pickle.load(f)
lstm_p1 = pickle.load(f)
lstm_p2 = pickle.load(f)
f.close()
f = open('../data/analysis/analysis_PW_50.pkl', 'rb')
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

    plt.savefig('jpg/{}'.format(name))



# plt.subplot(311)
# lstm_loss_ = [np.mean(lstm_loss_[ind-num:ind+num]) for ind ,val in enumerate(lstm_loss_) if ind%ran==num]
# pw_loss_ = [np.mean(pw_loss_[ind-num:ind+num]) for ind ,val in enumerate(pw_loss_) if ind%ran==num]
# pw_loss = [np.mean(pw_loss[ind-num:ind+num]) for ind ,val in enumerate(pw_loss) if ind%ran==num]
# lstm_loss = [np.mean(lstm_loss[ind-num:ind+num]) for ind ,val in enumerate(lstm_loss) if ind%ran==num]
# plt.plot(range(len(lstm_loss_)),lstm_loss_,label='lstm-loss_')
# plt.plot(range(len(lstm_loss)),lstm_loss,label='lstm-loss')
# plt.plot(range(len(pw_loss_)),pw_loss_,label='pw-loss_')
# plt.plot(range(len(pw_loss)),pw_loss,label='pw-loss')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(312)
# lstm_p1_ = [np.mean(lstm_p1_[ind-num:ind+num]) for ind ,val in enumerate(lstm_p1_) if ind%ran==num]
# pw_p1_ = [np.mean(pw_p1_[ind-num:ind+num]) for ind ,val in enumerate(pw_p1_) if ind%ran==num]
# pw_p1 = [np.mean(pw_p1[ind-num:ind+num]) for ind ,val in enumerate(pw_p1) if ind%ran==num]
# lstm_p1 = [np.mean(lstm_p1[ind-num:ind+num]) for ind ,val in enumerate(lstm_p1) if ind%ran==num]
# plt.plot(range(len(lstm_p1_)),lstm_p1_,label='lstm-prec1_')
# plt.plot(range(len(lstm_p1)),lstm_p1,label='lstm-prec1')
# plt.plot(range(len(pw_p1_)),pw_p1_,label='pw-prec1_')
# plt.plot(range(len(pw_p1)),pw_p1,label='pw-prec1')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(313)
# lstm_p2_ = [np.mean(lstm_p2_[ind-num:ind+num]) for ind ,val in enumerate(lstm_p2_) if ind%ran==num]
# pw_p2_ = [np.mean(pw_p2_[ind-num:ind+num]) for ind ,val in enumerate(pw_p2_) if ind%ran==num]
# pw_p2 = [np.mean(pw_p2[ind-num:ind+num]) for ind ,val in enumerate(pw_p2) if ind%ran==num]
# lstm_p2 = [np.mean(lstm_p2[ind-num:ind+num]) for ind ,val in enumerate(lstm_p2) if ind%ran==num]
# plt.plot(range(len(lstm_p2_)),lstm_p2_,label='lstm-prec2_')
# plt.plot(range(len(lstm_p2)),lstm_p2,label='lstm-prec2')
# plt.plot(range(len(pw_p2_)),pw_p2_,label='pw-prec2_')
# plt.plot(range(len(pw_p2)),pw_p2,label='pw-prec2')
# plt.xlabel('step')
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('jpg/all.jpg')

if __name__ == '__main__':
    #1. 对比pw和lstm效果的区别
    plot([lstm_loss_,pw_loss_],['loss-lstm_','loss-pw_'],'loss_',num=10)
    plot([lstm_p1_,pw_p1_],['p1-lstm_','p1-pw_'],'p1_',num=10)
    plot([lstm_p2_,pw_p2_],['p2-lstm_','p2-pw_'],'p2_',num=10)

    plot([lstm_loss,pw_loss],['loss-lstm','loss-pw'],'loss',num=10)
    plot([lstm_p1,pw_p1],['p1-lstm','p1-pw'],'p1',num=10)
    plot([lstm_p2,pw_p2],['p2-lstm','p2-pw'],'p2',num=10)

    # 1. 对比循环方式的区别
    plot([lstm_loss_,lstm_loss],['loss-lstm_','loss-lstm'],'loss_lstm')
    plot([pw_loss_,pw_loss],['loss_pw_','loss_pw'],'loss_pw')

    plot([lstm_p1_,lstm_p1],['p1-lstm_','p1-lstm'],'p1_lstm')
    plot([pw_p1_,pw_p1],['p1_pw_','p1-pw'],'p1_pw',num=10)

    plot([lstm_p2_,lstm_p2],['p2-lstm_','p2-lstm'],'p2_lstm')
    plot([pw_p2_,pw_p2],['p2-pw_','p2-pw'],'p2_pw')


