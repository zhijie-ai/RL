#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/9/30 14:44                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np
import pickle

data = np.random.rand(10,7)
y = np.random.randint(0,2,(10,1))
data = np.concatenate([data,y],axis=1)
print(type(data[:,1:6]))

def get_train_data(batch_size=60,time_step=4,train_begin=0,train_end=5800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集x和y初定义
    for i in range(len(normalized_train_data)-time_step):
        if i % batch_size==0:
            batch_index.append(i)
        x=normalized_train_data[i:i+time_step,:7]
        y=normalized_train_data[i:i+time_step,7,np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
        print(y)
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

def load_data(path='../data/session.pickle',time_step=7):
    historys=[]
    actions=[]
    rewards=[]
    with open(path,'rb') as f:
        trajectory,rewards_= pickle.load(f)
        for t,r in zip(trajectory,rewards_):
            for i in range(len(t)-time_step):
                historys.append(list(t[i:i+time_step]))
                actions.append(t[i+time_step])
                rewards.append(r[i+time_step])


    return historys,actions,rewards

print(load_data()[2])