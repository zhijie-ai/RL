#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/26 10:54                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd

import numpy as np

import random

train_data = np.zeros([4,100],dtype=np.int32)

random.seed(1)

for i in range(100):
    train_data[0,i] = random.randint(0,20)
    train_data[1,i] = random.randint(0,200)
    train_data[2,i] = random.randint(0,2000)
    train_data[3,i] = random.randint(0,2000)


train_data = np.transpose(train_data)


train_df = pd.DataFrame(train_data,columns=['userid','itemid','rating','timestamp']).to_csv('train2.csv')