#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/11/20 14:37                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------

import pandas as pd
import numpy as np

path='../data/ratings.dat'
df = pd.read_csv(path,delimiter='::',index_col=None,header=None,names=['userid','itemid','rating','timestamp'],engine='python')

user = range(1,6041)
test_user = np.random.choice(user,1000,replace=False)
train_user = [i for i in user if i not in test_user]

train = df[~(df.userid.isin(test_user))]
test = df[df.userid.isin(test_user)]
train.to_csv('../data/train_ratings.csv',index=None)
test.to_csv('../data/test_ratings.csv',index=None)