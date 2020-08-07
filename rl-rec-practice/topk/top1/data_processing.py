#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/8/3 14:26                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
# 数据预处理环节
import pickle
import numpy as np

with open('../data/session.pickle','rb') as f:
    data = pickle.load(f)
    actions = data[0]
    rewards = data[1]

print(rewards[1])