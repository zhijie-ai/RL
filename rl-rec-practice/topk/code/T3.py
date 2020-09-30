#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/9/30 15:46                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np

actions = [1,10,22,14,16,8]

def get_index(actions):
    idx = []
    for i,v in enumerate(actions):
        idx.append([i,v])
    return idx

print(get_index(actions))