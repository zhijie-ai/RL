#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/21 21:31                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd

actions = pd.DataFrame([[1,'A',1],[1,'A',2],[1,'B',1],
                     [1,'C',3],[2,'A',1]],columns=['user_id','sku_id','type'])
print(actions)
df = pd.get_dummies(actions['type'])
actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
print(actions)