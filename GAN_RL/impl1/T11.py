#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/7 11:17                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np

user_set = [9, 2, 5, 1, 8, 6, 5, 3, 5, 8]
states_id = [np.random.choice(10,np.random.choice(range(1,6),1),replace=False).tolist() for _ in range(10)]
print(states_id)
history_order = [[] for _ in range(len(user_set))]
history_user = [[] for _ in range(len(user_set))]
for uu in range(len(user_set)):
    user = user_set[uu]
    id_cnt = len(states_id[uu])
    history_order[uu] = np.arange(id_cnt, dtype=np.int64).tolist()
    history_user[uu] = list(uu * np.ones(id_cnt, dtype=np.int64))

print(history_order)
print(history_user)