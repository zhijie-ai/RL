#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/2 10:59                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
from collections import deque
import numpy as np

training_user = [0,1,2,4]
max_l = len(training_user)*5+6
_k = 10
data_collection={'action':deque(maxlen=max_l)}

for t in range(5):
    random_action = [[] for _ in range(len(training_user))]
    for u_i in range(len(training_user)):
        user_i = training_user[u_i]
        random_action[u_i] = np.random.choice(np.arange(1000), _k, replace=False).tolist()
    data_collection['action'].extend(random_action)

print(len(data_collection['action']),data_collection['action'])

batch_sample = np.random.choice(len(data_collection['action']), 10, replace=False)
print(batch_sample)
action_batch = [data_collection['action'][c] for c in batch_sample]
print(action_batch)