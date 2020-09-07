#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/22 下午5:36                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import random
from collections import deque
import numpy as np
from drr.preprocesing import process_data


'''
将训练数据整理成(s,a,r,s_)存放到replay_buffer中
'''
class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

        # 训练数据
        path1 = 'data/dumped/user_movies.json'  # 与各个中心点最近的用户对电影的评分
        path2 = 'data/user_train.npy'  # u.user 文件中的用户,然后用train_and_test切分成2部分,训练部分用户
        path3 = 'data/dumped/movie_items.json'  # u.item文件中去除4个字段后的数据,相当与把id当成了index的df
        history_len = 5
        data, item_embed, user_embed = process_data(path1, path2, path3, history_len)
        data = data.T
        for idx, row in data.iterrows():#存放了训练用户中每个用户对电影的评分,dict
            # print(len(row['state_float']))
            for i in range(1):
                sample = []
                state = {key: item_embed[str(key)] for key in row['state_float'][i]}
                action = item_embed[str(row['action_float'][i])]
                n_state = {key: item_embed[str(key)] for key in row['n_state_float'][i]}
                sample.append(np.array(list(state.values())))
                sample.append(action)
                sample.append(np.array(row['reward_float'][i]))
                sample.append(np.array(list(n_state.values())))
                self.buffer.append(sample)

    def add(self, state, action, reward, next_reward):
        experience = (state, action, reward, next_reward)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self,batch_size):
        return random.sample(self.buffer,batch_size)

    def clear(self):
        self.buffer.clear()
        self.count=0