#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/6/15 下午3:12                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd
import pickle
import numpy as np


def process_data(ori_data, history_len, action_len=1):
    whole_data = {}
    for u in ori_data.keys():
        data = {}
        items = ori_data[u]

        state_list = []
        action_list = []
        n_state_list = []
        reward_list = []

        for i in range(len(items) - (action_len + history_len)):

            state_list.append([di['movie_id'] for di in items[i:i + history_len]])
            action_list.append([di['movie_id'] for di in items[i + 1:i + 1 + action_len]])
            reward_list.append([di['ratings'] for di in items[i + 1:i + 1 + action_len]])
            n_state_list.append([di['movie_id'] for di in items[i + action_len:i + action_len + history_len]])


        data['state_float'] = state_list# We only need state in full load,which means 5 elements
        data['action_float'] = action_list
        data['reward_float'] = reward_list
        data['n_state_float'] = n_state_list
        whole_data[u] =data

    return pd.DataFrame.from_dict(whole_data),len(np.unique([item['movie_id'] for di in ori_data.values() for item in di]))




if __name__ == '__main__':
    with open('../data/whole_user_movies.pickle','rb') as f:
        whole_data = pickle.load(f)

    df = process_data(whole_data,10)