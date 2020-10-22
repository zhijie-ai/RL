#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/10/22 15:56                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
# reward的预测
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

def load_data_movie_length(path='../data/ratings.dat',time_step=15,gamma=.9):
    historys=[]
    actions=[]
    rewards=[]

    def _discount_and_norm_rewards(rewards):
        discounted_episode_rewards = np.zeros_like(rewards,dtype='float64')
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[t]
            discounted_episode_rewards[t] = cumulative
        # Normalize the rewards
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    ratings = pd.read_csv(path,delimiter='::',index_col=None,header=None,names=['userid','itemid','rating','timestamp'],engine='python')
    print(ratings.head())

    items = list(sorted(ratings.itemid.unique()))
    key_to_id_item = dict(zip(items,range(len(items))))
    id_to_key_item = dict(zip(range(len(items)),items))
    users = list(set(sorted(ratings.userid.unique())))
    key_to_id_user = dict(zip(users,range(len(users))))
    id_to_key_user = dict(zip(range(len(users)),users))

    ratings.userid = ratings.userid.map(key_to_id_user)
    ratings.itemid = ratings.itemid.map(key_to_id_item)
    ratings = ratings.sort_values(by=['timestamp']).drop('timestamp',axis=1).groupby('userid')
    for _,df in ratings:
        r = _discount_and_norm_rewards(df.rating.values)
        items = df.itemid.values
        for i in range(len(items)-time_step):
            historys.append(list(items[i:i+time_step]))
            actions.append(items[i+time_step])
            rewards.append(r[i+time_step])


    return np.array(historys),np.array(actions),np.array(rewards)


class Reward():
    def __init__(self,batch_size=256,embedding_size=64,epochs=1000):
        self.batch_size = batch_size
        self.embedding_size=64
        self.epochs = epochs


    def build_network(self):
        pass




