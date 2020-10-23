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

gamma = 0.95
time_step=15
historys,actions,rewards = load_data_movie_length()

class Reward(keras.Model):
    def __init__(self,batch_size=256,embedding_size=64,epochs=1000,item_count=6040,unit=128,time_step = 15):
        self.batch_size = batch_size
        self.embedding_size=embedding_size
        self.epochs = epochs
        self.item_count = item_count
        self.time_step = time_step

        self.embedding = keras.layers.Embedding(self.item_count+1,self.embedding_size)
        self.lstm = keras.layers.LSTM(units=unit)
        self.full_conn1 = keras.layers.Dense(64)
        self.full_conn2 = keras.layers.Dense(1)


    def call(self,inputs):
        state,action = inputs
        x = self.embedding(state)
        x = self.lstm(x)
        print('AAAAA',x.shape)
        inp = tf.concat([x,action],axis=1)
        print('BBBBB',inp.shape,action.shape)
        out = self.full_conn1(inp)
        out = self.full_conn2(out)
        return out


def main():
    model = Reward()
    # out = model((historys,actions))
    model.summary()

    # model.compile(optimizer = keras.optimizers.Adam(0.001),
    #               loss=keras.losses.MeanSquaredError(),
    #               metrics=['mse'])
    #
    # # train
    # model.fit((historys,actions),rewards,epochs=10)


main()



