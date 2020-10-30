# -----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/10/22 15:56                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# -----------------------------------------------
# reward的预测
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
import time

# 自定义模型时，在加载模型时如果采用偷懒的方式采用model_from_yaml加载会报相关错，如load.py文件，但重新定义一遍模型，再load模型权重，是可以的。
def load_data_movie_length(path='../data/ratings.dat', time_step=15, gamma=.9):
    historys = []
    actions = []
    rewards = []

    def _discount_and_norm_rewards(rewards):
        discounted_episode_rewards = np.zeros_like(rewards, dtype='float64')
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[t]
            discounted_episode_rewards[t] = cumulative
        # Normalize the rewards
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    ratings = pd.read_csv(path, delimiter='::', index_col=None, header=None,
                          names=['userid', 'itemid', 'rating', 'timestamp'], engine='python')
    print(ratings.head())

    items = list(sorted(ratings.itemid.unique()))
    key_to_id_item = dict(zip(items, range(len(items))))
    id_to_key_item = dict(zip(range(len(items)), items))
    users = list(set(sorted(ratings.userid.unique())))
    key_to_id_user = dict(zip(users, range(len(users))))
    id_to_key_user = dict(zip(range(len(users)), users))

    ratings.userid = ratings.userid.map(key_to_id_user)
    ratings.itemid = ratings.itemid.map(key_to_id_item)
    ratings = ratings.sort_values(by=['timestamp']).drop('timestamp', axis=1).groupby('userid')
    for _, df in ratings:
        r = _discount_and_norm_rewards(df.rating.values)
        items = df.itemid.values
        for i in range(len(items) - time_step):
            historys.append(list(items[i:i + time_step]))
            actions.append(items[i + time_step])
            rewards.append(r[i + time_step])

    return np.array(historys), np.array(actions), np.array(rewards)


class Reward(keras.Model):
    def __init__(self, embedding_size=64, epochs=1000, item_count=6040, unit=128, time_step=15,batch_size=512):
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.item_count = item_count
        self.time_step = time_step
        self.units = unit
        self.batch_size=batch_size

        ipt1 = keras.Input(shape=(self.time_step,), name="history")
        ipt2 = keras.Input(shape=(1,), name='actions')

        # action
        action = keras.layers.Embedding(self.item_count, self.embedding_size, input_length=self.time_step)(ipt2)
        print('action------------', action.shape)
        action = K.squeeze(action, axis=1)
        # action = K.reshape(action,(-1,self.embedding_size))
        print('action', action.shape)
        x = keras.layers.Embedding(self.item_count, self.embedding_size, input_length=self.time_step)(ipt1)
        x = keras.layers.LSTM(units=self.units)(x)
        print('x.shape', x.shape)
        x = keras.layers.concatenate([action, x])
        print('AAAA', x.shape)
        x = keras.layers.Dense(64)(x)
        out = keras.layers.Dense(1)(x)
        super(Reward, self).__init__(inputs=[ipt1, ipt2], outputs=out)


def main():
    t1 = time.time()
    print('loading model.......{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))))
    # historys, actions, rewards = load_data_movie_length()

    model = Reward()

    filepath = "model/weights3.best.hdf5"
    model.load_weights(filepath)
    # print(model.predict([historys[0:10],actions[0:10]]))

    t2 = time.time()
    print('model training end~~~~~~{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t2))))
    print('time cost :{} m'.format((t2-t1)/60))


if __name__ == '__main__':
    main()
