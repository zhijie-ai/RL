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
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import time


def load_data_movie_length(path='../data/ratings_1m.dat', time_step=15, gamma=.9):
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
        x = keras.layers.concatenate([action, x])  # https://zhuanlan.zhihu.com/p/81721574
        print('AAAA', x.shape)
        x = keras.layers.Dense(64)(x)
        out = keras.layers.Dense(1)(x)
        super(Reward, self).__init__(inputs=[ipt1, ipt2], outputs=out)


def main():
    t1 = time.time()
    print('start model training.......{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))))
    historys, actions, rewards = load_data_movie_length()
    historys_train, historys_val, action_train, action_val, rewards_train, rewards_val = train_test_split(historys,
                                                                                                          actions,
                                                                                                          rewards,
                                                                                                          test_size=0.2)

    model = Reward(epochs=1)
    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    model_yaml = model.to_yaml()
    print(model_yaml)
    with open('model/model.yaml', 'w') as f:
        f.write(model_yaml)

    print('EEEEE',model.metrics_names)
    filepath = "model/weights.best.hdf5"

    ckp = ModelCheckpoint(filepath, save_best_only=True, verbose=1,monitor='val_mse')
    stop = EarlyStopping(patience=10, verbose=1)

    # train
    model.fit([historys_train, action_train], rewards_train,
              epochs=model.epochs, batch_size=model.batch_size, verbose=1,
              validation_data=([historys_val, action_val], rewards_val),
              shuffle=True,
              callbacks=[stop,ckp])

    t2 = time.time()
    filepath = "model/weights3.best.hdf5"
    model.save_weights(filepath)
    print('model training end~~~~~~{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t2))))
    print('time cost :{} m'.format((t2-t1)/60))


if __name__ == '__main__':
    main()
    # tf.saved_model.save(model,'model')
