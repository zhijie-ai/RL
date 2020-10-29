# -----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/10/29 9:55                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# -----------------------------------------------
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


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


class Reward():
    def __init__(self, embedding_size=64, epochs=1000, item_count=6040, unit=128, time_step=15, gamma=0.9,
                 batch_size=512):
        self.embedding_size = embedding_size
        self.item_count = item_count
        self.time_step = time_step
        self.units = unit
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size

        self.historys, self.actions, self.rewards = load_data_movie_length(time_step=self.time_step, gamma=self.gamma)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the model
        self.model = self.build_model()

        self.model.compile(optimizer=optimizer,
                           loss=keras.losses.MeanSquaredError(),
                           metrics=['mae'])

    def build_model(self):
        inp1 = Input(shape=(self.time_step,), name='history')
        inp2 = Input(shape=(1,), name='action')

        action = Embedding(self.item_count, self.embedding_size, input_length=self.time_step)(inp2)
        print('action------------', action.shape)
        action = K.squeeze(action, axis=1)
        # action = K.reshape(action,(-1,self.embedding_size))
        print('action', action.shape)
        x = Embedding(self.item_count, self.embedding_size, input_length=self.time_step)(inp1)
        x = LSTM(units=self.units)(x)
        print('x.shape', x.shape)
        x = concatenate([action, x])
        print('AAAA', x.shape)
        x = Dense(64)(x)
        out = Dense(1)(x)

        model = Model([inp1, inp2], out, name='reward_model')

        plot_model(model, to_file='./png/model.png', show_shapes=True)
        return model

    def train(self):
        historys_train, historys_val, action_train, action_val, rewards_train, rewards_val = train_test_split(
            self.historys, self.actions,
            self.rewards, test_size=0.2)
        print('GGGGGGGG', historys_train.shape, historys_val.shape, action_train.shape, action_val.shape,
              rewards_train.shape, rewards_val.shape)
        filepath = "weights.best.hdf5"

        ckp = ModelCheckpoint(filepath, save_best_only=True, verbose=1,monitor='val_mae')
        stop = EarlyStopping(patience=10, verbose=1)

        # train
        self.model.fit([historys_train, action_train], rewards_train,
                       epochs=self.epochs,
                       batch_size=self.batch_size, verbose=1, shuffle=True,
                       validation_data=([historys_val,action_val],rewards_val),
                       callbacks=[ckp,stop])


if __name__ == '__main__':
    reward = Reward(epochs=1)
    reward.train()
