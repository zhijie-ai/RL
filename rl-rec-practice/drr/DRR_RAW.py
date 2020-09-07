#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/23 上午11:16                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

# Importing tensorflow
import tensorflow as tf
import json
# Importing some more libraries
import pandas as pd
import numpy as np
import os
import argparse
import pprint as pp
import random
from collections import deque
from sklearn.preprocessing import minmax_scale
from scipy.special import comb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import time

# In[ ]:


# Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('/content/gdrive/My Drive/drr/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

# Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('/content/gdrive/My Drive/drr/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

# Reading items file:
i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('/content/gdrive/My Drive/drr/ml-100k/u.item', sep='|', names=i_cols,
                    encoding='latin-1')

# In[ ]:


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('/content/gdrive/My Drive/drr/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('/content/gdrive/My Drive/drr/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_train.shape, ratings_test.shape

# encode string into integers in new_users
new_users = users.values
new_items = items.values
new_items = np.delete(new_items, [1, 2, 3, 4], 1)

# convert ratings to 1/2(ratings-3)
new_ratings = ratings.values
new_ratings = new_ratings.astype(float)
for row in new_ratings:
    a = row[2]
    a = (a - 3) / 2
    row[2] = a

a = LabelEncoder()
new_users[:, 2] = a.fit_transform(new_users[:, 2])
new_users[:, 3] = a.fit_transform(new_users[:, 3])
new_users[:, 4] = a.fit_transform(new_users[:, 4])

user_train, user_test = train_test_split(new_users, test_size=0.2)

# In[ ]:


print(users.shape)
users.head()

# In[ ]:


len(user_train)

# In[ ]:


# begin the drr
"""
Created by Yunfei Wang on 2019/07/26
"""


# In[ ]:


def cluster_data(num_clusters):
    # n clusters for users: young/old, male/female, 16 jobs, zip code
    train_data = np.delete(user_train, 0, 1)  # traning user data without user_id
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(train_data)
    b = {}
    unique = set(kmeans)
    for i in unique:
        num = 0
        for j in kmeans:
            if j == i:
                num += 1
        b[i] = num

    # find the n nearest data points of each cluster center and get their user_id
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(train_data)
    center = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(center, train_data)
    train_userid = []
    for i in closest:
        train_userid.append(user_train[i][0])

    # find the clusters for test users
    test_clusters = {}
    index = kmeans.predict(np.delete(user_test, 0, 1))
    for i in range(len(user_test)):
        test_clusters[int(user_test[i][0])] = int(index[i])

        # find all relevant movies_id and ratings of these picked train users
    train_movieid = {}
    for i in train_userid:
        item = []
        for row in new_ratings:
            movie = {}
            if i == row[0]:
                movie['movie_id'] = int(row[1])
                movie['ratings'] = float(row[2])
                item.append(movie)
        train_movieid[int(i)] = item

    # convert items into dictionary
    movie_items = {}
    for row in new_items:
        movie_items[int(row[0])] = list(row[1:])

    # find all relevant movies_id and ratings of these picked test users
    test_movieid = {}
    for i in user_test[:, 0]:
        item = []
        for row in new_ratings:
            movie = {}
            if i == row[0]:
                movie['movie_id'] = int(row[1])
                movie['ratings'] = float(row[2])
                item.append(movie)
        test_movieid[int(i)] = item

    whole_movieid = {}
    for i in new_users[:, 0]:
        item = []
        for row in new_ratings:
            movie = {}
            if i == row[0]:
                movie['movie_id'] = int(row[1])
                movie['ratings'] = float(row[2])
                item.append(movie)
        whole_movieid[int(i)] = item

    with open('/content/gdrive/My Drive/drr/user_movies.json', 'w') as fp:
        json.dump(train_movieid, fp)

    with open('/content/gdrive/My Drive/drr/test_user_movies.json', 'w') as fp:
        json.dump(test_movieid, fp)

    with open('/content/gdrive/My Drive/drr/whole_user_movies.json', 'w') as fp:
        json.dump(whole_movieid, fp)

    with open('/content/gdrive/My Drive/drr/movie_items.json', 'w') as fp:
        json.dump(movie_items, fp)

    with open('/content/gdrive/My Drive/drr/test_users.json', 'w') as fp:
        json.dump(test_clusters, fp)

    np.save('/content/gdrive/My Drive/drr/user_train', user_train)
    np.save('/content/gdrive/My Drive/drr/user_test', user_test)


# In[ ]:


# process the data to given form(state, action, reward, next_state, recall)

def process_data(data_path, users_path, items_path, history_len):
    with open(data_path, 'r') as fp:
        ori_data = json.load(fp)

    whole_data = {}
    for k in ori_data.keys():
        data = {}
        state = deque()
        n_state = deque()
        items = ori_data[k]

        state_list = []
        action_list = []
        n_state_list = []
        reward_list = []

        n_state.append(items[0]['movie_id'])
        for i in range(len(items) - 1):
            if len(state) < history_len:
                state.append(items[i]['movie_id'])
            else:
                state.popleft()
                state.append(items[i]['movie_id'])

            state_list.append(list(state))
            action_list.append(items[i + 1]['movie_id'])
            reward_list.append(items[i + 1]['ratings'])
            if len(n_state) < history_len:
                n_state.append(items[i + 1]['movie_id'])
            else:
                n_state.popleft()
                n_state.append(items[i + 1]['movie_id'])
            n_state_list.append(list(n_state))

        # print(action_list)
        # print(state_list)
        # sample.append((list(state), action, reward, list(n_state), recall))

        data['state_float'] = state_list[history_len - 1:]  # we only need state in full load, which means 5 elements
        data['action_float'] = action_list[history_len - 1:]
        data['reward_float'] = reward_list[history_len - 1:]
        data['n_state_float'] = n_state_list[history_len - 1:]
        whole_data[int(k)] = data

    with open(items_path, 'r') as fp:
        item_embed = json.load(fp)

    user = np.load(users_path, allow_pickle=True)
    new_mat = user[:, 1:]

    user[:, 1:] = minmax_scale(new_mat, axis=0)

    paddle = 15 * [1]

    user_embed = {}
    for row in user:
        new_row = list(row[1:])
        new_row.extend(paddle)
        user_embed[row[0]] = new_row

    return pd.DataFrame.from_dict(whole_data), item_embed, user_embed


# In[ ]:


# to predict an item’s feedback that the user never rates before


class Simulator(object):
    def __init__(self, alpha=0.3, sigma=0.9):
        self.alpha = alpha
        self.sigma = sigma
        # self.init_state = self.reset()
        # self.current_state = self.init_state
        # self.rewards, self.group_sizes, self.avg_states, self.avg_actions = self.avg_group()

    def state_module(self, user, embedding, item_index):
        item_mat = []
        state = []
        user_item = []

        width = len(item_index)

        for i in item_index:
            item_mat.append(embedding[str(i)])

        for i in range(width):
            for j in range(i + 1, width):
                state.append(np.multiply(item_mat[i], item_mat[j]))
            user_item.append(np.multiply(item_mat[i], user))

        state.extend(user_item)
        return state

    def state_module_item(self, embedding, item_index):
        item_mat = []

        width = len(item_index)

        for i in item_index:
            item_mat.append(embedding[str(i)])

        for i in range(width):
            for j in range(i + 1, width):
                item_mat.append(np.multiply(item_mat[i], item_mat[j]))

        return item_mat

    def reset(self, user_idx, user_embed):
        # mat = self.state_module(user_embed, item_embed, data.loc[user_idx]['state_float'][4]) this one is for user-item embedding
        mat = self.state_module_item(item_embed, whole_data.loc[user_idx]['state_float'][4])
        init_state = np.array(mat).reshape((15, 19))
        self.current_state = init_state
        return init_state

    def step(self, action, user_idx):
        actions = np.array(item_embed[action])
        simulate_rewards = self.simulate_reward((self.current_state.reshape((1, 15 * 19)),
                                                 actions.reshape((1, 1 * 19))), user_idx)

        actions = actions.reshape(1, 1 * 19)
        for i, r in enumerate(simulate_rewards):  # if simulate_reward>0, then change the state
            if r > 0:
                # self.current_state.append(action[i])

                tmp = np.append(self.current_state, actions[i].reshape((1, 19)), axis=0)
                tmp = np.delete(tmp, 0, axis=0)
                # self.current_state = tmp[np.newaxis, :]
                self.current_state = tmp
        return simulate_rewards, self.current_state

    def simulate_reward(self, pair, user_idx):

        """use the average result to calculate simulated reward.
        Args:
            pair (tuple): <state, action> pair
        Returns:
            simulated reward for the pair.
        """

        probability = []
        denominator = 0.
        max_prob = 0.
        result = 0.
        simulate_rewards = ""
        new_data = ((whole_data.loc[user_idx]).to_frame()).T
        # calculate simulated reward in normal way
        for idx, row in new_data.iterrows():
            state_values = row['state_float']
            action_values = row['action_float']
            length = len(action_values)
            for i in range(4, length):
                item_mat = pair[0][0][0:5 * 19]
                curr_embed = {key: item_embed[str(key)] for key in state_values[i]}
                curr_state = np.array(list(curr_embed.values())).reshape(19 * 5, 1)
                curr_embed1 = {action_values[i]: item_embed[str(action_values[i])]}
                curr_action = np.array(list(curr_embed1.values())).reshape(19, 1)

                numerator = self.alpha * (
                        np.dot(item_mat, curr_state)[0] / (np.linalg.norm(item_mat, 2) * np.linalg.norm(curr_state, 2))
                ) + (1 - self.alpha) * (
                                    np.dot(pair[1], curr_action)[0] / (
                                        np.linalg.norm(pair[1], 2) * np.linalg.norm(curr_action, 2))
                            )
                probability.append(numerator)
                denominator += numerator
        probability /= denominator
        simulate_rewards = [new_data.loc[user_idx]['reward_float'][int(np.argmax(probability))]]

        return simulate_rewards


# In[ ]:


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        for idx, row in data.iterrows():
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

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()
        self.count = 0


# In[ ]:


class Actor(object):
    """policy function approximator"""

    def __init__(self, sess, s_dim, a_dim, batch_size, output_size, weights_len, tau, learning_rate, scope="actor"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.batch_size = batch_size
        self.output_size = output_size
        self.weights_len = weights_len
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # estimator actor network
            self.state, self.action_weights, self.len_seq = self._build_net("estimator_actor")
            self.network_params = tf.trainable_variables()
            # self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='estimator_actor')

            # target actor network
            self.target_state, self.target_action_weights, self.target_len_seq = self._build_net("target_actor")
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]
            # self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')

            # operator for periodically updating target network with estimator network weights
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]
            self.hard_update_target_network_params = [
                self.target_network_params[i].assign(
                    self.network_params[i]
                ) for i in range(len(self.target_network_params))
            ]

            self.a_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
            self.params_gradients = list(
                map(
                    lambda x: tf.div(x, self.batch_size * self.a_dim),
                    tf.gradients(tf.reshape(self.action_weights, [self.batch_size, self.a_dim]),
                                 self.network_params, -self.a_gradient)
                )
            )
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(self.params_gradients, self.network_params)
            )
            self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    @staticmethod
    def cli_value(x, v):
        x = tf.cast(x, tf.int64)
        y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
        return tf.where(tf.greater(x, y), x, y)

    def _gather_last_output(self, data, seq_lens):
        this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype=tf.int64)
        tmp_end = tf.map_fn(lambda x: self.cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
        indices = tf.stack([this_range, tmp_end], axis=1)
        return tf.gather_nd(data, indices)

    def _build_net(self, scope):
        """build the tensorflow graph"""
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
            state_ = tf.reshape(state, [-1, self.weights_len, int(self.s_dim / self.weights_len)])
            len_seq = tf.placeholder(tf.int32, [None])
            cell = tf.nn.rnn_cell.GRUCell(self.output_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.initializers.random_normal(),
                                          bias_initializer=tf.zeros_initializer())
            outputs, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=len_seq)
            outputs1 = self._gather_last_output(outputs, len_seq)

            layer1 = tf.layers.Dense(64, activation=tf.nn.relu)(outputs1)
            layer2 = tf.layers.Dense(32, activation=tf.nn.relu)(layer1)
            outputs = tf.layers.Dense(self.output_size, activation=tf.nn.tanh)(layer2)

        return state, outputs, len_seq

    def train(self, state, a_gradient, len_seq):
        self.sess.run(self.optimizer, feed_dict={self.state: state, self.a_gradient: a_gradient, self.len_seq: len_seq})

    def predict(self, state, len_seq):
        return self.sess.run(self.action_weights, feed_dict={self.state: state, self.len_seq: len_seq})

    def predict_target(self, state, len_seq):
        return self.sess.run(self.target_action_weights, feed_dict={self.target_state: state,
                                                                    self.target_len_seq: len_seq})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


# In[ ]:


class Critic(object):
    """value function approximator"""

    def __init__(self, sess, s_dim, a_dim, num_actor_vars, weights_len, gamma, tau, learning_rate, scope="critic"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.num_actor_vars = num_actor_vars
        self.weights_len = weights_len
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # estimator critic network
            self.state, self.action, self.q_value, self.len_seq = self._build_net("estimator_critic")
            # self.network_params = tf.trainable_variables()[self.num_actor_vars:]
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="estimator_critic")

            # target critic network
            self.target_state, self.target_action, self.target_q_value, self.target_len_seq = self._build_net(
                "target_critic")
            # self.target_network_params = tf.trainable_variables()[(len(self.network_params) + self.num_actor_vars):]
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_critic")

            # operator for periodically updating target network with estimator network weights
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]
            self.hard_update_target_network_params = [
                self.target_network_params[i].assgin(
                    self.network_params[i]
                ) for i in range(len(self.target_network_params))
            ]

            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
            self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.q_value))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.a_gradient = tf.gradients(self.q_value, self.action)

    @staticmethod
    def cli_value(x, v):
        x = tf.cast(x, tf.int64)
        y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
        return tf.where(tf.greater(x, y), x, y)

    def _gather_last_output(self, data, seq_lens):
        this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype=tf.int64)
        tmp_end = tf.map_fn(lambda x: self.cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
        indices = tf.stack([this_range, tmp_end], axis=1)
        return tf.gather_nd(data, indices)

    def _build_net(self, scope):
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
            state_ = tf.reshape(state, [-1, self.weights_len, int(self.s_dim / self.weights_len)])
            action = tf.placeholder(tf.float32, [None, self.a_dim], "action")
            len_seq = tf.placeholder(tf.int32, [None])
            cell = tf.nn.rnn_cell.GRUCell(self.weights_len,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.initializers.random_normal(),
                                          bias_initializer=tf.zeros_initializer()
                                          )
            out_state, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=len_seq)
            out_state = self._gather_last_output(out_state, len_seq)

            inputs = tf.concat([out_state, action], axis=-1)
            layer1 = tf.layers.Dense(32, activation=tf.nn.relu)(inputs)
            layer2 = tf.layers.Dense(16, activation=tf.nn.relu)(layer1)
            q_value = tf.layers.Dense(1)(layer2)
            return state, action, q_value, len_seq

    def train(self, state, action, predicted_q_value, len_seq):
        return self.sess.run([self.q_value, self.loss, self.optimizer], feed_dict={
            self.state: state,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.len_seq: len_seq
        })

    def predict(self, state, action, len_seq):
        return self.sess.run(self.q_value, feed_dict={self.state: state, self.action: action, self.len_seq: len_seq})

    def predict_target(self, state, action, len_seq):
        return self.sess.run(self.target_q_value, feed_dict={self.target_state: state, self.target_action: action,
                                                             self.target_len_seq: len_seq})

    def action_gradients(self, state, action, len_seq):
        return self.sess.run(self.a_gradient, feed_dict={self.state: state, self.action: action, self.len_seq: len_seq})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)


# In[ ]:


class OUNoise:
    """noise for action"""

    def __init__(self, a_dim, mu=0, theta=0.5, sigma=0.2):
        self.a_dim = a_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.a_dim) * self.mu  # a_dim=19*num_action
        self.reset()

    def reset(self):
        self.state = np.ones(self.a_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(len(x))
        self.state = x + dx
        return self.state


# state representation module
# require user has same dimension with items (both 1*k dimension)


def gene_actions(item_space, weight_batch):
    """use output of actor network to calculate action list
    Args:
        item_space: recall items, dict: id: embedding
        weight_batch: actor network outputs
    Returns:
        recommendation list
    """
    item_ids = list(item_space.keys())
    item_weights = list(item_space.values())
    max_ids = list()
    for weight in weight_batch:
        score = np.dot(item_weights, weight)
        idx = np.argmax(score)
        max_ids.append(item_ids[idx])
    return max_ids


def gene_action(item_space, weight):
    item_ids = list(item_space.keys())
    item_weights = list(item_space.values())
    score = np.dot(item_weights, weight)
    idx = np.argmax(score)
    return item_ids[idx]


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("reward", episode_reward)
    episode_max_q = tf.Variable(0.)
    tf.summary.scalar("max_q_value", episode_max_q)
    critic_loss = tf.Variable(0.)
    tf.summary.scalar("critic_loss", critic_loss)

    summary_vars = [episode_reward, episode_max_q, critic_loss]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars


def state_mod(state):
    sample = []
    for row in state:
        # print(row)
        x, y = row.shape
        for i in range(x):
            for j in range(i + 1, x):
                row = np.vstack([row, np.multiply(row[i], row[j])])
        sample.append(row)

    return np.array(sample)


def learn_from_batch(replay_buffer, batch_size, actor, critic, item_space, action_len, s_dim, a_dim):
    seq_len = np.array([a_dim / action_len], dtype=np.int32)
    if replay_buffer.size() < batch_size:
        pass
    samples = replay_buffer.sample_batch(batch_size)
    # print(samples)
    state_batch = []
    action_batch = []
    reward_batch = []
    n_state_batch = []
    for row in samples:
        state_batch.append(row[0])
        action_batch.append(row[1])
        reward_batch.append(row[2])
        n_state_batch.append(row[3])
    state_batch = np.array(state_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    n_state_batch = np.array(n_state_batch)

    # calculate predicted q value
    new_state = np.concatenate(state_mod(state_batch), axis=0)
    new_state = np.reshape(new_state, [-1, s_dim])

    action_weights = actor.predict_target(new_state, seq_len)
    n_action_batch = gene_actions(item_space, action_weights)

    new_action_batch = []
    for idx in n_action_batch:
        new_action_batch.append(item_space[idx])
    new_action_batch = np.array(new_action_batch)

    n_new_state = np.concatenate(state_mod(n_state_batch), axis=0)
    n_new_state = np.reshape(n_new_state, (-1, s_dim))
    target_q_batch = critic.predict_target(n_new_state, new_action_batch.reshape((-1, a_dim)), len_seq=seq_len)

    y_batch = []
    for i in range(batch_size):
        y_batch.append(reward_batch[i] + critic.gamma * target_q_batch[i])
    y_batch = np.array(y_batch)
    y_batch = np.concatenate(y_batch, axis=0)

    # train critic
    q_value, critic_loss, _ = critic.train(new_state, action_batch, np.reshape(y_batch, (batch_size, 1)), seq_len)

    # train actor
    action_weight_batch_for_gradients = actor.predict(new_state, seq_len)
    action_batch_for_gradients = gene_actions(item_space, action_weight_batch_for_gradients)

    action_batch_gra = []
    for idx in action_batch_for_gradients:
        action_batch_gra.append(item_space[idx])
    action_batch_gra = np.array(action_batch_gra)

    a_gradient_batch = critic.action_gradients(new_state, action_batch_gra.reshape((-1, a_dim)), seq_len)

    actor.train(new_state, a_gradient_batch[0], seq_len)

    # update target networks
    actor.update_target_network()
    critic.update_target_network()

    return np.amax(q_value), critic_loss


# In[ ]:


def train_test(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args, replay_buffer):
    start = time.time()
    # set up summary operators
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # initialize target network weights
    actor.hard_update_target_network()
    critic.hard_update_target_network()

    for i in range(int(args['max_episodes'])):
        ep_reward = 0.
        ep_q_value = 0.
        loss = 0.
        item_space = item_embed
        epoch_num = 0
        for idx in list(data.index):
            epoch_num += 1
            user_space = user_embed[idx]
            if epoch_num == 1:
                state = env.reset(user_idx=idx, user_embed=user_space)
            # update average parameters every 10 episodes
            for j in range(args['max_episodes_len']):
                weight = actor.predict(np.reshape(state, [-1, s_dim]),
                                       [int(args['embedding'])]) + exploration_noise.noise()
                action = gene_actions(item_space, weight)
                reward, n_state = env.step(action[0], idx)

                # print(state,action,reward,n_state)

                replay_buffer.add(state[:args['state_item_num']],
                                  item_embed[str(action[0])],  # need more work
                                  np.array(reward),
                                  np.vstack((n_state[:args['state_item_num'] - 1], n_state[-2:-1])))

                ep_reward += reward[0]
                ep_q_value_, critic_loss = learn_from_batch(replay_buffer, args['batch_size'], actor, critic,
                                                            item_space,
                                                            args['action_item_num'], s_dim, a_dim)
                ep_q_value += ep_q_value_
                loss += critic_loss
                state = n_state
                '''
                if (j + 1) % 50 == 0:
                    print("=========={0} episode of {1} round of {2}-th user: reward {3} loss {4}=========".format(
                        i, j, idx, ep_reward, critic_loss))
                '''

        print('======{0}-th episode, {1} total reward======'.format(i, ep_reward))
        summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: ep_reward,
                                                       summary_vars[1]: ep_q_value,
                                                       summary_vars[2]: loss})
        writer.add_summary(summary_str, i)

    writer.close()
    end = time.time()
    time_consuming = end - start
    # print('Training time = ', time_consuming)

    ep_reward_ = 0.
    ep_q_value = 0.
    loss = 0.
    item_space = item_embed_test
    for idx in list(test_data.index):
        user_space = user_embed_test[idx]
        # update average parameters every 10 episodes
        for j in range(args['test_episodes_len']):
            weight = actor.predict(np.reshape(state, [-1, s_dim]), [int(args['embedding'])]) + exploration_noise.noise()
            action = gene_actions(item_space, weight)
            reward, n_state = env.step(action[0], idx)

            # print(state,action,reward,n_state)
            '''
            replay_buffer.add(state[:args['state_item_num']],
                              item_space[str(action[0])],  
                              np.array(reward),
                              np.vstack((n_state[:args['state_item_num']-1], n_state[-2:-1])))
  
            ep_q_value_, critic_loss = learn_from_batch(replay_buffer, args['batch_size'], actor, critic, item_space,
                                                        args['action_item_num'], s_dim, a_dim)
            ep_q_value += ep_q_value_
            loss += critic_loss
            '''
            ep_reward_ += reward[0]
            state = n_state

    saver = tf.train.Saver()
    model_name = 'drr-1st.ckpt'
    log_dir = '/content/gdrive/My Drive/drr/logs/{}'.format(model_name)
    saver.save(sess, log_dir, write_meta_graph=False)

    return ep_reward_, time_consuming


# In[ ]:


def main(args):
    tf.reset_default_graph()

    with tf.Session() as sess:
        # simulated environment
        env = Simulator()

        # initialize replay memory
        replay_buffer = ReplayBuffer(int(args['buffer_size']))

        s_dim = int(args['embedding']) * (
                    int(args['state_item_num']) + int(comb(int(args['state_item_num']), 2)))  ### need more work here
        a_dim = int(args['embedding']) * int(args['action_item_num'])

        actor = Actor(sess, s_dim, a_dim,
                      int(args['batch_size']), int(args['embedding']),
                      int(args['action_item_num']), float(args['tau']),
                      float(args['actor_lr']))

        critic = Critic(sess, s_dim, a_dim,
                        actor.get_num_trainable_vars(), int(args['action_item_num']), float(args['gamma']),
                        float(args['tau']), float(args['critic_lr']))

        exploration_noise = OUNoise(a_dim)

        test_reward, time_consuming = train_test(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args,
                                                 replay_buffer)

    return time_consuming, test_reward


# In[ ]:


args = {}
args['embedding'] = 19
args['state_item_num'] = 5
args['action_item_num'] = 1  # currently only generate 1 item
args['actor_lr'] = 0.0001
args['critic_lr'] = 0.001
args['gamma'] = 0.9
args['tau'] = 0.001
args['buffer_size'] = 1000000
args['batch_size'] = 1
args['max_episodes'] = 10
args['max_episodes_len'] = 50
args['test_episodes_len'] = 10
args['summary_dir'] = '/content/gdrive/My Drive/drr/results'
args['summary_dir_test'] = '/content/gdrive/My Drive/drr/test_results'

clusters_accuracy = {}

# num_of_clusters = [2, 4, 16, 64, 128, 256, 512, len(user_train)]
num_of_clusters = [len(user_train)]

for num in num_of_clusters:
    cluster_data(num)

    path1 = '/content/gdrive/My Drive/drr/user_movies.json'
    path2 = '/content/gdrive/My Drive/drr/user_train.npy'
    path3 = '/content/gdrive/My Drive/drr/movie_items.json'
    path4 = '/content/gdrive/My Drive/drr/user_test.npy'
    path5 = '/content/gdrive/My Drive/drr/test_user_movies.json'
    path6 = '/content/gdrive/My Drive/drr/whole_user_movies.json'
    history_len = 5
    test_data, item_embed_test, user_embed_test = process_data(path5, path4, path3, history_len)
    test_data = test_data.T
    data, item_embed, user_embed = process_data(path1, path2, path3, history_len)
    data = data.T
    whole_data, item_embed, user_embed = process_data(path6, path2, path3, history_len)
    whole_data = whole_data.T # 行代表用户数,列为4,其中每列又为一个向量

    time_consuming, test_reward = main(args)
    print('total training time {0}, total test reward {1}'.format(time_consuming, test_reward))

    clusters_accuracy[num] = [time_consuming, test_reward]

with open('/content/gdrive/My Drive/drr/episode50session10.json', 'w') as fp:
    json.dump(clusters_accuracy, fp)


with open('/content/gdrive/My Drive/drr/time_accuracy.json', 'r') as fp:
    a = json.load(fp)
a


# 64 reward 930.5

# In[ ]:


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")