#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/11 21:41                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import datetime
import numpy as np
import os
import tensorflow as tf
from collections import deque
from itertools import chain
import sys

from rl_recmd import sample_new_states
from rl_recmd import save_results
from rl_recmd import initialize_environment
from rl_recmd import compute_average_reward
from rl_recmd import sample_new_states_for_train


def construct_placeholder():
    global disp_action_feature, Xs_clicked, news_size, user_size, disp_indices
    global disp_2d_split_user_ind, history_order_indices, history_user_indices

    disp_action_feature = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])
    Xs_clicked = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])

    news_size = tf.placeholder(dtype=tf.int64, shape=[])
    user_size = tf.placeholder(dtype=tf.int64, shape=[])
    disp_indices = tf.placeholder(dtype=tf.int64, shape=[None, 2])

    disp_2d_split_user_ind = tf.placeholder(dtype=tf.int64, shape=[None])

    history_order_indices = tf.placeholder(dtype=tf.int64, shape=[None])
    history_user_indices = tf.placeholder(dtype=tf.int64, shape=[None])


def construct_p():
    global u_disp, p_disp, agg_variables, position_weight, user_states

    denseshape = [user_size, news_size]

    # (1) history feature --- net ---> clicked_feature
    # (1) construct cumulative history
    click_history = [[] for _ in range(_weighted_dim)]
    position_weight = [[] for _ in range(_weighted_dim)]
    for ii in range(_weighted_dim):
        position_weight[ii] = tf.get_variable('p_w'+str(ii), [_band_size], initializer=tf.constant_initializer(0.0001))
        # np.arange(id_cnt) 当前用户上一时刻的点击的item的数量
        position_weight_values = tf.gather(position_weight[ii], history_order_indices)
        weighted_feature = tf.multiply(Xs_clicked, tf.reshape(position_weight_values, [-1, 1]))  # Xs_clicked: section by _f_dim
        click_history[ii] = tf.segment_sum(weighted_feature, history_user_indices)
    user_states = tf.concat(click_history, axis=1)#假设4个user，则shape为 4*80

    disp_history_feature = tf.gather(user_states, disp_2d_split_user_ind)

    # (4) combine features
    concat_disp_features = tf.reshape(tf.concat([disp_history_feature, disp_action_feature], axis=1), [-1, _f_dim * _weighted_dim + _f_dim])

    # (5) compute utility
    n1 = 256
    y1 = tf.layers.dense(concat_disp_features, n1, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_E3_sd))  #, kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4))
    y1 = tf.nn.elu(y1)
    # create layer2

    n2 = 32
    y2 = tf.layers.dense(y1, n2, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_E3_sd))
    y2 = tf.nn.elu(y2)

    # output layer
    u_disp = tf.layers.dense(y2, 1, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_E3_sd))

    # (5)
    u_disp = tf.reshape(u_disp, [-1])
    exp_u_disp = tf.exp(u_disp)
    sum_exp_disp = tf.segment_sum(exp_u_disp, disp_2d_split_user_ind) + float(np.exp(_noclick_weight))#相当于公式中的正的部分
    scatter_sum_exp_disp = tf.gather(sum_exp_disp, disp_2d_split_user_ind)
    p_disp = tf.div(exp_u_disp, scatter_sum_exp_disp)#权重

    agg_variables = tf.global_variables()


def construct_reward():
    # 1. reward
    Reward_r = tf.segment_sum(tf.multiply(u_disp, p_disp), disp_2d_split_user_ind)
    Reward_1 = tf.segment_sum(p_disp, disp_2d_split_user_ind)

    reward_feed_dict = {Xs_clicked: [], history_order_indices: [], history_user_indices: [], disp_2d_split_user_ind: [], disp_action_feature:[]}

    trans_p = tf.reshape(p_disp, [-1, _k])

    return Reward_r, Reward_1, reward_feed_dict, trans_p


# 这里定义Q function还有对应的loss。定义的时候，假设同时处理一个batch的数据，所以稍微复杂一点。
# 输出_k个Q function，_k个loss，_k个train op
def construct_Q_and_loss():
    global current_action_space, y_label
    global action_states, action_space_mean, action_space_std
    global action_k_id

    # placeholder
    current_action_space = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])
    action_space_mean = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])
    action_space_std = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])
    y_label = tf.placeholder(dtype=tf.float32, shape=[None])

    # (1) action states - offline的实验受到数据的限制，所以加了一个mean和std。
    # 做online实验没有数据的限制，我觉得这部分的input可以直接不要
    action_states = tf.concat([action_space_mean, action_space_std], axis=1)

    # (2) action id - 推荐的items的id。online的版本可以直接输入feature vector而不是id。
    # 换言之，可以忽略action_k_id，直接把（3）的action_k_feature_gather定义成placeholder，输入item features。
    action_k_id = [[] for _ in range(_k)]
    for ii in range(_k):
        action_k_id[ii] = tf.placeholder(dtype=tf.int64, shape=[None])#id的占位符
    # (3) action features
    action_k_feature_gather = [[] for _ in range(_k)]
    for ii in range(_k):
        # action_k_feature_gather[ii] 代表推荐的第ii个item的feature。（总共推荐_k个item）
        action_k_feature_gather[ii] = tf.gather(current_action_space, action_k_id[ii])
    # (4) user states
    # 还要构造一个user states，这个定义了在construct_p()里面。

    # 定义Q: input：（user_states, action_states, action_feature）
    concate_input_k = [[] for _ in range(_k)]
    action_feature_list = []
    q_value_k = [[] for _ in range(_k)]
    loss_k = [[] for _ in range(_k)]
    opt_k = [[] for _ in range(_k)]
    train_variable_k = [[] for _ in range(_k)]
    train_op_k = [[] for _ in range(_k)]

    for ii in range(_k):
        # 把（user_states, action_states, action_feature）三种vectors concat在一起，作为input。（online版本可以忽略action_states）
        # 注意：action_feature_list是一步步变大的，从length=1 到 length=_k
        action_feature_list.append(action_k_feature_gather[ii])
        # user states 定义在construct_p()里面
        concate_input_k[ii] = tf.concat([user_states, action_states] + action_feature_list, axis=1)
        # 实际并没有进行reshape，这里只是为了明确dim，否则做dense layer的时候会报错。
        concate_input_k[ii] = tf.reshape(concate_input_k[ii], [-1, _weighted_dim * _f_dim + 2 * _f_dim + int(ii+1) * _f_dim])

        current_variables = tf.trainable_variables()
        # q_value_k[ii]: 构造paper里面提到的Q^j, where j=1,...,_k
        with tf.variable_scope('Q'+str(ii)+'-function', reuse=False):
            q_y1 = tf.layers.dense(concate_input_k[ii], 256, activation=tf.nn.elu, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))
            q_y2 = tf.layers.dense(q_y1, 32, activation=tf.nn.elu, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))
            q_value_k[ii] = tf.layers.dense(q_y2, 1, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))
        q_value_k[ii] = tf.reshape(q_value_k[ii], [-1])

        # loss
        # y_label为reward
        loss_k[ii] = tf.reduce_mean(tf.squared_difference(q_value_k[ii], y_label))#y_label就是env算出来的reward
        opt_k[ii] = tf.train.AdamOptimizer(learning_rate=_lr)

        train_variable_k[ii] = list(set(tf.trainable_variables()) - set(current_variables))
        train_op_k[ii] = opt_k[ii].minimize(loss_k[ii], var_list=train_variable_k[ii])

    sess.run(tf.variables_initializer(list(set(tf.global_variables()) - set(agg_variables))))

    q_feed_dict = {current_action_space: [], action_space_mean: [], action_space_std: [], Xs_clicked: [], history_order_indices: [], history_user_indices: [], y_label: []}
    for ii in range(_k):
        q_feed_dict[action_k_id[ii]] = []

    return q_feed_dict, loss_k, train_op_k


# 这里定义argmax Q 和 max_Q
def construct_max_Q():

    global all_action_user_indices, all_action_tensor_indices, all_action_tensor_shape, action_count, action_space_count, all_action_id

    all_action_user_indices = tf.placeholder(dtype=tf.int64, shape=[None])
    all_action_tensor_indices = tf.placeholder(dtype=tf.int64, shape=[None, 2])
    all_action_tensor_shape = tf.placeholder(dtype=tf.int64, shape=[2])
    action_count = tf.placeholder(dtype=tf.int64, shape=[None])
    action_space_count = tf.placeholder(dtype=tf.int64, shape=[None])


    # online版本：建议直接把all_action_feature_gather作为placeholder，输入所有可以选的items的features
    all_action_id = tf.placeholder(dtype=tf.int64, shape=[None])
    all_action_feature_gather = tf.gather(current_action_space, all_action_id)

    user_states_scatter = tf.gather(user_states, all_action_user_indices)
    # online版本：建议：action states可以不需要
    action_states_scatter = tf.gather(action_states, all_action_user_indices)

    max_action_feature_list = []
    max_action_k = [[] for _ in range(_k)]
    max_action_feature_k = [[] for _ in range(_k)]
    to_avoid_repeat_tensor = tf.zeros(tf.cast(all_action_tensor_shape, tf.int32))
    for ii in range(_k):
        # 构造Q_j的input（notation: j就是ii）
        # 注意：max_action_feature_list是逐步变大，从length=0到length=_k - 1
        concate_input = tf.concat([user_states_scatter, action_states_scatter]+max_action_feature_list+[all_action_feature_gather], axis=1)
        concate_input = tf.reshape(concate_input, [-1, _weighted_dim * _f_dim + 2 * _f_dim + _f_dim * int(ii+1)])
        # 把所有action（所有items）对应的Q_j values算出来
        # 注意：Q_j要reuse在construct_Q_and_loss()中定义的Q_j
        with tf.variable_scope('Q'+str(ii)+'-function', reuse=True):
            q1_y1 = tf.layers.dense(concate_input, 256, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))
            q1_y2 = tf.layers.dense(q1_y1, 32, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))
            q_value_all = tf.layers.dense(q1_y2, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))

        q_value_all = tf.reshape(q_value_all, [-1])
        q1_tensor = tf.sparse_to_dense(all_action_tensor_indices, all_action_tensor_shape, q_value_all, default_value=-1000000000.0)
        q1_tensor += to_avoid_repeat_tensor

        # max_action_k[ii]: 得到Q_j值最优的item。作为推荐的第j个item。
        max_action_k[ii] = tf.argmax(q1_tensor, axis=1)
        # to_avoid_repeat_tensor是为了避免重复推荐一样的item。因为我们希望得到_k个不同的items。
        to_avoid_repeat_tensor += tf.one_hot(max_action_k[ii], tf.cast(all_action_tensor_shape[1], tf.int32), on_value=-1000000000.0, off_value=0.0)
        # 下面几行是把max_action_k[ii]变成真实的item id。这部分应该根据自己的实验数据格式来决定如何写。
        max_action_k[ii] = tf.add(max_action_k[ii], action_count)
        max_action_k[ii] = tf.gather(all_action_id, max_action_k[ii])
        max_action_feature_k[ii] = tf.gather(current_action_space, max_action_k[ii])
        max_action_k[ii] = max_action_k[ii] - action_space_count

        # 把argmax Q_j得到的最优item的特征存起来，作为下一个Q_{j+1}的input
        max_action_feature_k_scatter = tf.gather(max_action_feature_k[ii], all_action_user_indices)
        max_action_feature_list.append(max_action_feature_k_scatter)

    max_q_value = tf.segment_max(q_value_all, all_action_user_indices)

    max_action = tf.stack(max_action_k, axis=1)
    max_action_disp_features = tf.concat(max_action_feature_k, axis=1)
    max_action_disp_features = tf.reshape(max_action_disp_features, [-1, _f_dim])

    max_q_feed_dict = {all_action_id: [], all_action_user_indices: [], all_action_tensor_indices: [], all_action_tensor_shape: [],
                         current_action_space: [], Xs_clicked: [], history_order_indices: [], history_user_indices: [],
                       action_count: [], action_space_count: [], action_space_mean: [], action_space_std: []}

    return max_q_value, max_action, max_action_disp_features, max_q_feed_dict


def form_init_Q_feed_dict(user_set, states_id):

    # states_feature = np.zeros([len(user_set), _f_dim], dtype=np.float32)
    states_feature = [[] for _ in range(len(user_set))]
    history_order = [[] for _ in range(len(user_set))]  # np.zeros([len(user_set)], dtype=np.int64)
    history_user = [[] for _ in range(len(user_set))]  # np.arange(len(user_set), dtype=np.int64)

    for uu in range(len(user_set)):
        user = user_set[uu]

        if len(states_id[uu]) == 0:
            states_feature[uu] = np.zeros([1, _f_dim], dtype=np.float32).tolist()
            history_order[uu].append(0)
            history_user[uu].append(uu)
        else:
            states_feature[uu] = deque(maxlen=_band_size)
            for idd in states_id[uu]:#用户选择的item
                states_feature[uu].append(feature_space[user][idd])

            states_feature[uu] = list(states_feature[uu])
            id_cnt = len(states_feature[uu])
            history_order[uu] = np.arange(id_cnt, dtype=np.int64).tolist()
            history_user[uu] = list(uu * np.ones(id_cnt, dtype=np.int64))

    states_feature = list(chain.from_iterable(states_feature))
    history_order = list(chain.from_iterable(history_order))
    history_user = list(chain.from_iterable(history_user))

    return states_feature, history_order, history_user


def test_during_training(current_best_reward):
    # initialize empty states
    sim_vali_user = _users_to_test
    states = [[] for _ in range(len(sim_vali_user))]
    sim_u_reward = {}

    for t in range(_time_horizon):
        action_mean_tr, action_std_tr, action_user_indice_tr, action_tensor_indice_tr, action_shape_tr, \
        action_space_tr, states_tr, history_order_tr, history_user_tr, action_cnt_tr, \
        action_space_cnt_tr, action_id_tr = form_max_q_feed_dict(sim_vali_user, states)

        max_q_feed_dict[all_action_id] = action_id_tr
        max_q_feed_dict[all_action_user_indices] = action_user_indice_tr
        max_q_feed_dict[all_action_tensor_indices] = action_tensor_indice_tr
        max_q_feed_dict[all_action_tensor_shape] = action_shape_tr
        max_q_feed_dict[current_action_space] = action_space_tr
        max_q_feed_dict[action_space_mean] = action_mean_tr
        max_q_feed_dict[action_space_std] = action_std_tr
        max_q_feed_dict[Xs_clicked] = states_tr
        max_q_feed_dict[history_order_indices] = history_order_tr
        max_q_feed_dict[history_user_indices] = history_user_tr
        max_q_feed_dict[action_count] = action_cnt_tr
        max_q_feed_dict[action_space_count] = action_space_cnt_tr
        # 1. find best recommend action
        best_action = sess.run([max_action, max_action_disp_features], feed_dict=max_q_feed_dict)
        best_action[0] = best_action[0].tolist()
        # 2. compute reward
        disp_2d_split_user = np.kron(np.arange(len(sim_vali_user)), np.ones(_k))
        reward_feed_dict[Xs_clicked] = states_tr
        reward_feed_dict[history_order_indices] = history_order_tr
        reward_feed_dict[history_user_indices] = history_user_tr
        reward_feed_dict[disp_2d_split_user_ind] = disp_2d_split_user
        reward_feed_dict[disp_action_feature] = best_action[1]
        [reward_u, transition_p] = sess.run([u_disp, trans_p], feed_dict=reward_feed_dict)
        reward_u = np.reshape(reward_u, [-1, _k])
        # 5. sample reward and new states
        sim_vali_user, states, sim_u_reward = sample_new_states(sim_vali_user, states, transition_p, reward_u, sim_u_reward, feature_space,  best_action[0], _k)
        if len(sim_vali_user) == 0:
            break

    _, _, _, _, new_best_reward = compute_average_reward(_users_to_test, sim_u_reward, current_best_reward)
    return new_best_reward


def repeated_test(_repeated_best_reward, n_test):
    sim_user_reward = [{} for _ in range(n_test)]
    user_avg_reward = [[] for _ in range(n_test)]
    click_rate = [[] for _ in range(n_test)]
    mean_user_avg_reward = np.zeros(n_test)
    mean_click_rate = np.zeros(n_test)

    for i_th in range(n_test):
        sim_vali_user = _users_to_test
        states = [[] for _ in range(len(sim_vali_user))]

        for t in range(_time_horizon):
            action_mean_tr, action_std_tr, action_user_indice_tr, action_tensor_indice_tr, action_shape_tr, \
            action_space_tr, states_tr, history_order_tr, history_user_tr, action_cnt_tr, action_space_cnt_tr, action_id_tr = form_max_q_feed_dict(sim_vali_user, states)

            max_q_feed_dict[all_action_id] = action_id_tr
            max_q_feed_dict[all_action_user_indices] = action_user_indice_tr
            max_q_feed_dict[all_action_tensor_indices] = action_tensor_indice_tr
            max_q_feed_dict[all_action_tensor_shape] = action_shape_tr
            max_q_feed_dict[current_action_space] = action_space_tr
            max_q_feed_dict[action_space_mean] = action_mean_tr
            max_q_feed_dict[action_space_std] = action_std_tr
            max_q_feed_dict[Xs_clicked] = states_tr
            max_q_feed_dict[history_order_indices] = history_order_tr
            max_q_feed_dict[history_user_indices] = history_user_tr
            max_q_feed_dict[action_count] = action_cnt_tr
            max_q_feed_dict[action_space_count] = action_space_cnt_tr
            # 1. find best recommend action
            best_action = sess.run([max_action, max_action_disp_features], feed_dict=max_q_feed_dict)
            best_action[0] = best_action[0].tolist()
            # 2. compute reward
            disp_2d_split_user = np.kron(np.arange(len(sim_vali_user)), np.ones(_k))
            reward_feed_dict[Xs_clicked] = states_tr
            reward_feed_dict[history_order_indices] = history_order_tr
            reward_feed_dict[history_user_indices] = history_user_tr
            reward_feed_dict[disp_2d_split_user_ind] = disp_2d_split_user
            reward_feed_dict[disp_action_feature] = best_action[1]
            [reward_u, transition_p] = sess.run([u_disp, trans_p], feed_dict=reward_feed_dict)
            reward_u = np.reshape(reward_u, [-1, _k])
            # 5. sample reward and new states
            sim_vali_user, states, sim_user_reward[i_th] = sample_new_states(sim_vali_user, states, transition_p, reward_u, sim_user_reward[i_th], feature_space,  best_action[0], _k)
            if len(sim_vali_user) == 0:
                break

        user_avg_reward[i_th], mean_user_avg_reward[i_th], click_rate[i_th], mean_click_rate[i_th], _ = compute_average_reward(_users_to_test, sim_user_reward[i_th], 100.0)
    if np.mean(mean_user_avg_reward) > _repeated_best_reward:
        print('new repeated best reward!!!!!!!!!!')
        _repeated_best_reward = np.mean(mean_user_avg_reward)
        save_path = os.path.join(vali_path, 'repeated_best-reward')
        saver.save(sess, save_path)
        filename = 'RL_recommend'+str(_k)+'_'+str(_noclick_weight)+'.pickle'
        save_results(_time_horizon, _users_to_test, sim_user_reward, user_avg_reward, mean_user_avg_reward, click_rate, mean_click_rate, filename)
    else:
        print(['mean, reward of all experiments:', np.mean(mean_user_avg_reward)])
        print(['std, reward of all experiments:', np.std(mean_user_avg_reward)])
        print(['mean, click rate of all experiments:', np.mean(mean_click_rate)])
        print(['std, click rate of all experiments:', np.std(mean_click_rate)])

    return _repeated_best_reward


def test_of_training(training_set):
    # initialize empty states
    sim_vali_user = training_set
    states = [[] for _ in range(len(sim_vali_user))]
    sim_u_reward = {}

    for t in range(_time_horizon):
        action_mean_tr, action_std_tr, action_user_indice_tr, action_tensor_indice_tr, action_shape_tr, \
        action_space_tr, states_tr, history_order_tr, history_user_tr, action_cnt_tr, action_space_cnt_tr, action_id_tr = form_max_q_feed_dict(sim_vali_user, states)
        max_q_feed_dict[all_action_id] = action_id_tr
        max_q_feed_dict[all_action_user_indices] = action_user_indice_tr
        max_q_feed_dict[all_action_tensor_indices] = action_tensor_indice_tr
        max_q_feed_dict[all_action_tensor_shape] = action_shape_tr
        max_q_feed_dict[current_action_space] = action_space_tr
        max_q_feed_dict[action_space_mean] = action_mean_tr
        max_q_feed_dict[action_space_std] = action_std_tr
        max_q_feed_dict[Xs_clicked] = states_tr
        max_q_feed_dict[history_order_indices] = history_order_tr
        max_q_feed_dict[history_user_indices] = history_user_tr
        max_q_feed_dict[action_count] = action_cnt_tr
        max_q_feed_dict[action_space_count] = action_space_cnt_tr
        # 1. find best recommend action
        best_action = sess.run([max_action, max_action_disp_features], feed_dict=max_q_feed_dict)
        best_action[0] = best_action[0].tolist()
        # 2. compute reward
        disp_2d_split_user = np.kron(np.arange(len(sim_vali_user)), np.ones(_k))
        reward_feed_dict[Xs_clicked] = states_tr
        reward_feed_dict[history_order_indices] = history_order_tr
        reward_feed_dict[history_user_indices] = history_user_tr
        reward_feed_dict[disp_2d_split_user_ind] = disp_2d_split_user
        reward_feed_dict[disp_action_feature] = best_action[1]
        [reward_u, transition_p] = sess.run([u_disp, trans_p], feed_dict=reward_feed_dict)
        reward_u = np.reshape(reward_u, [-1, _k])
        # 5. sample reward and new states
        sim_vali_user, states, sim_u_reward = sample_new_states(sim_vali_user, states, transition_p, reward_u, sim_u_reward, feature_space,  best_action[0], _k)
        if len(sim_vali_user) == 0:
            break

    _, _, _, _, n_best_reward = compute_average_reward(training_set, sim_u_reward, 100.0)
    return n_best_reward


def form_max_q_feed_dict(user_set, states_id):

    # states_feature = np.zeros([len(user_set), _f_dim], dtype=np.float32)

    states_feature = [[] for _ in range(len(user_set))]
    history_order = [[] for _ in range(len(user_set))]  # np.zeros([len(user_set)], dtype=np.int64)
    history_user = [[] for _ in range(len(user_set))]  # np.arange(len(user_set), dtype=np.int64)

    action_space = []

    # action_indicate_u = [[] for _ in range(len(user_set))]

    # action_indicate = []

    action_id = []

    action_user_indice = []
    action_tensor_indice = []

    max_act_size = 0

    candidate_action_mean = [[] for _ in range(len(user_set))]
    candidate_action_std = [[] for _ in range(len(user_set))]

    action_cnt = [0 for _ in range(len(user_set))]

    action_space_cnt = [0 for _ in range(len(user_set))]

    for uu in range(len(user_set)):
        user = user_set[uu]

        candidate_action = []

        if len(states_id[uu]) == 0:
            states_feature[uu] = np.zeros([1, _f_dim], dtype=np.float32).tolist()
            history_order[uu].append(0)
            history_user[uu].append(uu)
        else:
            states_feature[uu] = deque(maxlen=_band_size)
            for idd in states_id[uu]:
                states_feature[uu].append(feature_space[user][idd])

            states_feature[uu] = list(states_feature[uu])
            id_cnt = len(states_feature[uu])
            history_order[uu] = np.arange(id_cnt, dtype=np.int64).tolist()
            history_user[uu] = list(uu * np.ones(id_cnt, dtype=np.int64))

        action_candidate = np.array(list(set(np.arange(len(feature_space[user]))) - set(states_id[uu])))

        for idd in action_candidate:
            candidate_action.append(feature_space[user][idd])

        candidate_action_mean[uu] = np.mean(np.array(candidate_action), axis=0)
        candidate_action_std[uu] = np.std(np.array(candidate_action), axis=0)

        action_space_cnt[uu] = len(action_space)

        action_id_u = list(action_candidate + action_space_cnt[uu])
        action_id += action_id_u
        # all possible actions
        # action_indicate_u[uu] = list(chain.from_iterable(combinations(action_candidate + action_space_cnt[uu], _k)))
        # action_indicate += action_indicate_u[uu]

        action_cnt[uu] = len(action_id_u)
        action_user_indice += [uu for _ in range(action_cnt[uu])]

        if action_cnt[uu] == 0:
            print('action_cnt 0')
            print(action_candidate)
            print(action_candidate + action_space_cnt[uu])
            print(states_id[uu])
        max_act_size = max(max_act_size, action_cnt[uu])
        # action_user_indice += [uu for _ in range(action_cnt[uu])]
        action_tensor_indice += map(lambda x: [uu, x], np.arange(action_cnt[uu]))

        # action space
        action_space += feature_space[user]

    action_cnt = np.cumsum(action_cnt)
    action_cnt = [0] + list(action_cnt[:-1])

    action_shape = [len(user_set), max_act_size]

    states_feature = list(chain.from_iterable(states_feature))
    history_order = list(chain.from_iterable(history_order))
    history_user = list(chain.from_iterable(history_user))

    return candidate_action_mean, candidate_action_std, action_user_indice, action_tensor_indice, action_shape, \
           action_space, states_feature, history_order, history_user, action_cnt, action_space_cnt, action_id


def form_loss_feed_dict(user_set, states_id, action_id):

    states_feature = [[] for _ in range(len(user_set))]
    history_order = [[] for _ in range(len(user_set))]  # np.zeros([len(user_set)], dtype=np.int64)
    history_user = [[] for _ in range(len(user_set))]  # np.arange(len(user_set), dtype=np.int64)

    action_space = []

    candidate_action_mean = [[] for _ in range(len(user_set))]
    candidate_action_std = [[] for _ in range(len(user_set))]

    action_ids_k = [[] for _ in range(_k)]

    for uu in range(len(user_set)):
        user = user_set[uu]
        candidate_action = []

        if len(states_id[uu]) == 0:
            states_feature[uu] = np.zeros([1, _f_dim], dtype=np.float32).tolist()
            history_order[uu].append(0)
            history_user[uu].append(uu)
        else:
            states_feature[uu] = deque(maxlen=_band_size)
            for idd in states_id[uu]:
                states_feature[uu].append(feature_space[user][idd])

            states_feature[uu] = list(states_feature[uu])
            id_cnt = len(states_feature[uu])
            history_order[uu] = np.arange(id_cnt, dtype=np.int64).tolist()
            history_user[uu] = list(uu * np.ones(id_cnt, dtype=np.int64))

        action_candidate = np.array(list(set(np.arange(len(feature_space[user]))) - set(states_id[uu])))
        for idd in action_candidate:
            candidate_action.append(feature_space[user][idd])

        candidate_action_mean[uu] = np.mean(np.array(candidate_action), axis=0)
        candidate_action_std[uu] = np.std(np.array(candidate_action), axis=0)

        # all actions
        # action_indicate += map(lambda x: x + len(action_space), action_id[uu])
        for jj in range(_k):
            action_ids_k[jj].append(action_id[uu][jj] + len(action_space))

        # action space
        action_space += feature_space[user]

    states_feature = list(chain.from_iterable(states_feature))
    history_order = list(chain.from_iterable(history_order))
    history_user = list(chain.from_iterable(history_user))

    return action_ids_k, action_space, states_feature, history_order, history_user, candidate_action_mean, candidate_action_std


#定义一些变量
_f_dim, _k, iterations, _noclick_weight, _band_size, _weighted_dim, train_user, vali_user, test_user, feature_space, \
_users_to_test, _time_horizon, num_test, sim_user_reward, user_avg_reward, click_rate, \
mean_user_avg_reward, mean_click_rate = initialize_environment(sys.argv)

_E3_sd = 1e-3
_candidate_sd = 1e-6
_q_sd = 1e-2
_lr = 0.001
_num_thread = 10

current_best_reward = 0.0

print(['Adam', _lr, 'E3_sd', _E3_sd, '_candidate_sd', _candidate_sd, '_q_sd', _q_sd, 'band_size', _band_size])


# 下面9行：输入一个之前学出来的用户模型作为simulator。
construct_placeholder()
construct_p()
sess = tf.Session()
sess.run(tf.variables_initializer(agg_variables))

vali_path = './saved_models/E3_agg_split1/'
saver = tf.train.Saver(max_to_keep=None)
save_path = os.path.join(vali_path, 'convert_best-loss')
saver.restore(sess, save_path)
# 计算reward
Reward_r, Reward_1, reward_feed_dict, trans_p = construct_reward()

# 构造 Q，max Q，train op
q_feed_dict, loss_k, train_op_k = construct_Q_and_loss()
max_q_value, max_action, max_action_disp_features, max_q_feed_dict = construct_max_Q()

_sample_batch_size = 100
_training_batch_size = 50

# 超参_gamma：可调。代表decaying factor。
_gamma = 0.98

data_size = _time_horizon * _sample_batch_size + 100

_is_finite = 1

vali_path = './saved_models/agg_rl_k'+str(_k)+'u'+str(_noclick_weight)+'/'
saver = tf.train.Saver(max_to_keep=None)

# (1) first step: initialize Q as the expected rwd function
# current_best_reward = 0.0
training_user = np.random.choice(train_user, 1000, replace=False).tolist()
training_user_copy = np.array(training_user).tolist()
states = [[] for _ in range(len(training_user))]

data_size_init = len(training_user) * _time_horizon + 100
data_collection = {'user': deque(maxlen=data_size_init), 'state': deque(maxlen=data_size_init),
                   'action': deque(maxlen=data_size_init), 'y': deque(maxlen=data_size_init)}

for t in range(_time_horizon):#100
    data_collection['state'].extend(states)
    data_collection['user'].extend(training_user)
    # prepare to feed max_Q
    states_tr, history_order_tr, history_user_tr = form_init_Q_feed_dict(training_user, states)

    # 1. sample random action，模仿推荐系统推荐的action
    # feature_space 保存的是和用户相关的sku的特征:其实就是sku的embedding向量
    random_action = [[] for _ in range(len(training_user))]
    random_action_feature = []
    # 给每个用户曝光10个item
    for u_i in range(len(training_user)):
        user_i = training_user[u_i]
        #从用户曝光过的sku中随机选k个sku作为推荐,模型推荐引擎的选择
        random_action[u_i] = np.random.choice(list(set(np.arange(len(feature_space[user_i])))-set(states[u_i])), _k, replace=False).tolist()
        random_action_feature += [feature_space[user_i][jj] for jj in random_action[u_i]]

    best_action = [random_action, random_action_feature]
    data_collection['action'].extend(best_action[0])

    # 2. compute expected immediate reward
    disp_2d_split_user = np.kron(np.arange(len(training_user)), np.ones(_k))
    # reward_feed_dict = {Xs_clicked: [], history_order_indices: [], history_user_indices: [], disp_2d_split_user_ind: [], disp_action_feature:[]}
    reward_feed_dict[Xs_clicked] = states_tr
    reward_feed_dict[history_order_indices] = history_order_tr#相当于是用户的点击次数
    reward_feed_dict[history_user_indices] = history_user_tr
    reward_feed_dict[disp_2d_split_user_ind] = disp_2d_split_user
    reward_feed_dict[disp_action_feature] = best_action[1]
    # 为exp操作前的reward*对应的权重
    # Reward_r = tf.segment_sum(tf.multiply(u_disp, p_disp), disp_2d_split_user_ind)
    [best_action_reward, transition_p] = sess.run([Reward_r, trans_p], feed_dict=reward_feed_dict)

    # 4. save to memory
    y_value = best_action_reward
    data_collection['y'].extend(y_value.tolist())#y存的是用户推荐引擎推荐的10个item，也即对用户曝光的10个item对应的reward

    # 5. sample new states
    remove_set = []
    for j in range(len(training_user)):
        if len(feature_space[training_user[j]]) - len(states[j]) <= _k+1:
            remove_set.append(j)

        disp_item = best_action[0][j]
        #transition_p[j, :]得到的是每个用户对k个item的权重
        no_click = [max(1.0 - np.sum(transition_p[j, :]), 0.0)]
        prob = np.array(transition_p[j, :].tolist()+no_click)
        #用户根据概率选action，此次曝光有可能选1个，有可能选n个,用户对此次曝光没兴趣，一个都没选的情况呢？
        prob = prob / float(prob.sum())
        #模拟用户的选择
        rand_choice = np.random.choice(disp_item + [-100], 1, p=prob)
        if rand_choice[0] != -100:
            states[j] += rand_choice.tolist()

    previous_size = len(training_user)
    states = [states[j] for j in range(previous_size) if j not in remove_set]
    training_user = [training_user[j] for j in range(previous_size) if j not in remove_set]

# START TRAINING for this batch of users
num_samples = len(data_collection['user'])
batch_iterations = int(np.ceil(num_samples * 5 / _training_batch_size))
for n in range(batch_iterations):
    batch_sample = np.random.choice(len(data_collection['user']), _training_batch_size, replace=False)
    states_batch = [data_collection['state'][c] for c in batch_sample]
    user_batch = [data_collection['user'][c] for c in batch_sample]
    action_batch = [data_collection['action'][c] for c in batch_sample]
    y_batch = [data_collection['y'][c] for c in batch_sample]

    action_ids_k_tr, action_space_tr, states_feature_tr, history_order_tr, history_user_tr,\
    action_mean_tr, action_std_tr = form_loss_feed_dict(user_batch, states_batch, action_batch)

    # action_space_tr:action的特征,所有用户的feature_space的矩阵。
    # action_space_mean:当前用户所有可选的action向量按axis=0的均值
    q_feed_dict[current_action_space] = action_space_tr
    q_feed_dict[action_space_mean] = action_mean_tr
    q_feed_dict[action_space_std] = action_std_tr
    q_feed_dict[Xs_clicked] = states_feature_tr
    q_feed_dict[history_order_indices] = history_order_tr#当前用户点击的item的膈俞
    q_feed_dict[history_user_indices] = history_user_tr#当前用户的uid
    q_feed_dict[y_label] = y_batch

    # action_ids_k[jj].append(action_id[uu][jj] + len(action_space))
    # 所有用户u当前的k个曝光sku+上一个用户相关的action的数量
    for ii in range(_k):
        q_feed_dict[action_k_id[ii]] = action_ids_k_tr[ii]

    loss_val = sess.run(train_op_k+loss_k, feed_dict=q_feed_dict)

    if np.mod(n, 250) == 0:
        loss_val = list(loss_val[-_k:])
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print_loss = ' '
        for kkk in range(_k):
            print_loss += ' %.5f,'
        print(('%s: init itr(%d), training loss:'+print_loss) % tuple([log_time, n]+loss_val))

print('finish init iteration')
save_path = os.path.join(vali_path, 'init-q')
saver.save(sess, save_path)
# TEST
# initialize empty states
current_best_reward = test_during_training(current_best_reward)

for itr in range(iterations):
    training_start_point = (itr * _sample_batch_size) % 25000
    training_user = train_user[training_start_point: training_start_point + _sample_batch_size]
    training_user_copy = np.array(training_user).tolist()
    # initialize empty states
    states = [[] for _ in range(len(training_user))]
    data_collection = {'user': deque(maxlen=data_size), 'state': deque(maxlen=data_size), 'action': deque(maxlen=data_size), 'y': deque(maxlen=data_size)}

    for t in range(_time_horizon):
        data_collection['state'].extend(states)
        data_collection['user'].extend(training_user)
        # prepare to feed max_Q
        action_mean_tr, action_std_tr, action_user_indice_tr, action_tensor_indice_tr, action_shape_tr, \
        action_space_tr, states_tr, history_order_tr, history_user_tr, action_cnt_tr, action_space_cnt_tr, action_id_tr = form_max_q_feed_dict(training_user, states)

        max_q_feed_dict[all_action_id] = action_id_tr
        max_q_feed_dict[all_action_user_indices] = action_user_indice_tr
        max_q_feed_dict[all_action_tensor_indices] = action_tensor_indice_tr
        max_q_feed_dict[all_action_tensor_shape] = action_shape_tr
        max_q_feed_dict[current_action_space] = action_space_tr
        max_q_feed_dict[action_space_mean] = action_mean_tr
        max_q_feed_dict[action_space_std] = action_std_tr
        max_q_feed_dict[Xs_clicked] = states_tr
        max_q_feed_dict[history_order_indices] = history_order_tr
        max_q_feed_dict[history_user_indices] = history_user_tr
        max_q_feed_dict[action_count] = action_cnt_tr
        max_q_feed_dict[action_space_count] = action_space_cnt_tr
        # 1. find best recommend action
        best_action = sess.run([max_action, max_action_disp_features], feed_dict=max_q_feed_dict)
        best_action[0] = best_action[0].tolist()
        data_collection['action'].extend(best_action[0])
        # 2. compute reward
        disp_2d_split_user = np.kron(np.arange(len(training_user)), np.ones(_k))
        reward_feed_dict[Xs_clicked] = states_tr
        reward_feed_dict[history_order_indices] = history_order_tr
        reward_feed_dict[history_user_indices] = history_user_tr
        reward_feed_dict[disp_2d_split_user_ind] = disp_2d_split_user
        reward_feed_dict[disp_action_feature] = best_action[1]
        [reward_u, transition_p] = sess.run([u_disp, trans_p], feed_dict=reward_feed_dict)
        reward_u = np.reshape(reward_u, [-1, _k])
        # 3. sample new states
        states, training_user, old_training_user, old__new_states, sampled_reward, remove_set = sample_new_states_for_train(training_user, states, transition_p, reward_u, feature_space, best_action[0], _k)

        # 4. compute one-step delayed reward
        action_mean_tr, action_std_tr, action_user_indice_tr, action_tensor_indice_tr, action_shape_tr, \
        action_space_tr, states_tr, history_order_tr, history_user_tr, action_cnt_tr, action_space_cnt_tr, action_id_tr = form_max_q_feed_dict(old_training_user, old__new_states)
        max_q_feed_dict[all_action_id] = action_id_tr
        max_q_feed_dict[all_action_user_indices] = action_user_indice_tr
        max_q_feed_dict[all_action_tensor_indices] = action_tensor_indice_tr
        max_q_feed_dict[all_action_tensor_shape] = action_shape_tr
        max_q_feed_dict[current_action_space] = action_space_tr
        max_q_feed_dict[action_space_mean] = action_mean_tr
        max_q_feed_dict[action_space_std] = action_std_tr
        max_q_feed_dict[Xs_clicked] = states_tr
        max_q_feed_dict[history_order_indices] = history_order_tr
        max_q_feed_dict[history_user_indices] = history_user_tr
        max_q_feed_dict[action_count] = action_cnt_tr
        max_q_feed_dict[action_space_count] = action_space_cnt_tr
        max_q_val = sess.run(max_q_value, feed_dict=max_q_feed_dict)

        # 4. save to memory
        # if _is_finite:
        #     max_q_val[remove_set] = 0.0
        y_value = sampled_reward + _gamma * max_q_val
        data_collection['y'].extend(y_value.tolist())

    # START TRAINING for this batch of users
    num_samples = len(data_collection['user'])
    batch_iterations = int(np.ceil(num_samples * 5 / _training_batch_size))
    for n in range(batch_iterations):
        batch_sample = np.random.choice(len(data_collection['user']), _training_batch_size, replace=False)
        states_batch = [data_collection['state'][c] for c in batch_sample]
        user_batch = [data_collection['user'][c] for c in batch_sample]
        action_batch = [data_collection['action'][c] for c in batch_sample]
        y_batch = [data_collection['y'][c] for c in batch_sample]

        action_ids_k_tr, action_space_tr, states_feature_tr, history_order_tr, history_user_tr, action_mean_tr, action_std_tr = form_loss_feed_dict(user_batch, states_batch, action_batch)

        q_feed_dict[current_action_space] = action_space_tr
        q_feed_dict[action_space_mean] = action_mean_tr
        q_feed_dict[action_space_std] = action_std_tr
        q_feed_dict[Xs_clicked] = states_feature_tr
        q_feed_dict[history_order_indices] = history_order_tr
        q_feed_dict[history_user_indices] = history_user_tr
        q_feed_dict[y_label] = y_batch

        for ii in range(_k):
            q_feed_dict[action_k_id[ii]] = action_ids_k_tr[ii]

        loss_val = sess.run(train_op_k+loss_k, feed_dict=q_feed_dict)
        loss_val = list(loss_val[-_k:])

        if np.mod(n, 50) == 0:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print_loss = ''
            for kkk in range(_k):
                print_loss += ' %.5f,'
            print(('%s: itr(%d, %d), training loss:'+print_loss) % tuple([log_time, itr, n]+loss_val))

    print('finish iteration %d' % itr)
    # TEST
    new_reward = test_during_training(current_best_reward)
    if new_reward > current_best_reward:
        # repeated_test()
        save_path = os.path.join(vali_path, 'best-reward')
        saver.save(sess, save_path)
        current_best_reward = new_reward

save_path = os.path.join(vali_path, 'best-reward')
saver.restore(sess, save_path)
_ = repeated_test(0.0, num_test)