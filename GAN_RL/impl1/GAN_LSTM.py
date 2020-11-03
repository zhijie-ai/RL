#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/11 21:35                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import datetime
import numpy as np
import tensorflow as tf
import threading
import os
import sys

def format_data():
    size_user = 5000
    behavior_filename = "data/e3_behavior_v1.txt"
    category_filename = "data/e3_news_category_v1.txt"
    feature_filename = "data/e3_user_news_feature_v1.txt"
    splitter = '/t'

    # 1. data_behavior
    max_disp_size = 0

    data_behavior = [[] for x in range(size_user)]

    fd = open(behavior_filename)
    for row in fd:
        row = row.split()[0]
        row = row.split(splitter)
        u_id = int(row[0])
        time_stamp = int(row[1])
        disp_list = map(int, row[2].split(','))
        max_disp_size = max(max_disp_size, len(disp_list))
        pick_list = map(int, row[3].split(','))
        data_behavior[u_id].append([time_stamp, disp_list, pick_list])
    fd.close()
    k = 10

    for i in range(len(data_behavior)):
        data_behavior[i] = sorted(data_behavior[i], key=lambda x : x[0])

    # 1.1 click and disp behavior

    # 2. category
    max_category = 0
    min_category = 100
    movie_category = {}
    fd = open(category_filename)
    for row in fd:
        row = row.split()[0]
        row = row.split(splitter)
        news_id = int(row[0])
        news_cat = map(int, row[1].split(','))
        news_cat = list(np.array(news_cat)+1)   # let category start from 1. leave category 0 for non-clicking
        movie_category[news_id] = news_cat
        max_category = max(max_category, max(news_cat))
        min_category = min(min_category, min(news_cat))
    fd.close()

    KK = max_category + 1
    if max(movie_category.keys())!=len(movie_category.keys()):
        print('warning: news category wrong!')
        exit()
    else:
        size_movie = len(movie_category.keys()) + 1

    # movie_category[0] = [0]

    # 3. feature
    user_news_feature = {}
    fd = open(feature_filename)
    for row in fd:
        row = row.split()[0]
        row = row.split(splitter)
        key = 'u'+row[0]+'n'+row[1]
        user_news_feature[key] = map(float, row[2].split(','))
    fd.close()

    # 4. save synthetic data
    data_parameter = [KK, k, size_user, size_movie]

    # another set of data, <--- this is what we finally use
    # 这部分主要是为了把news_id按照每个user从0开始排序
    data_click = [[] for _ in range(size_user)]
    data_disp = [[] for _ in range(size_user)]
    data_time = np.zeros(size_user, dtype=np.int)
    data_news_cnt = np.zeros(size_user, dtype=np.int)
    feature = [[] for _ in range(size_user)]

    for user in range(len(data_behavior)):
        news_dict = {}
        click_t = 0
        for event in range(len(data_behavior[user])):
            disp_list = data_behavior[user][event][1]
            pick_list = data_behavior[user][event][2]
            for id in disp_list:
                if id not in news_dict:
                    news_dict[id] = len(news_dict)  # for each user, news id start from 0

            for id in pick_list:
                data_click[user].append([click_t, news_dict[id]])
                for idd in disp_list:
                    data_disp[user].append([click_t, news_dict[idd]])
                click_t += 1  # splitter a event with 2 clickings to 2 events

        data_time[user] = click_t
        data_news_cnt[user] = len(news_dict)

        feature[user] = np.zeros([data_news_cnt[user], 20])
        for id in news_dict:
            key = 'u'+str(user)+'n'+str(id)
            feature[user][news_dict[id]] = user_news_feature[key]
        feature[user] = feature[user].tolist()
    return data_click, data_disp, feature, data_time, data_news_cnt, data_parameter

def data_perform(user_set):
    max_news_per_user = 0  # max number of news(per user)
    max_time = 0

    size_user = len(user_set)  # 这个batch的用户数目

    disp_tensor_indices = []  # [user_id(对于这个batch，从0开始数), 时间（对于每个user，从0开始数，整数），news_id（对于每个user，从0开始数）] SparseTensor的indices
                              # 本来只是sparsetensor的话，除了时间以外，别的不需要从0开始数，但是因为算precision的时候需要把它变成densetensor，所以要从0开始，让tensor size更小

    disp_tensor_indices_split = []  # [user_id, 时间], 长度和顺序都和disp_tensor_indices一样，只是不要第三列[news_id]
                                    # 这个数据如果不好准备，可以直接在tensorflow里面用tf.split操作

    click_tensor_indices = []  # 和disp_tensor_indices一样, 但是这里的第三列只包含被click了的news_id

    u_feature = []

    click_sub_index = []  # 因为click_tensor_indices是disp_tensor_indices的一个子集，找一下对应的sublist index.. 在tf.gather有用到

    # （1） 找出最长的时间。data_time[u]表示用户u对应的时间长度
    for u in user_set:
        max_time = max(max_time, data_time[u])

    user_time_dense = np.zeros([size_user, max_time], dtype=np.float32)
    click_feature = np.zeros([max_time, size_user, _f_dim])  # 这个作为LSTM的input
    for u_idx in range(size_user):
        u = user_set[u_idx]

        click_tensor_indices_tmp = []
        disp_tensor_indices_tmp = []

        for x in data_click[u]:
            t, click_id = x  # 这个表示：用户u在时间t点击了新闻click_id
            click_feature[t][u_idx] = feature[u][click_id]  # 把对应的特征collect起来，作为LSTM的input
            click_tensor_indices_tmp.append([u_idx, t, click_id])
            user_time_dense[u_idx, t] = 1.0

        click_tensor_indices = click_tensor_indices + click_tensor_indices_tmp

        for x in data_disp[u]:
            t, disp_id = x
            disp_tensor_indices_tmp.append([u_idx, t, disp_id])
            disp_tensor_indices_split.append([u_idx, t])
            u_feature.append(feature[u][disp_id])  # feature

        click_sub_index_tmp = map(lambda x: disp_tensor_indices_tmp.index(x), click_tensor_indices_tmp)  #找sublist index
        click_sub_index += map(lambda x: x+len(disp_tensor_indices), click_sub_index_tmp)

        disp_tensor_indices = disp_tensor_indices + disp_tensor_indices_tmp
        max_news_per_user = max(max_news_per_user, data_news_cnt[u])

    return size_user, max_time, max_news_per_user, \
           disp_tensor_indices, disp_tensor_indices_split, np.array(u_feature), click_feature, click_sub_index, \
           click_tensor_indices, user_time_dense

# 下面三个function是tf算子的定义，主要看这个。
def construct_placeholder():
    # 这个作为LSTM的input, 输入的是有点击的news的特征。shape是：[time, user=batch, _f_dim]。 _f_dim是特征dim
    clicked_feature = tf.placeholder(tf.float32, (None, None, _f_dim))

    # 这个是跟这个batch相关的所有特征，包括点击和未点击。
    u_current_feature = tf.placeholder(tf.float32, shape=[None, _f_dim])

    # [user_id(对于这个batch，从0开始数), 时间（对于每个user，从0开始数，整数），news_id（对于每个user，从0开始数）] SparseTensor的indices
    # 本来除了时间以外，别的不需要从0开始数，但是因为算precision的时候需要把它变成densetensor，所以要从0开始，让tensor size更小
    disp_tensor_indices = tf.placeholder(dtype=tf.int64, shape=[None, 3])

    # [user_id, 时间], 长度和顺序都和disp_tensor_indices一样，只是不要第三列[news_id]
    # 这个数据如果不好准备，可以直接在tensorflow里面用tf.split对disp_tensor_indices操作得到
    disp_tensor_indices_split = tf.placeholder(dtype=tf.int64, shape=[None, 2])

    # 和disp_tensor_indices一样, 但是这里的第三列只包含被click了的news_id
    clk_tensor_indices = tf.placeholder(dtype=tf.int64, shape=[None, 3])
    # 等价于np.ones(len(clk_tensor_indices))
    clk_tensor_val = tf.placeholder(dtype=tf.float32, shape=[None])

    # 等价于disp_tensor_indices.index(clk_tensor_indices)
    click_sublist_index = tf.placeholder(dtype=tf.int64, shape=[None])

    # 一个shape为[size of user(batch size), max length of time] 的dense matrix。(u,t)=1 if 有user u在时间t的数据。（不同的user时间长度不一样）
    ut_dense = tf.placeholder(dtype=tf.float32, shape=[None, None])

    # max_time。最长的时间长度（在这个batch的用户中）
    time = tf.placeholder(dtype=tf.int64)
    # max num of news per user。最多的news数目（在这个batch的用户中）
    news_size = tf.placeholder(dtype=tf.int64)

    return clicked_feature, u_current_feature, disp_tensor_indices_split, disp_tensor_indices, clk_tensor_indices, clk_tensor_val, click_sublist_index, ut_dense, time, news_size


def construct_graph(clicked_feature, u_current_feature, disp_tensor_indices_split, disp_tensor_indices, clk_tensor_indices, clk_tensor_val, click_sublist_index, ut_dense, time, news_size):
    batch_size = tf.shape(clicked_feature)[1]  # batch size of RNN input. 这里batch_size = number of users

    # construct lstm network
    cell = tf.contrib.rnn.BasicLSTMCell(_RNN_HIDDEN, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, clicked_feature, initial_state=initial_state, time_major=True)
    # rnn_outputs: shape --> (time, user=batch, rnn_hidden)

    # (1) move forward one-step
    u_history_feature = tf.concat([tf.zeros([1, batch_size, _RNN_HIDDEN], dtype=tf.float32), rnn_outputs], 0)
    # (2) transpose to reshape --> (user=batch, time, rnn_hidden)
    u_history_feature = tf.transpose(u_history_feature, perm=[1, 0, 2])  #

    # concat history feature and current feature
    u_history_feature_gather = tf.gather_nd(u_history_feature, disp_tensor_indices_split)
    combine_feature = tf.concat([u_history_feature_gather, u_current_feature], axis=1)
    combine_feature = tf.reshape(combine_feature, [-1, _RNN_HIDDEN + _f_dim]) # indicate size

    # neural net，可以调
    n1 = 256
    y1 = tf.layers.dense(combine_feature, n1, activation=tf.nn.elu, trainable=True,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=_combine_sd))
    n2 = 128
    y2 = tf.layers.dense(y1, n2, activation=tf.nn.elu, trainable=True,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=_combine_sd))
    n3 = 32
    y3 = tf.layers.dense(y2, n3, activation=tf.nn.elu, trainable=True,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=_combine_sd))
    u_net = tf.layers.dense(y3, 1, trainable=True,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=_combine_sd))
    u_net = tf.reshape(u_net, [-1])

    # construct loss function
    denseshape = tf.concat([tf.cast(tf.reshape(batch_size, [-1]), tf.int64), tf.reshape(time, [-1]), tf.reshape(news_size, [-1])], 0)  # (user, time, news_size)
    click_u_tensor = tf.SparseTensor(clk_tensor_indices, tf.gather(u_net, click_sublist_index), dense_shape=denseshape)
    disp_exp_u_tensor = tf.SparseTensor(disp_tensor_indices, tf.exp(u_net), dense_shape=denseshape)
    disp_sum_exp_u_tensor = tf.sparse_reduce_sum(disp_exp_u_tensor, axis=2)
    sum_click_u_tensor = tf.sparse_reduce_sum(click_u_tensor, axis=2)
    loss_tmp = - sum_click_u_tensor + tf.log(disp_sum_exp_u_tensor + 1)  # (user, time) loss
    loss_sum = tf.reduce_sum(tf.multiply(ut_dense, loss_tmp))
    event_cnt = tf.reduce_sum(ut_dense)
    loss = loss_sum / event_cnt

    # compute precision
    dense_exp_disp_util = tf.sparse_tensor_to_dense(disp_exp_u_tensor, default_value=0.0)

    click_tensor = tf.sparse_to_dense(clk_tensor_indices, denseshape, clk_tensor_val, default_value=0.0)
    argmax_click = tf.argmax(click_tensor, axis=2)
    argmax_disp = tf.argmax(dense_exp_disp_util, axis=2)

    top_2_disp = tf.nn.top_k(dense_exp_disp_util, k=2, sorted=False)[1]
    argmax_compare = tf.cast(tf.equal(argmax_click, argmax_disp), tf.float32)
    precision_1_sum = tf.reduce_sum(tf.multiply(ut_dense, argmax_compare))
    tmpshape = tf.concat([tf.cast(tf.reshape(batch_size, [-1]), tf.int64), tf.reshape(time, [-1]), tf.constant([1], dtype=tf.int64)], 0)
    top2_compare = tf.reduce_sum(tf.cast(tf.equal(tf.reshape(argmax_click, tmpshape), tf.cast(top_2_disp, tf.int64)), tf.float32), axis=2)
    precision_2_sum = tf.reduce_sum(tf.multiply(ut_dense, top2_compare))
    precision_1 = precision_1_sum / event_cnt
    precision_2 = precision_2_sum / event_cnt

    return loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt

def construct_model(clicked_feature, u_current_feature, disp_tensor_indices_split, disp_tensor_indices, clk_tensor_indices, clk_tensor_val, click_sublist_index, ut_dense, time, news_size, is_training, is_batch_norm, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt = construct_graph(clicked_feature, u_current_feature, disp_tensor_indices_split, disp_tensor_indices, clk_tensor_indices, clk_tensor_val, click_sublist_index, ut_dense, time, news_size)

    if is_training:
        opt = tf.train.AdamOptimizer(learning_rate=_lr)
        if is_batch_norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = opt.minimize(loss)
        else:
            train_op = opt.minimize(loss)

        return train_op, loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt
    else:
        return loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt


# 后面这几个都是一些操作性的function，你们可以不用看，对于alps基本用不上。
def prepare_vali_data(num_sets, v_user):
    vali_thread_u = [[] for _ in range(num_sets)]
    size_user_v = [[] for _ in range(num_sets)]
    max_time_v = [[] for _ in range(num_sets)]
    news_cnt_short_v = [[] for _ in range(num_sets)]
    u_t_dispid_v = [[] for _ in range(num_sets)]
    u_t_dispid_split_ut_v = [[] for _ in range(num_sets)]
    u_t_dispid_feature_v = [[] for _ in range(num_sets)]
    click_feature_v = [[] for _ in range(num_sets)]
    click_sub_index_v = [[] for _ in range(num_sets)]
    u_t_clickid_v = [[] for _ in range(num_sets)]
    ut_dense_v = [[] for _ in range(num_sets)]
    for ii in range(len(v_user)):
        vali_thread_u[ii % num_sets].append(v_user[ii])
    for ii in range(num_sets):
        size_user_v[ii], max_time_v[ii], news_cnt_short_v[ii], u_t_dispid_v[ii],\
        u_t_dispid_split_ut_v[ii], u_t_dispid_feature_v[ii], click_feature_v[ii], \
        click_sub_index_v[ii], u_t_clickid_v[ii], ut_dense_v[ii] = data_perform(vali_thread_u[ii])
    return vali_thread_u, size_user_v, max_time_v, news_cnt_short_v, u_t_dispid_v, u_t_dispid_split_ut_v,\
           u_t_dispid_feature_v, click_feature_v, click_sub_index_v, u_t_clickid_v, ut_dense_v

def multithread_compute_vali():
    global vali_sum, vali_cnt

    vali_sum = [0.0, 0.0, 0.0]
    vali_cnt = 0
    threads = []
    for ii in range(_num_thread):
        thread = threading.Thread(target=vali_eval, args=(1, ii))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return vali_sum[0]/vali_cnt, vali_sum[1]/vali_cnt, vali_sum[2]/vali_cnt


lock = threading.Lock()

def vali_eval(weird, ii):
    global vali_sum, vali_cnt
    vali_thread_eval = sess.run([train_loss_sum, train_prec1_sum, train_prec2_sum, train_event_cnt], feed_dict={clicked_feature: click_feature_vali[ii],
                                   u_current_feature: u_t_dispid_feature_vali[ii],
                                   disp_tensor_indices_split: np.array(u_t_dispid_split_ut_vali[ii], dtype=np.int64),
                                   disp_tensor_indices: np.array(u_t_dispid_vali[ii], dtype=np.int64),
                                   clk_tensor_indices: np.array(u_t_clickid_vali[ii], dtype=np.int64),
                                   clk_tensor_val: np.ones(len(u_t_clickid_vali[ii]), dtype=np.float32),
                                   click_sublist_index: np.array(click_sub_index_vali[ii], dtype=np.int64),
                                   ut_dense: ut_dense_vali[ii],
                                   time: max_time_vali[ii],
                                   news_size: news_cnt_short_vali[ii]
                                   })
    lock.acquire()
    vali_sum[0] += vali_thread_eval[0]
    vali_sum[1] += vali_thread_eval[1]
    vali_sum[2] += vali_thread_eval[2]
    vali_cnt += vali_thread_eval[3]
    lock.release()

def multithread_compute_test():
    global test_sum, test_cnt

    num_sets = 4 * _num_thread

    thread_dist = [[] for _ in range(_num_thread)]
    for ii in range(num_sets):
        thread_dist[ii % _num_thread].append(ii)

    test_sum = [0.0, 0.0, 0.0]
    test_cnt = 0
    threads = []
    for ii in range(_num_thread):
        thread = threading.Thread(target=test_eval, args=(1, thread_dist[ii]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return test_sum[0]/test_cnt, test_sum[1]/test_cnt, test_sum[2]/test_cnt


def test_eval(weird, thread_dist):
    global test_sum, test_cnt
    test_thread_eval = [0.0, 0.0, 0.0]
    test_thread_cnt = 0
    for ii in thread_dist:
        test_set_eval = sess.run([train_loss_sum, train_prec1_sum, train_prec2_sum, train_event_cnt], feed_dict={clicked_feature: click_feature_test[ii],
                                       u_current_feature: u_t_dispid_feature_test[ii],
                                       disp_tensor_indices_split: np.array(u_t_dispid_split_ut_test[ii], dtype=np.int64),
                                       disp_tensor_indices: np.array(u_t_dispid_test[ii], dtype=np.int64),
                                       clk_tensor_indices: np.array(u_t_clickid_test[ii], dtype=np.int64),
                                       clk_tensor_val: np.ones(len(u_t_clickid_test[ii]), dtype=np.float32),
                                       click_sublist_index: np.array(click_sub_index_test[ii], dtype=np.int64),
                                       ut_dense: ut_dense_test[ii],
                                       time: max_time_test[ii],
                                       news_size: news_cnt_short_test[ii]
                                       })
        test_thread_eval[0] += test_set_eval[0]
        test_thread_eval[1] += test_set_eval[1]
        test_thread_eval[2] += test_set_eval[2]
        test_thread_cnt += test_set_eval[3]

    lock.acquire()
    test_sum[0] += test_thread_eval[0]
    test_sum[1] += test_thread_eval[1]
    test_sum[2] += test_thread_eval[2]
    test_cnt += test_thread_cnt
    lock.release()

log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("%s, start" % log_time)
data_click, data_disp, feature, data_time, data_news_cnt, data_parameter = format_data()
log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("%s, load data completed" % log_time)

split_k = sys.argv[1]

_f_dim = 20

_RNN_HIDDEN = 20

_combine_sd = 1e-1
_lr = 0.001

_num_thread = 10

print(['Adam', _lr, 'combine_sd', _combine_sd])

clicked_feature, u_current_feature, disp_tensor_indices_split, disp_tensor_indices, clk_tensor_indices, \
clk_tensor_val, click_sublist_index, ut_dense, time, news_size = construct_placeholder()

train_opt, train_loss, train_prec1, train_prec2, train_loss_sum, train_prec1_sum, train_prec2_sum, train_event_cnt \
    = construct_model(clicked_feature, u_current_feature, disp_tensor_indices_split, disp_tensor_indices, clk_tensor_indices, clk_tensor_val, click_sublist_index, ut_dense, time, news_size, is_training=True, is_batch_norm=False, reuse=False)

train_user = np.arange(2500)
vali_user = np.arange(2500, 2500+2500/4)
test_user = np.arange(2500+2500/4, 5000)

batch = 50

sess = tf.Session()
sess.run(tf.global_variables_initializer())

iterations = 100000

# prepare validation data
log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("%s, start prepare vali data" % log_time)
vali_thread_user, size_user_vali, max_time_vali, news_cnt_short_vali, u_t_dispid_vali, \
u_t_dispid_split_ut_vali, u_t_dispid_feature_vali, click_feature_vali, click_sub_index_vali, \
u_t_clickid_vali, ut_dense_vali = prepare_vali_data(_num_thread, vali_user)
log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("%s, prepare vali data complete" % log_time)

best_metric = [100000.0, 0.0, 0.0]

vali_path = './saved_models/LSTM_split'+str(split_k)+'/'
saver = tf.train.Saver(max_to_keep=None)

for i in range(iterations):

    training_start_point = (i * batch) % 25000
    training_user = train_user[training_start_point : training_start_point + batch]
    if i == 0:
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, start prepare train data" % log_time)

    size_user_tr, max_time_tr, news_cnt_short_tr, u_t_dispid_tr, u_t_dispid_split_ut_tr, \
    u_t_dispid_feature_tr, click_feature_tr, click_sub_index_tr, u_t_clickid_tr, ut_dense_tr = data_perform(training_user)

    if i == 0:
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, prepare train data completed" % log_time)
        print("%s, start first iteration training" % log_time)

    sess.run(train_opt, feed_dict={clicked_feature: click_feature_tr,
                                   u_current_feature: u_t_dispid_feature_tr,
                                   disp_tensor_indices_split: np.array(u_t_dispid_split_ut_tr, dtype=np.int64),
                                   disp_tensor_indices: np.array(u_t_dispid_tr, dtype=np.int64),
                                   clk_tensor_indices: np.array(u_t_clickid_tr, dtype=np.int64),
                                   clk_tensor_val: np.ones(len(u_t_clickid_tr), dtype=np.float32),
                                   click_sublist_index: np.array(click_sub_index_tr, dtype=np.int64),
                                   ut_dense: ut_dense_tr,
                                   time: max_time_tr,
                                   news_size: news_cnt_short_tr
                                   })
    if i == 0:
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, first iteration training complete" % log_time)

    if np.mod(i, 250) == 0:
        loss_prc = sess.run([train_loss, train_prec1, train_prec2], feed_dict={clicked_feature: click_feature_tr,
                                   u_current_feature: u_t_dispid_feature_tr,
                                   disp_tensor_indices_split: np.array(u_t_dispid_split_ut_tr, dtype=np.int64),
                                   disp_tensor_indices: np.array(u_t_dispid_tr, dtype=np.int64),
                                   clk_tensor_indices: np.array(u_t_clickid_tr, dtype=np.int64),
                                   clk_tensor_val: np.ones(len(u_t_clickid_tr), dtype=np.float32),
                                   click_sublist_index: np.array(click_sub_index_tr, dtype=np.int64),
                                   ut_dense: ut_dense_tr,
                                   time: max_time_tr,
                                   news_size: news_cnt_short_tr
                                   })
        if i == 0:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s, start first iteration validation" % log_time)
        vali_loss_prc = multithread_compute_vali()
        if i == 0:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s, first iteration validation complete" % log_time)

        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s: itr%d, training: %.5f, %.5f, %.5f, vali: %.5f, %.5f, %.5f" %
              (log_time, i, loss_prc[0], loss_prc[1], loss_prc[2], vali_loss_prc[0], vali_loss_prc[1], vali_loss_prc[2]))

        if vali_loss_prc[0] < best_metric[0]:
            best_metric[0] = vali_loss_prc[0]
            best_save_path = os.path.join(vali_path, 'best-loss')
            best_save_path = saver.save(sess, best_save_path)
        if vali_loss_prc[1] > best_metric[1]:
            best_metric[1] = vali_loss_prc[1]
            best_save_path = os.path.join(vali_path, 'best-pre1')
            best_save_path = saver.save(sess, best_save_path)
        if vali_loss_prc[2] > best_metric[2]:
            best_metric[2] = vali_loss_prc[2]
            best_save_path = os.path.join(vali_path, 'best-pre2')
            best_save_path = saver.save(sess, best_save_path)

        save_path = os.path.join(vali_path, 'itr_{}_loss_{:.4f}_pre1_{:.4f}_pre2_{:.4f}'.format(i, vali_loss_prc[0], vali_loss_prc[1], vali_loss_prc[2]))
        save_path = saver.save(sess, save_path)

# test
log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("%s, start prepare test data" % log_time)
test_thread_user, size_user_test, max_time_test, news_cnt_short_test, u_t_dispid_test, \
u_t_dispid_split_ut_test, u_t_dispid_feature_test, click_feature_test, click_sub_index_test, \
u_t_clickid_test, ut_dense_test = prepare_vali_data(4*_num_thread, test_user)
log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("%s, prepare test data end" % log_time)

best_save_path = os.path.join(vali_path, 'best-loss')
saver.restore(sess, best_save_path)
test_loss_prc = multithread_compute_test()
vali_loss_prc = multithread_compute_vali()
print("test!!!loss!!!, test: %.5f, vali: %.5f" % (test_loss_prc[0], vali_loss_prc[0]))

best_save_path = os.path.join(vali_path, 'best-pre1')
saver.restore(sess, best_save_path)
test_loss_prc = multithread_compute_test()
vali_loss_prc = multithread_compute_vali()
print("test!!!pre1!!!, test: %.5f, vali: %.5f" % (test_loss_prc[1], vali_loss_prc[1]))

best_save_path = os.path.join(vali_path, 'best-pre2')
saver.restore(sess, best_save_path)
test_loss_prc = multithread_compute_test()
vali_loss_prc = multithread_compute_vali()
print("test!!!pre1!!!, test: %.5f, vali: %.5f" % (test_loss_prc[2], vali_loss_prc[2]))
