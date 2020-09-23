#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/11 10:35                       #
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
import sys
import os

def format_data():
    size_user=6
    behavior_filename='data/e3_behavior_v1.txt'
    category_filename='data/e3_news_category_v1.txt'
    feature_filename='data/e3_user_news_feature_v1.txt'
    splitter='/t'

    # 1. data_behavior
    max_disp_size=0

    data_behavior=[[] for _ in range(size_user)]
    fd = open(behavior_filename)
    for row in fd:
        row = row.split()[0]
        row = row.split(splitter)
        u_id=int(row[0])
        time_stamp = int(row[1])
        disp_list=list(map(int,row[2].split(',')))
        max_disp_size=max(max_disp_size,len(disp_list))
        pick_list=list(map(int,row[3].split(',')))
        data_behavior[u_id].append([time_stamp,disp_list,pick_list])
    fd.close()
    k=10

    # data_behavior[1]=[1584090631,[1,2,3],[5,6,7,8]],有可能是N维3列
    for i in range(len(data_behavior)):
        data_behavior[i]=sorted(data_behavior[i],key=lambda x:x[0])# 按时间戳排序

    # 1.1 click and disp behavior

    # 2. category,处理每个news_id的类别。并记录一个最大的类别和最小的类别
    max_category=0
    min_category=0
    movie_category={}#e.g movie_category[1]=[1,2,3,4]
    fd = open(category_filename)
    for row in fd:
        row=row.split()[0]
        row = row.split(splitter)
        news_id=int(row[0])
        news_cat = list(map(int,row[1].split(',')))
        news_cat=list(np.array(news_cat)+1) # let category start from 1. leave category 0 for non-clicking
        movie_category[news_id]=news_cat
        max_category=max(max_category,max(news_cat))
        min_category=min(min_category,min(news_cat))

    fd.close()

    KK= max_category+1
    if max(movie_category.keys())!=len(movie_category.keys()):
        print('warning:news category wrong')
        exit()
    else:
        size_movie=len(movie_category.keys())+1


    # movie_category[0]=[0]

    # 3. feature
    user_news_feature={}
    fd = open(feature_filename)
    for row in fd:
        row=row.split()[0]
        row = row.split(splitter)
        key = 'u'+row[0]+'n'+row[1]
        user_news_feature[key] = list(map(float,row[2].split(',')))
    fd.close()
    #4. save synthetic data
    data_parameter = [KK,k,size_user,size_movie]

    # another set of data,<--- this is what we finally use
    # 这部分主要是为了把news_id按照每个user 从0开始排序
    data_click = [[] for _ in range(size_user)]
    data_disp=[[] for _ in range(size_user)]
    data_time = np.zeros(size_user,dtype=np.int)
    data_news_cnt=np.zeros(size_user,dtype=np.int)
    feature=[[] for _ in range(size_user)]
    feature_click=[[] for _ in range(size_user)]

    #data_behavior[u_id].append([time_stamp,disp_list,pick_list])
    for user in range(len(data_behavior)):
        # (1) count number of click
        click_t=0 # 每个用户的点击item的个数
        for event in range(len(data_behavior[user])):
            pick_list=data_behavior[user][event][2]
            click_t+=len(pick_list)#splitter    a event with 2 clicking to 2 events
        data_time[user]=click_t #假设为10
        # (2)
        news_dict={} #news_dict[1]=0,为disp id
        feature_click[user]=np.zeros([click_t,20])#10*20
        click_t=0
        # data_behavior[1] = [1584090631, [1, 2, 3], [5, 6, 7, 8]]
        # [time_stamp, disp_list, pick_list]
        for event in range(len(data_behavior[user])):
            disp_list=data_behavior[user][event][1]
            pick_list=data_behavior[user][event][2]
            for id in disp_list:
                if id not in news_dict:
                    news_dict[id]=len(news_dict)#for each user ,news id start from 0
            for id in pick_list:
                data_click[user].append([click_t,news_dict[id]])
                key = 'u'+str(user)+'n'+str(id)
                #user_news_feature[key] 是一个vector
                feature_click[user][click_t]=user_news_feature[key]
                for idd in disp_list:
                    data_disp[user].append([click_t,news_dict[idd]])
                click_t+=1# splitter a event with 2 clickings to 2 events


        data_news_cnt[user]=len(news_dict)

        feature[user]=np.zeros([data_news_cnt[user],20])

        for id in news_dict:
            key = 'u'+str(user)+'n'+str(id)
            feature[user][news_dict[id]]=user_news_feature[key]
        feature[user]=feature[user].tolist()
        feature_click[user]=feature_click[user].tolist()
    return data_click,data_disp,feature,data_time,data_news_cnt,data_parameter,feature_click

# 将用户np.arange(2500, 2500+2500/4)对num_sets=8取模，分到8个桶当中。
# user_set=[8,16,32...]用户id，b_size=20,data_perform在for循环里面，遍历的是每个桶里的用户
def data_perform(user_set,b_size):
    #(1) [session,news_id(对于每个user，从0开始)] SparseTensor的indices
    # 没一个session代表依次曝光。从0开始，按照user,然后时间排序。eg:session 0:user 0 at time 0;session 1:user 0 at time 1
    display_tensor_indice=[]

    # (2) [session],长度和顺序和disp_tensor_indices一样，只是不要和第二列[news_id]
    # 这个数据如果不好准备，可以直接在TensorFlow里面用tf.split操作
    display_tensor_indice_split=[]

    # (3) 和display_tensor_indice是一样的，但是第二列只包含被click了的news_id
    click_tensor_indice=[]

    #(4) 下面这2个稍微比较难理解一点。这里是构造一个triangular matrix,用来aggregate history
    # 比较难用comment解释
    # 三角矩阵
    tril_indice=[]
    tril_value_indice=[]

    # (5) 点击过的news特征。要按某个顺序排序。具体可以看下面的逻辑
    feature_clicked_x=[]
    #(6) 所有news特征(包括点击/未点击).也是要注意顺序
    disp_current_feature_x=[]
    #(7) 等价于display_tensor_indice.index(click_tensor_indice)
    click_sub_index_2d=[]

    # 数总共有多少session，所有用户所有的点击次数,比如所有用户共点击了100个item，则sec_cnt_x=100
    sec_cnt_x=0
    # max number of news(per user)，所有用户里面最大的展示次数(只是这个set里面的用户，共8个set)
    max_news_per_user=0

    for u in user_set:
        # 构造一个triangular matrix的indices
        t_indice=[]
        #data_time[user]=click_t
        # data_time是一个数组，数组索引可看成用户id，每个位置放置的是该用户对应的点击次数
        for kk in range(min(b_size-1,data_time[u]-1)):#在特征维度和点击次数之间选最小值
            t_indice += map(lambda x: [x + kk + 1 + sec_cnt_x, x + sec_cnt_x], np.arange(data_time[u] - (kk + 1)))

        # t_indice=[[14, 0], [15, 1], [16, 2], [17, 3], [18, 4], [19, 5]]
        tril_indice += t_indice# 三角矩阵的索引
        tril_value_indice += map(lambda x: (x[0] - x[1] - 1), t_indice)#索引对应的值，是由索引得到的。

        # 。。。
        # data_click[user].append([click_t,news_dict[id]])
        click_tensor_indice_tmp = list(map(lambda x: [x[0] + sec_cnt_x, x[1]], data_click[u]))
        click_tensor_indice += click_tensor_indice_tmp#可以看做是data_click的数据格式

        #data_disp[user].append([click_t,news_dict[idd]])，在展示的index中根据点击的数据查找
        display_tensor_indice_tmp = list(map(lambda x: [x[0] + sec_cnt_x, x[1]], data_disp[u]))
        print('data_disp[u]',data_disp[u])
        # index 求元素所在的索引
        click_sub_index_tmp = list(map(lambda x: display_tensor_indice_tmp.index(x), click_tensor_indice_tmp))

        #click_sub_index_2d:[0,1,2,3,4],在点击数据的索引上加上上一个用户的展示次数
        click_sub_index_2d += map(lambda x: x + len(display_tensor_indice), click_sub_index_tmp)
        display_tensor_indice += display_tensor_indice_tmp#data_disp[u]
        # 按点击来切分[0,0,0,1,1,2,2,3,3]
        # print('data_disp[u]', len(data_disp[u]))
        display_tensor_indice_split += map(lambda x: x[0] + sec_cnt_x, data_disp[u])

        sec_cnt_x += data_time[u]
        #data_news_cnt，每个用户的历史展示次数,max_news_per_user,所有用户里面，最大的展示次数
        max_news_per_user = max(max_news_per_user, data_news_cnt[u])
        disp_current_feature_x += map(lambda x: feature[u][x], [idd[1] for idd in data_disp[u]])
        feature_clicked_x += feature_click[u]

    print('feature_clicked_x',np.array(feature_clicked_x).shape)#(6, 20)

    #disp_current_feature_x 代表当前这个set的用户的所有展示的item的特征
    return click_tensor_indice, display_tensor_indice, \
           disp_current_feature_x, sec_cnt_x, tril_indice, tril_value_indice, \
           display_tensor_indice_split, max_news_per_user, click_sub_index_2d, feature_clicked_x

# 下面3个function是tf算子的定义，主要看这个
def construct_placeholder():
    global disp_current_feature, Xs_clicked, news_size, section_length, clk_tensor_indice, clk_tensor_val, disp_tensor_indice
    global cumsum_tril_indices, cumsum_tril_value_indices
    global disp_tensor_indice_split, click_2d_subindex

    #_f_dim=20,这个20就是feature文件中定义的特征数目
    disp_current_feature = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])  # 所有特征，包括点击和未点击。
    Xs_clicked = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])  # 点击了的特征。
    news_size = tf.placeholder(dtype=tf.int64, shape=[])  # max number of news(per user)
    section_length = tf.placeholder(dtype=tf.int64)  # 总共有多少session

    # [session, news_id（对于每个user，从0开始数）] SparseTensor的indices
    disp_tensor_indice = tf.placeholder(dtype=tf.int64, shape=[None, 2])
    # [session], 长度和顺序都和disp_tensor_indices一样，只是不要第二列[news_id]
    # 这个数据如果不好准备，可以直接在tensorflow里面用tf.split对disp_tensor_indices操作得到
    #[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    disp_tensor_indice_split = tf.placeholder(dtype=tf.int64, shape=[None])

    # 和disp_tensor_indices一样, 但是这里的第二列只包含被click了的news_id
    clk_tensor_indice = tf.placeholder(dtype=tf.int64, shape=[None, 2])
    # 等价于np.ones(len(clk_tensor_indices))
    clk_tensor_val = tf.placeholder(dtype=tf.float32, shape=[None])
    # 等价于disp_tensor_indices.index(clk_tensor_indices)
    click_2d_subindex = tf.placeholder(dtype=tf.int64, shape=[None])

    # 下面这两个稍微比较难理解一点。这里是构造一个triangular matrix，用来aggregate history。
    # 比较难用comment解释，可以到时候语音我或者直接问宋老师
    cumsum_tril_indices = tf.placeholder(dtype=tf.int64, shape=[None, 2])
    cumsum_tril_value_indices = tf.placeholder(dtype=tf.int64, shape=[None])

def construct_graph(is_training,is_batch_norm):

    # (1) history feature --- net ---> clicked_feature
    # (1) construct cumulative history
    click_history=[[] for _ in range(_weighted_dim)] # _weighted_dim 代表我们要考虑多少组不同的position weight。是一个超参。
    for ii in range(_weighted_dim):
        position_weight = tf.get_variable('p_w' + str(ii), [_band_size], initializer=tf.constant_initializer(
            0.0001))  # position weight，一个要学出来的weight matrix
        #cumsum_tril_value_indices为三角矩阵的值索引，一维数组,由t_indices得到。
        #cumsum_tril_indices,三角矩阵的索引，
        print('YYYYYYYYYYYYYYYYY',cumsum_tril_value_indices)#[0 0 0 0 0 1 1 1 1 2 2 2 3 3 4]
        cumsum_tril_value = tf.gather(position_weight, cumsum_tril_value_indices)
        cumsum_tril_matrix = tf.SparseTensor(cumsum_tril_indices, cumsum_tril_value,
                                             [section_length, section_length])  # sec by sec
        #Xs_clicked为用户所有点击的特征feature_click[user][click_t]=user_news_feature[key]
        # cumsum_tril_matrix应该就是是论文中的W矩阵，应该是可以不采用这种方式来初始化的。随机初始化？
        click_history[ii] = tf.sparse_tensor_dense_matmul(cumsum_tril_matrix, Xs_clicked)
        # print(Xs_clicked)#(6, 20)
        # print(cumsum_tril_matrix)

        # print('click_history.shape',click_history[ii])# shape=(?, 20)
    # 通过position weight得到代表history的特征
    u_history_feature = tf.concat(click_history, axis=1)#(4,6,20)#这似乎就是st的定义
    print('u_history_feature.shape',u_history_feature.shape)#(6,80)
    # concat history feature and current feature
    # disp_tensor_indice_split 按时间排序的点击次数[0,0,1,1,2],下一个用户点击次数=自己的点击次数+上一个用户总的次数
    #[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],该数组的长度是
    #   当前用户的总的展示次数
    # 从u_history_feature根据该用户的展示次数选出27行(27,80)
    disp_history_feature = tf.gather(u_history_feature, disp_tensor_indice_split)#(27,80),要考虑到所有展示的item中每个item的reward
    # 用户历史点击的特征，用户历史展示的特征，disp_current_feature:当前用户所有的展示的item的特征
    print('disp_history_feature.shape',disp_history_feature.shape)
    print('disp_current_feature',disp_current_feature.shape)#(27,20)
    print('tf.concat([disp_history_feature, disp_current_feature], axis=1)',
          tf.concat([disp_history_feature, disp_current_feature], axis=1).shape)
    combine_feature = tf.reshape(tf.concat([disp_history_feature, disp_current_feature], axis=1),
                                 [-1, _f_dim * _weighted_dim + _f_dim])
    print('combine_feature.shape',combine_feature.shape)#(27,100)，如果将(27,20),(27,20)2个矩阵拼接，也是可以的吧

    # neural net，可以调
    n1 = 256
    y1 = tf.layers.dense(combine_feature, n1, activation=tf.nn.elu, trainable=True,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=_E3_sd))
    print('y1.shape',y1.shape)#(27,256)

    n2 = 32
    y2 = tf.layers.dense(y1, n2, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_E3_sd))
    y2 = tf.nn.elu(y2)
    print('y2.shape', y2.shape)#(27,32)

    # output layer
    u_disp = tf.layers.dense(y2, 1, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_E3_sd))
    print('u_disp.shape',u_disp.shape)#(27,1)，公共有27个item，输入27个score

    # construct loss function
    exp_u_disp = tf.exp(u_disp)
    print('exp_u_disp.shape',exp_u_disp.kafka_producer.py)#(27,1)
    # 分段求和函数segment_sum，理论每推荐k个(比如k=10，在造的数据中，由于之前对数据理解不准确，
    #   在展示的时候，应该是都是k个的。假设都是k个，那么在disp_tensor_indice_split参数中，相同的index算一次推荐)
    sum_exp_disp_ubar_ut = tf.segment_sum(exp_u_disp, disp_tensor_indice_split)
    #click_2d_subindex,对每个用户来说，点击的index在展示的index中的位置加上上一个用户的展示的长度。
    # 由于u_disp计算的是所有(s,a)的reward，而公式中只需要点击的item的reward。
    sum_click_u_bar_ut = tf.gather(u_disp, click_2d_subindex)
    # section_length:当前batch总共的点击次数，news_size:当前batch里用户最大的展示次数
    denseshape = [section_length, news_size]
    #clk_tensor_indice:用户的点击及点击的id在展示数据中的index，即click_data
    #clk_tensor_val: np.ones(len(click_2d)
    click_tensor = tf.SparseTensor(clk_tensor_indice, clk_tensor_val, denseshape)
    print('click_tensor',click_tensor)

    click_cnt = tf.sparse_reduce_sum(click_tensor, axis=1)
    # 下面这个loss即论文中lemma 2中证明的loss。只有一个loss即可优化φ和θ。
    print('aaaa',sum_click_u_bar_ut.shape,sum_exp_disp_ubar_ut.shape)#(?, 1) (?, 1)
    loss_sum = tf.reduce_sum(- sum_click_u_bar_ut + tf.log(sum_exp_disp_ubar_ut + 1))
    event_cnt = tf.reduce_sum(click_cnt)
    print('event_cnt',event_cnt)
    loss = loss_sum / event_cnt# 除以当前batch的点击次数

    # compute precision
    # disp_tensor_indice:当前batch中用户的展示index，同disp_data，只是增加了一个基数
    exp_disp_ubar_ut = tf.SparseTensor(disp_tensor_indice, tf.reshape(exp_u_disp, [-1]), denseshape)
    dense_exp_disp_util = tf.sparse_tensor_to_dense(exp_disp_ubar_ut, default_value=0.0)
    argmax_click = tf.argmax(tf.sparse_tensor_to_dense(click_tensor, default_value=0.0), axis=1)
    argmax_disp = tf.argmax(dense_exp_disp_util, axis=1)

    top_2_disp = tf.nn.top_k(dense_exp_disp_util, k=2, sorted=False)[1]

    # 点击对应的index是不是对应的reward最大
    precision_1_sum = tf.reduce_sum(tf.cast(tf.equal(argmax_click, argmax_disp), tf.float32))
    precision_1 = precision_1_sum / event_cnt
    # 点击的数据是不是在当前batch中top2之内
    precision_2_sum = tf.reduce_sum(
        tf.cast(tf.equal(tf.reshape(argmax_click, [-1, 1]), tf.cast(top_2_disp, tf.int64)), tf.float32))
    precision_2 = precision_2_sum / event_cnt

    return loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt

def construct_model(is_training,is_batch_norm,reuse=False):
    global lossL2
    with tf.variable_scope('model', reuse=reuse):
        loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt = construct_graph(
            is_training, is_batch_norm)

    if is_training:
        opt=tf.train.AdamOptimizer(learning_rate=_lr)
        if is_batch_norm:
            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # train_op = opt.minimize(loss+lossL2)
                train_op = opt.minimize(loss)
        else:
            # train_op = opt.minimize(loss+lossL2)
            train_op = opt.minimize(loss)

        return train_op, loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt
    else:
        return loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt



log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("%s, start" % log_time)

# 数据准备
'''
data_click：记录了每个用户点击的id在所有展示中的index
data_disp：记录了每个用户展示的id在所有展示中的index
feature:每个用户所有展示的item的特征[20,20],第一个20代表当前用户共展示了20个item，第二个20代表每个item的维度
data_time:每个用户的点击次数
data_news_cnt:每个用户展示的次数
feature_click:同feature，只是记录的是每个用户点击特征[5,20],5代表5个点击，20代表item的特征维度
'''
data_click, data_disp, feature, data_time, data_news_cnt, data_parameter, feature_click = format_data()

log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("%s, load data completed" % log_time)

_f_dim = 20

_KK, k, size_user_1, size_movie_1 = data_parameter

_regularity = 0.05

_E3_sd = 1e-3
_CCF_sd = 1e-2
_combine_sd = 1e-2
_lr = 0.001
_band_size = 20
_num_thread = 8

_weighted_dim = 4


print(['Adam', _lr, 'E3_sd', _E3_sd, 'CCF_sd', _CCF_sd, 'combine_sd', _combine_sd, 'band_size', _band_size])


log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("%s, start construct graph" % log_time)
construct_placeholder()
use_batch_normalization = False
train_opt, train_loss, train_prec1, train_prec2, train_loss_sum, train_prec1_sum, train_prec2_sum, train_event_cnt = construct_model(is_training=True, is_batch_norm=use_batch_normalization, reuse=False)
log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("%s, construct graph complete" % log_time)

train_user = np.arange(6)

batch_size = 1

sess = tf.Session()
sess.run(tf.global_variables_initializer())

iterations = 1

best_metric = [100000.0, 0.0, 0.0]


saver = tf.train.Saver(max_to_keep=None)

for i in range(iterations):

    training_start_point = (i * batch_size) % 6
    training_user = train_user[training_start_point : training_start_point + batch_size]
    print('training_user',training_user)
    if i == 0:
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, start prepare train data" % log_time)

    click_2d, disp_2d, feature_tr, sec_cnt, tril_ind, tril_value_ind, disp_2d_split_sect, \
    news_cnt_sht, click_2d_subind, feature_clicked_tr = data_perform(training_user, _band_size)
    print('click_2d',click_2d)#[[0, 0], [1, 2], [2, 3], [3, 0], [4, 2], [5, 4]]
    print('disp_2d',disp_2d)
    print('feature_tr',np.array(feature_tr).shape)#(27, 20)
    # print('tril_ind',np.array(tril_ind).shape)#(15, 2)
    print('tril_value_ind',np.array(tril_value_ind))#(15,)
    print('disp_2d_split_sect',disp_2d_split_sect)#(27,)
    print('news_cnt_sht',news_cnt_sht)#6
    # print('click_2d_subind',click_2d_subind)
    print('sec_cnt',sec_cnt)#6
    print('clk_tensor_val',len(click_2d))
    print('click_2d_subind',click_2d_subind)#[0, 7, 13, 15, 20, 25]

    # print('BBBB',np.array(feature_clicked_tr).shape)
    #cumsum_tril_indices = tf.placeholder(dtype=tf.int64, shape=[None, 2])
    #cumsum_tril_value_indices = tf.placeholder(dtype=tf.int64, shape=[None])
    print('CCCCCCCCCCCCCCCC',sec_cnt)
    print('DDDDDDDDDDDDDDDDDDD',tril_value_ind)#[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
    sess.run(train_opt, feed_dict={disp_current_feature: feature_tr, news_size: news_cnt_sht,
                                   section_length: sec_cnt,
                                   clk_tensor_indice: click_2d,
                                   clk_tensor_val: np.ones(len(click_2d), dtype=np.float32),
                                   disp_tensor_indice: np.array(disp_2d),
                                   cumsum_tril_indices: tril_ind,
                                   cumsum_tril_value_indices: np.array(tril_value_ind, dtype=np.int64),
                                   click_2d_subindex: click_2d_subind,
                                   disp_tensor_indice_split: disp_2d_split_sect,
                                   Xs_clicked: feature_clicked_tr})

    if i == 0:
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, first iteration training complete" % log_time)