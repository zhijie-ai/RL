#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/21 10:23                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import tensorflow as tf
import numpy as np
from collections import deque
from itertools import chain
import os
from utils.yjp_decorator import cost_time_def

class DQN():
    def __init__(self,env,args):
        self.env=env
        self.f_dim = env.f_dim
        self.type = args.dqn_type
        self.pw_dim = args.pw_dim
        self.k = args.k
        self.model_path = args.model_path
        self.std = args.q_std
        self.hidden_dims = args.dqn_dims
        self.lr = args.dqn_lr
        self.min_value = args.min_value
        self.band_size = args.pw_band_size
        self.placeholder = {}

        self.sess=tf.compat.v1.InteractiveSession()
        self.global_step = tf.train.get_or_create_global_step()#trainable=False
        self._init()
        self.sess.run(tf.global_variables_initializer())
        self.saver =tf.compat.v1.train.Saver()
        self.agg_variables = tf.compat.v1.trainable_variables()

    def _init(self):
        self.construct_placeholder()
        self.construct_Q_and_loss()
        self.construct_max_Q()

    def construct_placeholder(self):
        # max Q placeholder  这里定义argmax Q 和 max_Q
        self.placeholder['all_action_user_indices'] = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None])
        self.placeholder['all_action_tensor_indices'] = tf.compat.v1.placeholder(dtype=tf.int64,shape=[None,2])
        # 行当前batch的用户数，列为当前batch的用户中可选的action最大的数目。
        self.placeholder['all_action_tensor_shape'] = tf.compat.v1.placeholder(dtype=tf.int64,shape=[2])

        # action_cnt = np.cumsum(action_cnt)
        # action_cnt = [0] + list(action_cnt[:-1])
        # 当前用户之前的用户所有的可选的action的数量
        self.placeholder['action_count'] = tf.compat.v1.placeholder(dtype=tf.int64,shape=[None])
        #action_space_cnt[uu] = len(action_space)
        self.placeholder['action_space_count'] = tf.compat.v1.placeholder(dtype=tf.int64,shape=[None])

        # online版本：建议直接把all_action_feature_gather作为placeholder，输入所有可以选的items的features
        self.placeholder['all_action_id'] = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None])

        #----------------------------Q and loss placeholder
        # 这里定义Q function还有对应的loss。定义的时候，假设同时处理一个batch的数据，所以稍微复杂一点。
        # 输出_k个Q function，_k个loss，_k个train op
        self.placeholder['current_action_space'] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.f_dim])
        self.placeholder['action_space_mean']=tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,self.f_dim])
        self.placeholder['action_space_std']=tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,self.f_dim])
        self.placeholder['y_label'] = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None])

    def mlp(self,x,hidden_dims,output_dim,activation,sd,act_last=False):
        hidden_dims = tuple(map(int,hidden_dims.split('-')))
        for h in hidden_dims:
            x = tf.layers.dense(x,h,activation=activation,trainable=True,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=sd))

        if act_last:
            return tf.layers.dense(x,output_dim,activation=activation,trainable=True,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=sd))
        else:
            return tf.layers.dense(x, output_dim, trainable=True,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=sd))

    # 这里定义Q function还有对应的loss。定义的时候，假设同时处理一个batch的数据，所以稍微复杂一点。
    # 输出_k个Q function，_k个loss，_k个train op,相当于是DQN的前向网络
    def construct_Q_and_loss(self):
        # (1) action states - offline的实验受到数据的限制，所以加了一个mean和std。
        # 做online实验没有数据的限制，我觉得这部分的input可以直接不要
        if self.type=='offline':
            self.action_state = tf.concat([self.placeholder['action_space_mean'],self.placeholder['action_space_std']],axis=1)

            # (2) action id - 推荐的items的id。online的版本可以直接输入feature vector而不是id。
            # 换言之，可以忽略action_k_id，直接把（3）的action_k_feature_gather定义成placeholder，输入item features。
            # action_k_id = [[] for _ in range(self.k)]
            action_k_id = ['action_k_{}'.format(i) for i in np.arange(self.k)]
            for ii in range(self.k):
                self.placeholder[action_k_id[ii]] = tf.compat.v1.placeholder(dtype=tf.int64,shape=[None])
            action_k_feature_gather = [[] for _ in range(self.k)]
            for ii in range(self.k):
                # action_k_feature_gather[ii] 代表推荐的第ii个item的feature。（总共推荐_k个item）
                action_k_feature_gather[ii] = tf.gather(self.placeholder['current_action_space'],self.placeholder[action_k_id[ii]])

            # 定义Q: input：（user_states, action_states, action_feature）
            concate_input_k = [[] for _ in range(self.k)]
            action_feature_list=[]
            q_value_k = [[] for _ in range(self.k)]
            self.loss_k = [[] for _ in range(self.k)]
            opt_k = [[] for _ in range(self.k)]
            train_variable_k = [[] for _ in range(self.k)]
            self.train_op_k = [[] for _ in range(self.k)]

            for ii in range(self.k):
                # 把（user_states, action_states, action_feature）三种vectors concat在一起，作为input。（online版本可以忽略action_states）
                # 注意:action_feature_list 是一步步变大的，从length=1到self.k
                action_feature_list.append(action_k_feature_gather[ii])
                action_feature_list_ = tf.concat(action_feature_list,axis=1)
                concate_input_k[ii]=tf.concat([self.env.user_states,self.action_state,action_feature_list_],axis=1)
                concate_input_k[ii] = tf.reshape(concate_input_k[ii],[-1,self.pw_dim*self.f_dim+2*self.f_dim+int(ii+1)*self.f_dim])

                current_variables = tf.trainable_variables()
                # q_value_k[ii]: 构造paper里面提到的Q^j, where j=1,...,_k
                with tf.variable_scope('Q'+str(ii)+'-function',reuse=False):
                    q_value_k[ii] = self.mlp(concate_input_k[ii],self.hidden_dims,1,tf.nn.elu,sd=self.std,act_last=False)

                q_value_k[ii]=tf.reshape(q_value_k[ii],[-1])

                # loss
                # y_label为reward
                ##y_label就是env算出来的reward,每个用户的reward，segment_sum操作了
                self.loss_k[ii] = tf.reduce_mean(tf.squared_difference(q_value_k[ii],self.placeholder['y_label']))#
                opt_k[ii] = tf.train.AdamOptimizer(learning_rate=self.lr)

                train_variable_k[ii] = list(set(tf.trainable_variables())-set(current_variables))
                self.train_op_k[ii] = opt_k[ii].minimize(self.loss_k[ii],var_list=train_variable_k[ii],global_step=self.global_step)

            # self.sess.run(tf.variables_initializer(list(set(tf.global_variables())-set(agg_variables))))

            self.q_feed_dict={self.placeholder['current_action_space']:[],self.placeholder['action_space_mean']:[],
                         self.placeholder['action_space_std']:[],self.env.placeholder['Xs_clicked']:[],
                         self.env.placeholder['history_order_indices']:[],self.env.placeholder['history_user_indices']:[],
                         self.placeholder['y_label']:[]}

            # return q_feed_dict,loss_k,train_op_k

        else:# online
            action_k_feature_gather = ['action_k_feature_gather:{}'.format(i) for i in np.arange(self.k)]
            for ii in range(self.k):
                self.placeholder[action_k_feature_gather[ii]] = tf.compat.v1.placeholder(dtype=tf.int64,shape=[None])

            # 定义Q: input：（user_states, action_states, action_feature）
            concate_input_k = [[] for _ in range(self.k)]
            action_feature_list=[]
            q_value_k = [[] for _ in range(self.k)]
            loss_k = [[] for _ in range(self.k)]
            opt_k = [[] for _ in range(self.k)]
            train_variable_k = [[] for _ in range(self.k)]
            train_op_k = [[] for _ in range(self.k)]

            for ii in range(self.k):
                # 把（user_states, action_states, action_feature）三种vectors concat在一起，作为input。（online版本可以忽略action_states）
                # 注意:action_feature_list 是一步步变大的，从length=1到self.k
                action_feature_list.append(self.placeholder[action_k_feature_gather[ii]])
                action_feature_list_ = tf.concat(action_feature_list,axis=1)
                concate_input_k[ii]=tf.concat([self.env.user_states,self.action_state,action_feature_list_],axis=1)
                concate_input_k[ii]=tf.concat([self.env.user_states,action_feature_list],axis=1)
                concate_input_k[ii] = tf.reshape(concate_input_k[ii],[-1,self.pw_dim*self.f_dim+int(ii+1)*self.f_dim])

                current_variables = tf.trainable_variables()
                # q_value_k[ii]: 构造paper里面提到的Q^j, where j=1,...,_k
                with tf.variable_scope('Q'+str(ii)+'-function',reuse=False):
                    q_value_k[ii] = self.mlp(concate_input_k[ii],self.hidden_dims,1,tf.nn.elu,sd=self.std,act_last=False)

                q_value_k[ii] = tf.reshape(q_value_k[ii],[-1])

                # loss
                # y_label为reward
                ##y_label就是env算出来的reward,每个用户的reward，segment_sum操作了
                loss_k[ii] = tf.reduce_mean(tf.squared_difference(q_value_k[ii],self.placeholder['y_label']))#
                opt_k[ii] = tf.train.AdamOptimizer(learning_rate=self.lr)

                train_variable_k[ii] = list(set(tf.trainable_variables())-set(current_variables))
                train_op_k[ii] = opt_k[ii].minimize(loss_k[ii],var_list=train_variable_k[ii],global_step=self.global_step)

            # self.sess.run(tf.variables_initializer(list(set(tf.global_variables())-set(agg_variables))))

            q_feed_dict={self.placeholder['current_action_space']:[],self.placeholder['action_space_mean']:[],
                         self.placeholder['action_space_std']:[],self.env.placeholder['Xs_clicked']:[],
                         self.env.placeholder['history_order_indices']:[],self.env.placeholder['history_user_indices']:[],
                         self.placeholder['y_label']:[]}

            for ii in range(self.k):
                self.placeholder[action_k_feature_gather[ii]]=[]

            # return q_feed_dict,loss_k,train_op_k

    #这里定义argmax Q 和 max_Q 其实就是prediction
    def construct_max_Q(self):
        if self.type == 'offline':
            # online版本：建议直接把all_action_feature_gather作为placeholder，输入所有可以选的items的features
            # current_action_space:action_space += feature_space[user]
            all_action_feature_gather = tf.gather(self.placeholder['current_action_space'], self.placeholder['all_action_id'])
            user_states_scatter = tf.gather(self.env.user_states,self.placeholder['all_action_user_indices'])
            # online版本：建议：action states可以不需要
            action_states_scatter = tf.gather(self.action_state,self.placeholder['all_action_user_indices'])

            max_action_feature_list=[]
            max_action_k = [[] for _ in range(self.k)]
            max_action_feature_k = [[] for _ in range(self.k)]
            to_avoid_repeat_tensor = tf.zeros(tf.cast(self.placeholder['all_action_tensor_shape'],tf.int32))

            max_q_value=[]
            for ii in range(self.k):
                # 构造Q_j的input（notation: j就是ii）
                # 注意：max_action_feature_list是逐步变大，从length=0到length=_k - 1
                if ii ==0:
                    concate_input = tf.concat([user_states_scatter,action_states_scatter,all_action_feature_gather],axis=1)
                else:
                    max_action_feature_list_ = tf.concat(max_action_feature_list,axis=1)
                    concate_input = tf.concat([user_states_scatter,action_states_scatter,max_action_feature_list_,all_action_feature_gather],axis=1)
                concate_input = tf.reshape(concate_input,[-1,self.pw_dim*self.f_dim+2*self.f_dim+self.f_dim*int(ii+1)])
                # 把所有action(所有item)对应的Q_j values算出来
                # 注意:Q_j 要reuse 在construct_Q_and_loss中定义的Q_j
                with tf.variable_scope('Q'+str(ii)+'-function', reuse=True):
                    q_value_all = self.mlp(concate_input,self.hidden_dims,1,tf.nn.elu,sd=self.std,act_last=False)

                q_value_all = tf.reshape(q_value_all,[-1])
                q1_tensor = tf.sparse_to_dense(self.placeholder['all_action_tensor_indices'],
                                               self.placeholder['all_action_tensor_shape'],q_value_all,default_value=self.min_value)
                q1_tensor += to_avoid_repeat_tensor

                #max_action_k[ii]:得到Q_j值最优的item，作为推荐的第j个item
                max_action_k[ii] = tf.argmax(q1_tensor,axis=1)
                # to_avoid_repeat_tensor是为了避免重复推荐一样的item。因为我们希望得到_k个不同的items。
                to_avoid_repeat_tensor += tf.one_hot(max_action_k[ii],tf.cast(self.placeholder['all_action_tensor_shape'][1],tf.int32),
                                                     on_value=self.min_value,off_value=0.0)
                # 下面几行是把max_action_k[ii]变成真实的item id。这部分应该根据自己的实验数据格式来决定如何写。
                # 截止到当前用户，之前用户所有的可选的action的总和。不包括当前用户的cnt
                # action_count
                max_action_k[ii] = tf.add(max_action_k[ii],self.placeholder['action_count'])
                # action_id += action_id_u 为下面的all_action_id
                max_action_k[ii] = tf.gather(self.placeholder['all_action_id'],max_action_k[ii])
                max_action_feature_k[ii] = tf.gather(self.placeholder['current_action_space'],max_action_k[ii])
                max_action_k[ii] = max_action_k[ii] - self.placeholder['action_space_count']

                # 把argmax Q_j得到的最优item的特征存起来，作为下一个Q_{j+1}的input
                max_action_feature_k_scatter = tf.gather(max_action_feature_k[ii],self.placeholder['all_action_user_indices'])
                max_action_feature_list.append(max_action_feature_k_scatter)

                max_q_val_k = tf.math.segment_max(q_value_all,self.placeholder['all_action_user_indices'])
                max_q_value.append(max_q_val_k)


            self.max_q_value = tf.math.reduce_max(max_q_value,axis=0)
            # self.max_q_value = tf.math.segment_max(q_value_all,self.placeholder['all_action_user_indices'])

            self.max_action = tf.stack(max_action_k,axis=1)
            max_action_disp_features = tf.concat(max_action_feature_k,axis=1)
            self.max_action_disp_features = tf.reshape(max_action_disp_features,[-1,self.f_dim])

            max_q_feed_dict = {self.placeholder['all_action_id']: [], self.placeholder['all_action_user_indices']: [],
                               self.placeholder['all_action_tensor_indices']: [], self.placeholder['all_action_tensor_shape']: [],
                               self.placeholder['current_action_space']: [], self.env.placeholder['Xs_clicked']: [],
                               self.env.placeholder['history_order_indices']: [], self.env.placeholder['history_user_indices']: [],
                               self.placeholder['action_count']: [], self.placeholder['action_space_count']: [],
                               self.placeholder['action_space_mean']: [], self.placeholder['action_space_std']: []}

            # return max_q_value,max_action,max_action_disp_features,max_q_feed_dict

    def train_on_batch(self,q_feed_dict):
        _,loss_k,step = self.sess.run([self.train_op_k,self.loss_k,self.global_step],feed_dict=q_feed_dict)
        return loss_k,step

    def save(self,model_name):
        save_path = os.path.join(self.model_path, model_name)
        self.saver.save(self.sess,save_path)
        print('model:{} saved success!!!!'.format(save_path))

    def restore(self,model_name):
        best_save_path = os.path.join(self.model_path, model_name)
        self.saver.restore(self.sess, best_save_path)
        print('model:{} loaded success!!!!'.format(best_save_path))

    def choose_action(self,max_q_feed_dict):
        max_action,max_action_disp_feature = self.sess.run([self.max_action,self.max_action_disp_features],feed_dict=max_q_feed_dict)
        return max_action,max_action_disp_feature

    def get_max_q_value(self,max_q_feed_dict):
        max_q_value = self.sess.run(self.max_q_value,feed_dict=max_q_feed_dict)
        return max_q_value