#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/1 22:41                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 理解:https://mp.weixin.qq.com/s/HzGrOE3si7YvQUdUA1RPNA
# 1、裁剪
# 2、投影
# 3、更新价值采样点

# 注意:在一般的DQN算法中，目标Q值的计算时r+gamma*Q(s_,a_),而在DisDQN中，由于输出的是每个价值采样点的概率，
#   Q=p(s,a)*z,目标Q=p(s_,a)*z,目标Q中的a不是a_,而是根据greedy_action(s)选出的动作

import tensorflow as tf
import numpy as np
import random
from collections import deque
from .Config import Categorical_DQN_Config
from .utils import conv, dense
import math

class Categorical_DQN():
    def __init__(self,env,config):
        self.sess = tf.InteractiveSession()
        self.config = config
        self.v_max = self.config.v_max
        self.v_min = self.config.v_min
        self.atoms = self.config.atoms # 采样价值点的个数

        self.epsilon = self.config.INITIAL_EPSILON
        self.state_shape = env.observation_shape.shape
        self.action_dim = env.action_space.n

        self.time_step = 0

        target_state_shape=[1]
        target_state_shape.extend(self.state_shape)
        self.state_input = tf.placeholder(tf.float32,target_state_shape)
        self.action_input = tf.placeholder(tf.int32,[1,1])
        self.m_input = tf.placeholder(tf.float32,[self.atoms])

        self.delta_z = (self.v_max-self.v_min)/(self.atoms-1)
        self.z = [self.v_min+i*self.delta_z for i in range(self.atoms)]

        self.build_cate_dqn_net()

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        self.save_model()
        self.restore_model()

    def builde_layers(self,state,action,c_names,units_1,units_2,w_i,b_i,reg=None):
        with tf.variable_scope('conv1'):
            conv1 = conv(state,[5,5,3,6],[6],[1,2,2,1],w_i,b_i)
        with tf.variable_scope('conv2'):
            conv2 = conv(conv1,[3,3,6,12],[12],[1,2,2,1],w_i,b_i)
        with tf.variable_scope('flatten'):
            flatten = tf.contrib.layers.flatten(conv2)

        with tf.variable_scope('dense1'):
            dense1 = dense(flatten,units_1,[units_1],w_i,b_i)
        with tf.variable_scope('dense2'):
            dense2 = dense(dense1,units_2,[units_2],w_i,b_i)
        with tf.variable_scope('concat'):
            concatenated = tf.concat([dense2,tf.cast(action,tf.float32)],1)
        with tf.variable_scope('dense3'):
            dense3 = dense(concatenated,self.atoms,[self.atoms],w_i,b_i)

        return tf.nn.softmax(dense3)

    def build_cate_dqn_net(self):
        with tf.variable_scope('target_net'):
            c_names = ['target_net_arams',tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1,0.1)
            b_i = tf.constant_initializer(0.1)
            self.z_target = self.builde_layers(self.state_input,self.action_input,
                                               c_names,24,24,w_i,b_i)

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1,0.1)
            b_i = tf.constant_initializer(0.1)
            self.z_eval = self.builde_layers(self.state_input,self.action_input,
                                             c_names,24,24,w_i,b_i)

        self.q_eval = tf.reduce_sum(self.z_eval*self.z)
        self.q_target = tf.reduce_sum(self.z_target*self.z)

        self.cross_entropy_loss = -tf.reduce_sum(self.m_input*tf.log(self.z_eval))
        self.optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE).minimize(self.cross_entropy_loss)

        eval_params = tf.get_collection('eval_net_params')
        target_params = tf.get_collection('target_net_params')

        self.update_target_net = [tf.assign(t,e) for t,e in zip(target_params,eval_params)]

    def train(self,s,r,action,s_,gamma):
        self.time_step +=1
        list_q_ = [self.sess.run(self.q_target,feed_dict={self.state_input:[s_],
                                                          self.action_input:[[a]]})
                   for a in range(self.action_dim)]
        a_ = tf.argmax(list_q_).eval()
        p = self.sess.run(self.z_target,feed_dict={self.state_input:[s_],
                                                   self.action_input:[[a_]]})[0]
        m = np.zeros(self.atoms)
        for j in range(self.atoms):
            Tz = min(self.v_max,max(self.v_min,r+gamma*self.z[j]))
            bj = (Tz-self.v_min)/self.delta_z # 分布在第几块里
            l,u = math.floor(bj),math.ceil(bj)# 上下界

            pj = p[j]

            m[int(l)] += pj*(u-bj)
            m[int(u)] += pj*(bj-1)

        self.sess.run(self.optimizer,feed_dict={self.state_input:[s],self.action_input:[action],
                                                self.m_input:m})

        if self.time_step %self.config.UPDATE_TARGET_NET ==0:
            self.sess.run(self.update_target_net)

    def save_model(self):
        print("Model saved in : ", self.saver.save(self.sess, self.config.MODEL_PATH))

    def restore_model(self):
        self.saver.restore(self.sess,self.config.MODEL_PATH)
        print('Model restored.')

    def greedy_action(self,s):
        self.epsilon = max(self.config.FINAL_EPSILON,self.epsilon*self.config.EPSILON_DECAY)
        if random.random()<=self.epsilon:
            return random.randint(0,self.action_dim-1)
        return np.max([self.sess.run(self.q_target,feed_dict={self.state_input:[s],
                                                              self.action_input:[[a]]})
                       for a in range(self.action_dim)])




