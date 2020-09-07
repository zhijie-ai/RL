#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/8 18:13                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np
import gym

np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][1]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'
# print(env.observation_space.shape,env.action_space.shape,a_bound)#(3,) (1,) [2.]

###############################  Actor  ####################################

class Actor():
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s,output a
            self.a = sess.build_net(S,scope='eval_net',trainable=True)

            # input s_,output a_,get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if sess.replacement['name'] == 'hard':
            self.t_replacement_counter = 0
            self.hard_replace = [tf.assign(t,e) for t,e in zip(self.t_parmas,self.e_params)]
        else:
            self.soft_replace = [tf.assgin(t,(1-self.replacement['tau'])*t+self.replacement['tau']*e)
                                 for t,e in zip(self.t_params,self.e_params)]

    def _build_net(self,s,scope,trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s,30,activation=tf.nn.relu,
                                  kernel_initializer=init_w,bias_initializer=init_b,
                                  trainable=trainable,name='l1')

            with tf.variable_scope('a'):
                actions = tf.layers.dense(net,self.a_dim,activation=tf.nn.relu,
                                          kernel_initializer=init_w,bias_initializer=init_b,
                                          name='a',trainable=trainable)
                scaled_a = tf.multiply(actions,self.action_bound,name='scaled_a')# Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self,s):#batch update
        self.sess.run(self.train_op,feed_dict={S:s})

        if self.replacement['name']=='soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter %self.replacement['rep_iter_a'] ==0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter+=1

    def choose_action(self,s):
        s = s[np.newaxis,:]
        return self.sess.run(self.a,feed_dict={S:s})[0]

    