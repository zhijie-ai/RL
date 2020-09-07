#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#        2019/3/11 15:27                      #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

## https://www.cnblogs.com/pinard/p/10345762.html.html ##
## 强化学习(十六) 深度确定性策略梯度(DDPG) ##
#连续动作，输出的是一个某个具体的动作。
# https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/ddpg.py

import tensorflow as tf
import numpy as np
import gym
import time

MAX_EPISODES = 2000
MAX_EP_STEPS = 200
LR_A = 0.001 # learning rate for actor
LR_C = 0.002 # learning rate for critic
GAMMA = 0.9
TAU = 0.01 # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME='Pendulum-v0'

###############################  DDPG  ####################################

class DDPG():
    def __init__(self,a_dim,s_dim,a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY,s_dim*2+a_dim+1),dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim,self.s_dim,self.a_bound = a_dim,s_dim,a_bound
        self.S = tf.placeholder(tf.float32,[None,s_dim],'s')
        self.S_ = tf.placeholder(tf.float32,[None,s_dim],'s_')
        self.R = tf.placeholder(tf.float32,[None,1],'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S,scope='eval',trainable=True)
            a_ = self._build_a(self.S_,scope='target',trainable=False)

        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S,self.a,scope='eval',trainable=True)
            q_ = self._build_c(self.S_,a_,scope='target',trainable=False)

        # networks parameters
        self.ae_params= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t,(1-TAU)*t+TAU*e)
                             for t,e in zip(self.at_params+self.ct_params,self.ae_params+self.ce_params)]

        q_target = self.R+GAMMA*q_
        # in the feed_dict for the td_error,the self.a should change to action in memory
        td_error = tf.losses.mean_squared_error(labels=q_target,predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error,var_list=self.ce_params)

        a_loss = tf.reduce_mean(q) # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss,var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self,s):
        return self.sess.run(self.a,{self.S:s[np.newaxis,:]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY,size=BATCH_SIZE)
        bt = self.memory[indices,:]
        bs = bt[:,:self.s_dim]
        ba = bt[:,self.s_dim:self.s_dim+self.a_dim]
        br = bt[:,-self.s_dim-1:-self.s_dim]
        bs_ = bt[:,-self.s_dim:]

        self.sess.run(self.atrain,{self.S:bs})
        self.sess.run(self.ctrain,{self.S:bs,self.a:ba,self.R:br,self.S_:bs_})

    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,a,[r],s_))
        index= self.pointer %MEMORY_CAPACITY # replace the old memory with new momery
        self.memory[index,:] = transition
        self.pointer +=1

    def _build_a(self,s,scope,trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s,30,activation=tf.nn.relu,name='l1',trainable=trainable)
            a = tf.layers.dense(net,self.a_dim,activation=tf.nn.tanh,name='a',trainable=trainable)
            return tf.multiply(a,self.a_bound,name='scaled_a')

    def _build_c(self,s,a,scope,trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s',[self.s_dim,n_l1],trainable=trainable)
            w1_a = tf.get_variable('w1_a',[self.a_dim,n_l1],trainable=trainable)
            b1 = tf.get_variable('b1',[1,n_l1],trainable=trainable)
            net = tf.nn.relu(tf.matmul(s,w1_s)+tf.matmul(a,w1_a)+b1)
            return tf.layers.dense(net,1,trainable=trainable) # Q(s,a)
###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

print(env.action_space)#一维向量Box(1,)
print(env.observation_space)#Box(3,)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
print(env.observation_space.shape,env.action_space.shape,a_bound)#(3,) (1,) [2.]

ddpg = DDPG(a_dim,s_dim,a_bound)

var =3 # control exploration
t1 = time.time()
for episode in range(MAX_EPISODES):
    s= env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        print('action:',a)
        a = np.clip(np.random.normal(a,var),-2,2) # add randomness to action selection for exploration
        s_,r,done,info = env.step(a)

        ddpg.store_transition(s,a,r/10,s_)

        if ddpg.pointer >MEMORY_CAPACITY:
            var *= .9995 # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:',episode,'Reward:%i' %int(ep_reward),'Explore:%.2f' %var)
            # if ep_reward > -300:RENDER = True
            break

    if episode %100 == 0:
        total_reward = 0
        for i in range(10):
            state = env.reset()
            for j in range(MAX_EP_STEPS):
                env.render()
                action = ddpg.choose_action(state) # direct action for test
                state,reward,done,_ = env.step(action)
                total_reward += reward
                if done:
                    break
        avg_reward = total_reward/10
        print('episode:',episode,'Evaluation Average reward:',avg_reward)
print('Running time:',time.time()-t1)