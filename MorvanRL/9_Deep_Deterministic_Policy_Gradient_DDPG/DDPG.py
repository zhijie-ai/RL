#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/10/14 17:38                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG.py

"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""
#  这种方式的DDPG的实现和刘建平老师的DDPG中actor的实现不太一样。虽然原始的梯度都是pi对theta求导乘以
#   Q对a求导，该文章的实现即时原始公式的实现。但在刘建平老师的实现中，经过推理演变，Actor可以直接通过最小化-Q来实现，
#  而critic的实现中都是一样的。通过最小化TD error来优化



import tensorflow as tf
import numpy as np
import gym
import time


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
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'

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
            # input s,output,a
            self.a = self._build_net(S,scope='eval_net',trainable=True)

            # input s_,output a,ge a_ for critic
            self.a_ = self._build_net(S_,scope='target_net',trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t,e) for t,e in zip(self.t_params,self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]


    def _build_net(self,s,scope,trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0.,0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s,30,activation=tf.nn.relu,
                                  kernel_initializer=init_w,bias_initializer=init_b,name='l1',
                                  trainable=trainable)

            with tf.variable_scope('a'):
                actions = tf.layers.dense(net,self.a_dim,activation=tf.nn.tanh,
                                          kernel_initializer=init_w,bias_initializer=init_b,
                                          name='a',trainable=trainable)
                scaled = tf.multiply(actions,self.action_bound,name='scaled_a') # Scale output to -action_bound to action_bound

        return scaled

    def learn(self,s):#batch update
        self.sess.run(self.train_op,feed_dict={S:s})

        if self.replacement['name'] =='soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self,s):
        s = s[np.newaxis,:] # single state
        return self.sess.run(self.a,feed_dict={S:s})[0] # single action

    def add_grad_to_graph(self,a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            # grad_ys的意义在于对xs中的每个梯度加权重
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr) # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads,self.e_params))


    ###############################  Critic  ####################################

class Critic():
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement


        with tf.variable_scope('Critic'):
            # Input (s,a) output q
            self.a = tf.stop_gradient(a) # stop critic update flows to actor
            self.q = self._build_net(S,self.a,'eval_net',trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_,a_,'target_net',trainable=False)# target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')


        with tf.variable_scope('target_q'):
            self.target_q = R+self.gamma*self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]  # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1

#####################  Memory  ####################

class Memory():
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print()
action_bound = env.action_space.high

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

var = 3  # control exploration

t1 = time.time()

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    for j in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        '''
        1.刚开始我以为计算目标Q值时使用的a'是否是同一个网络产生的来区分on还是off，网上都说DDPG是off-policy,如果采用这种方式
            来区分的话，a'确实来自actor‘网络，是一个off-policy,可是同时，ppo却是一个on-policy,ppo计算目标Q的时候(计算advantage的时候)
            也是theta'(论文，而不是代码实现),DDPG和PPO都是参数复制，DDPG在计算target的时候是actor’，而PPO在计算advantage的时候是theta',
            同样是参数复制，为啥一个是on-policy一个是off-policy？
        2.我想明白了，确实可以通过在计算target 的时候的策略和采样的时候的行为策略是否是一个来区分on还是off-policy，
            都是参数复制，明显actor和action'都是一样的。但因为是参数复制，PPO可以看作是同一个policy,而区别在于，DDPG在和环境交互时
            使用的action是加了随机噪声的，相当于是beta策略，而在计算target的时候是actor的确定策略。相反，PPO从始至终都相当于是一个策略。
            收集数据的策略和目标策略(计算目标的策略是同一回事？比如DQN中贪婪策略，行为策略是epsilon-贪婪策略)
        '''
        # Add exploration noise
        a = actor.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        M.store_transition(s, a, r / 10, s_)

        if M.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            b_M = M.sample(BATCH_SIZE)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]

            critic.learn(b_s, b_a, b_r, b_s_)
            actor.learn(b_s)

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:
                RENDER = True
            break

print('Running time: ', time.time()-t1)
