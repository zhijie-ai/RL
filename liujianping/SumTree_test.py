#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2019/2/21 17:35
 =================知行合一=============
'''

import gym
import tensorflow as tf
import numpy as np
import random

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 128 # size of minibatch
REPLACE_TARGET_FREQ = 10 # frequency to update target Q network

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from :
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority data in the tree
    """
    data_pointer = 0

    def __init__(self,capacity):
        self.capacity = capacity # for all priority values
        self.tree = np.zeros(2*capacity-1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity,dtype=object) # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self,p,data):
        # print('CCCCCCCCCC', p)
        tree_idx = self.data_pointer + self.capacity-1
        self.data[self.data_pointer] = data # update data_frame
        self.update(tree_idx,p) # update tree_frame

        self.data_pointer +=1
        if self.data_pointer >= self.capacity: # replace when exceed the capacity
            self.data_pointer = 0


    def update(self,tree_idx,p):
        change = p-self.tree[tree_idx]
        # print('BBBBBBBBBBBBB',change,p)
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:# this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx-1) //2
            self.tree[tree_idx] += change

    def get_leaf(self,v):
        """
        :param v:
        :return:
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_indx = 0
        while True:# the while loop is faster than the method in the reference code
            cl_idx = 2*parent_indx+1 # this leaf's left and right kids
            cr_idx = cl_idx+1
            if cl_idx >= len(self.tree):# reach bottom, end search
                leaf_idx = parent_indx
                break
            else:# downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_indx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_indx = cr_idx

        data_idx = leaf_idx-self.capacity+1
        return leaf_idx,self.tree[leaf_idx],self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0] # the root



class Memory(object): # stored  as (s,a,r,s_) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self,capacity):
        self.tree = SumTree(capacity)

    def store(self,transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p,transition)

    def sample(self,n):
        b_idx,b_momery,ISWeights = np.empty((n,),dtype=np.int32),\
                                   np.empty((n,self.tree.data[0].size)),\
                                   np.empty((n,1))
        pri_seg = self.tree.total_p/n # priority segment
        self.beta = np.min([1.,self.beta+self.beta_increment_per_sampling]) # max=1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:])/self.tree.total_p # for later calculate ISweight
        if min_prob == 0:
            min_prob= 0.00001
        for i in range(n):
            a,b = pri_seg*i,pri_seg*(i+1)
            v = np.random.uniform(a,b)
            idx,p,data = self.tree.get_leaf(v)
            prob = p/self.tree.total_p
            ISWeights[i,0] = np.power(prob/min_prob,-self.beta)
            b_idx[i],b_momery[i,:] = idx,data

        return b_idx,b_momery,ISWeights

    def batch_update(self,tree_idx,abs_erros):
        abs_erros += self.epsilon # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_erros,self.abs_err_upper)
        ps = np.power(clipped_errors,self.alpha)
        for ti,p in zip(tree_idx,ps):
            self.tree.update(ti,p)

class DQN():
    #DQN Agent
    def __init__(self,env):
        # init experience replay
        self.replay_total = 0
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.memory = Memory(capacity=REPLAY_SIZE)


    def egreedy_action(self,state):
        return random.randint(0, self.action_dim - 1)

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict={self.state_input:[state]})[0])

    def store_transition(self,s,a,r,s_,done):
        transition=np.hstack((s,a,r,s_,done))
        self.memory.store(transition) # have high priority for newly arrived transition

    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] =1
        self.store_transition(state,one_hot_action,reward,next_state,done)
        self.replay_total +=1
        if self.replay_total > BATCH_SIZE:
            print()
            # self.train_Q_network()
            # print('replay_total>{},BATCH_SIZE>{}'.format(self.replay_total,BATCH_SIZE))

    def train_Q_network(self):
        self.time_step +=1
        # Step 1:obtain random minibatch from replay memory
        tree_idx,minibatch,ISWeights=self.memory.sample(BATCH_SIZE)

        state_batch = minibatch[:,0:4]
        action_batch = minibatch[:,4:6]
        reward_batch = [data[6] for data in minibatch]
        next_state_batch = minibatch[:,7:11]
        # Step 2:calculate y
        y_batch = []
        current_Q_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        max_action_next = np.argmax(current_Q_batch,axis=1)
        target_Q_batch = self.target_Q_value.eval(feed_dict={self.state_input:next_state_batch})

        for i in range(0,BATCH_SIZE):
            done = minibatch[i][11]
            if done:
                y_batch.append(reward_batch[i])
            else:
                target_Q_value = target_Q_batch[i,max_action_next[i]]
                y_batch.append(reward_batch[i]+GAMMA*target_Q_value)


        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch,
            self.ISWeights:ISWeights
        })





# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 300 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 5 # The number of experiment test every 100 episode

def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    total_step = 0
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            total_step += 1
            action = agent.egreedy_action(state)
            next_state,reward,done,_ = env.step(action)
            # Define reward for agent
            reward = -1 if done else 0.1
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done :
                break
    print(total_step)
    print(agent.replay_total)
    print('*'*30)
    sumTree = agent.memory.tree
    print(len(sumTree.tree))
    print(len(sumTree.data))
    p = sumTree.tree[-sumTree.capacity:]
    p2 = sumTree.tree
    d = sumTree.data[-sumTree.capacity:]
    # print([d1 for d1 in p if d1>0.0])
    print('ddddd',np.unique(p2))
    print(np.max(p2))
    print(sumTree.total_p)

if __name__ == '__main__':
    main()


