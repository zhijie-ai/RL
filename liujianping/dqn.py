#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2019/2/20 10:13
 =================知行合一=============
'''

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

#Hyper parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5# starting value for epsilon
FINAL_EPSILON = 0.01 #final value of epsilon
REPLAY_SIZE = 10000#experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
    #DQN Agent
    def __init__(self,env):
        #init experience replay
        self.replay_buffer = deque()
        #init some parameters
        self.time_step=0
        self.epsilon = INITIAL_EPSILON
        self.state_dim=env.observation_space.shape[0]
        self.action_dim=env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        #Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # network weights
        w1 = self.weight_variable([self.state_dim,20])
        b1 = self.bias_variable([20])
        w2 = self.weight_variable([20,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        #input layer
        self.state_input = tf.placeholder('float',[None,self.state_dim])
        #hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,w1)+b1)
        #Q value layer
        self.Q_value = tf.matmul(h_layer,w2)+b2

    def create_training_method(self):
        self.action_input = tf.placeholder('float',[None,self.action_dim])#ont hot presentation
        self.y_input = tf.placeholder('float',[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input-Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            loss = self.train_Q_network()
            return loss

    def train_Q_network(self):
        self.time_step +=1
        # step 1:obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # step 2:calulate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        # print('AAAAAAAAA',Q_value_batch)
        # print('BBBBBBBBBB',action_batch)
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i]+GAMMA*np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
        })
        loss = self.session.run(self.cost,feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
        })
        return loss

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict={self.state_input:[state]})[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/10000
            return random.randint(0,self.action_dim-1)
        else :
            self.epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/10000
            return np.argmax(Q_value)

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict={self.state_input:[state]})[0])

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01,shape=shape)
        return tf.Variable(initial)



# Hyper Parameters
ENV_NAME='CartPole-v0'
EPISODE = 3000 # Episode limitation
STEP = 3000#step limitation in an episode
TEST = 10# The number of experience test every 100 episode

losses = []
def main():
    # initialize OpenAI Gym env dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)
    print(env.action_space)
    print(env.observation_space)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state) # e-greedy action for train
            next_state,reward,done,_ = env.step(action)
            # Define reward for agent
            reward = -1 if done else 0.1
            loss = agent.perceive(state,action,reward,next_state,done)
            losses.append(loss)
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode %100 == 0:
            total_reward = 0
            for i in range(TEST):
                # env.render()
                action = agent.action(state) # direct action for test
                state,reward,done,_ = env.step(action)
                print(reward)
                total_reward += reward
                if done:
                    break

            ave_reward = total_reward/TEST
            print('episode:',episode,'Evaluation Average Reward:',ave_reward)

    plot(losses)

def plot(data):
    import matplotlib.pyplot as plt
    num=10
    ran = 15
    data = [np.mean(data[ind-num:ind+num]) for ind ,val in enumerate(data) if ind%ran==num]
    plt.plot(range(len(data)),data)
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('dqn_loss.jpg')


if __name__ == '__main__':
    main()


