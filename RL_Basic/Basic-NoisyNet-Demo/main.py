#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/17 22:11                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import matplotlib.pyplot as plt
import gym

import numpy as np
import tensorflow as tf
import pickle

from Config import NoisyNetDQNConfig
from NosiyNetDQN import NoisyNetDQN

def map_scores(dqfd_scores=None,ddqn_scores=None,xlabel=None,ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores,'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores,'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()

def BreakOut_NosiyNetDQN(index,env):
    with tf.variable_scope('DQFD_'+str(index)):
        agent = NoisyNetDQN(env,NoisyNetDQNConfig())
        scores=[]
        for e in range(NoisyNetDQNConfig.episode):
            done = False
            score = 0 # sum of reward in one episode
            state = env.reset()
            # while done is False
            last_lives=5
            throw=True
            items_buffer=[]
            while not done:
                env.render()
                action=1 if throw else agent.noisy_action(state)
                next_state,real_reward,done,info=env.step(action)
                lives=info['ale.lives']
                train_reward= 1 if throw else -1 if lives<last_lives else real_reward
                score += real_reward
                throw = lives<last_lives
                last_lives = lives
                # agent.perceive(state,action,train_reward,next_state,done) # miss:-1 break nothing:0
                items_buffer.append([state,action,next_state,done]) # miss:-1 break:reward nothing:0
                state=next_state
                if train_reward !=0:#train when miss the ball or score or throw the ball in the begining
                    print('len(items_buffer):',len(items_buffer))
                    for item in items_buffer:
                        agent.perceive(item[0],item[1],-1 if throw else train_reward,item[2],item[3])
                        agent.train_q_network(update=False)
                    items_buffer=[]
            scores.append(score)
            agent.sess.run(agent.update_target_net)
            print('episode:',e,' score:',score,' memory length:',len(agent.replay_buffer))

        return scores

if __name__ == '__main__':
    env = gym.make('Breakout-v0')#打砖块游戏
    NosiyNetDQN_sum_scores=np.zeros(NoisyNetDQNConfig.episode)

    for i in range(NoisyNetDQNConfig.iteration):
        scores= BreakOut_NosiyNetDQN(i,env)
        dqfd_sum_scores=[a+b for a,b in zip(scores,NosiyNetDQN_sum_scores)]
    NoisyNetDQN_mean_scores = NosiyNetDQN_sum_scores/NoisyNetDQNConfig.iteration
    with open('../DQFD/NoisyNetDQN_mean_scores.p','wb') as f:
        pickle.dump(NoisyNetDQN_mean_scores,f,protocol=2)