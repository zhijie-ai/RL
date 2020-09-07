#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#        2019/3/12 14:25                      #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import gym
import time

env = gym.make('CartPole-v0')#Discrete(2)#倒立摆
# env = gym.make('MountainCar-v0')#Discrete(3)#小车过山
# env = gym.make('Acrobot-v1')#Discrete(3)
# env = gym.make('Pendulum-v0')#Box(1,)#单摆模型
# env = gym.make('BattleZone-ramNoFrameskip-v4')#Discrete(18)
# env = gym.make('Acrobot-v1')#Discrete(3)
# env = gym.make('AirRaid-ram-v0')#Discrete(6)
# env = gym.make('AirRaid-ramDeterministic-v0')#Discrete(6)
# env = gym.make('AirRaid-ramDeterministic-v4')#Discrete(6)
# env = gym.make('AirRaid-v0')#Discrete(6)
# env = gym.make('AirRaidNoFrameskip-v0')#Discrete(6)
# env = gym.make('AirRaidNoFrameskip-v4')#Discrete(6)
# env = gym.make('Alien-ram-v0')#Discrete(18)
# env = gym.make('Amidar-ramDeterministic-v4')#Discrete(10)
# env = gym.make('AirRaid-ram-v4')#Discrete(10)
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


init = env.reset()
print(init,env.action_space.sample())
for i in range(10):
    # env.render()
    ob, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
