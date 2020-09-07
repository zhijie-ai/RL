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

# env = gym.make('CartPole-v0')#Discrete(2)
# env = gym.make('MountainCar-v0')#Discrete(3)
# env = gym.make('Acrobot-v1')#Discrete(3)
# env = gym.make('Pendulum-v0')#Box(1,)
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
# print(env.action_space)

games = open('data/gym_env.txt').readlines()

n = 0
with open('data/gym_env_action_space.txt','w',encoding='utf8') as f:
    for game in games:
        game = game.strip()
        try:
            env = gym.make(game)  # Discrete(10)
            f.write('observation space: {},action space: {}\n'.format(env.observation_space,env.action_space))
        except Exception as e:
            n += 1
            print('=========',e)
            f.write(str(e) + '\n')

    print('total failed env {}'.format(n))
    f.write('total failed env {} \n'.format(n))
