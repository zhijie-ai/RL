#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/1 23:02                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 强化学习精要中第五章的代码实例
import numpy as np
import gym
from gym.spaces import Discrete

class SnakeEnv(gym.Env):
    SIZE = 100

    """
    ladder_num:梯子数量
    dices:不同投掷方法的最大值
    """
    def __init__(self,ladder_num,dices):
        self.ladder_num = ladder_num
        self.dices = dices
        self.ladders = dict(np.random.randint(1,self.SIZE,size=(self.ladder_num,2)))
        self.observation_space = Discrete(self.SIZE+1)
        self.action_space = Discrete(len(dices))

        for k,v in self.ladders.items():
            # self.ladders[v] = k
            print('ladders info:')
            print(self.ladders)
            print('dice ranges')
            print(self.dices)
        self.pos =1

    def reset(self):
        self.pos = 1
        return self.pos

    def step(self,a):
        step = np.random.randint(1,self.dices[a]+1)
        self.pos += step
        if self.pos == 100:
            return 100,100,1,{}
        elif self.pos > 100:
            self.pos = 200-self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        return self.pos,-1,0,{}

    def reward(self,s):
        if s == 100:
            return 100
        else:
            return -1

    def render(self):
        pass

if __name__ == '__main__':
    env = SnakeEnv(10,[3,6])
    env.reset()
    while True:
        state,reward,terminate ,_ = env.step(0)
        print(reward,state)
        if terminate == 1:
            break