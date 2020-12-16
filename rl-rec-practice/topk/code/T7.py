#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/11/18 10:06                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------

import pandas as pd
import numpy as np
np.random.seed(1126)

def _discount_and_norm_rewards(rewards,gamma=1.):
    rewards = list(rewards)
    print('AAAA',rewards)
    discounted_episode_rewards = np.zeros_like(rewards,dtype='float64')
    cumulative = 0
    for t in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[t]
        discounted_episode_rewards[t] = cumulative
    # Normalize the rewards
    # discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    # discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return discounted_episode_rewards

def T(x):
    print('AAAA',x)
    return x.sum()

df = pd.DataFrame({'name':np.random.choice(list('ABCD'),20),'ratings':np.random.randint(0,5,20),'ts':np.random.randint(1,1000,20),'gender':np.random.binomial(1,0.5,20)})
df.sort_values(by=['name','ts'],inplace=True)
# print(df.groupby('name').count())
# print(df)
# df['rewards']=df.groupby('name')['ratings'].transform(_discount_and_norm_rewards)
# print(df)
print(df.groupby('name').transform(T))

'''
总结:
print(df.groupby('name').transform(T)) 会把每个col依次传入
'''