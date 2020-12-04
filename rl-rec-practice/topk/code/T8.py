#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/11/19 10:02                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np
import pandas as pd

np.random.seed(1126)

ep_rs = np.random.randint(0,10,10)
GAMMA = .9
print(ep_rs)
discounted_ep_rs = np.zeros_like(ep_rs,dtype='float')
running_add = 0
for t in reversed(range(0,len(ep_rs))):
    running_add = running_add * GAMMA+ep_rs[t]
    discounted_ep_rs[t] = running_add

discounted_ep_rs -= np.mean(discounted_ep_rs)
discounted_ep_rs /= np.std(discounted_ep_rs)

print(discounted_ep_rs)

def load_data(path='../data/ratings_1m.dat'):
    def _discount_and_norm_rewards(rewards,gamma=.9):
        rewards = list(rewards)
        discounted_episode_rewards = np.zeros_like(rewards,dtype='float64')
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[t]
            discounted_episode_rewards[t] = cumulative
        # Normalize the rewards
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    ratings = pd.read_csv(path,delimiter='::',index_col=None,header=None,names=['userid','itemid','rating','timestamp'],engine='python')
    ratings.sort_values(by=['userid','timestamp'],inplace=True)
    items = list(sorted(ratings.itemid.unique()))
    key_to_id_item = dict(zip(items,range(len(items))))
    ratings.itemid = ratings.itemid.map(key_to_id_item)
    ratings['rewards'] = ratings.groupby('userid')['rating'].transform(_discount_and_norm_rewards)
    return ratings

df = load_data()
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
print(df.sort_values(by=['userid','timestamp']).head(56))