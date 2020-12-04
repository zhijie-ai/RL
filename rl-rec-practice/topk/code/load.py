#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/10/30 10:19                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
# 用这种自定义模型的方式保存的json 或者yaml模型文件时，有以reward命名的layer，如果直接采用model_from_yaml方式加载时报错,可以重新把模型定义一遍
# 读取模型网络结构
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_yaml
with open("model/model.yaml", "r") as f:
    yaml_string = f.read()  # 读取本地模型的yaml文件
model = model_from_yaml(yaml_string)  # 创建一个模型
filepath = "model/weights2.best.hdf5"
model.load_weights(filepath)
print(model)

def load_data_movie_length(path='../data/ratings_1m.dat', time_step=15, gamma=.9):
    historys = []
    actions = []
    rewards = []

    def _discount_and_norm_rewards(rewards):
        discounted_episode_rewards = np.zeros_like(rewards, dtype='float64')
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[t]
            discounted_episode_rewards[t] = cumulative
        # Normalize the rewards
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    ratings = pd.read_csv(path, delimiter='::', index_col=None, header=None,
                          names=['userid', 'itemid', 'rating', 'timestamp'], engine='python')
    print(ratings.head())

    items = list(sorted(ratings.itemid.unique()))
    key_to_id_item = dict(zip(items, range(len(items))))
    id_to_key_item = dict(zip(range(len(items)), items))
    users = list(set(sorted(ratings.userid.unique())))
    key_to_id_user = dict(zip(users, range(len(users))))
    id_to_key_user = dict(zip(range(len(users)), users))

    ratings.userid = ratings.userid.map(key_to_id_user)
    ratings.itemid = ratings.itemid.map(key_to_id_item)
    ratings = ratings.sort_values(by=['timestamp']).drop('timestamp', axis=1).groupby('userid')
    for _, df in ratings:
        r = _discount_and_norm_rewards(df.rating.values)
        items = df.itemid.values
        for i in range(len(items) - time_step):
            historys.append(list(items[i:i + time_step]))
            actions.append(items[i + time_step])
            rewards.append(r[i + time_step])

    return np.array(historys), np.array(actions), np.array(rewards)

historys, actions, rewards = load_data_movie_length()
pred = model.predict([historys[0:15],actions[0:15]])
print(pred,rewards[14])