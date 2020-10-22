#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/22 下午6:18                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
from scipy.special import comb
import numpy as np

from actor import Actor
from critic import Critic
from noise import OUNoise
from replay_buffer import ReplayBuffer
from simulator import Simulator
from preprocesing import process_data
from DRR import train_test

def main(args):
    tf.reset_default_graph()

    with tf.Session() as sess:
        # simulated environment
        env = Simulator(item_embed,whole_data)

        # initialize replay memory
        replay_buffer = ReplayBuffer(int(args['buffer_size']))


        #19*(5+C(5,2)=10)=19*15
        #emb的维度*(state的数量+state数量的两两组合)
        # state_item_num 每个state中包含item的个数
        s_dim = int(args['embedding']) * (
                    int(args['state_item_num']) + int(comb(int(args['state_item_num']), 2)))  ### need more work here
        a_dim = int(args['embedding']) * int(args['action_item_num']) #action_item_num:代表推荐几个item

        actor = Actor(sess, s_dim, a_dim,
                      int(args['batch_size']), int(args['embedding']),
                      int(args['action_item_num']), float(args['tau']),
                      float(args['actor_lr']))

        critic = Critic(sess, s_dim, a_dim,
                        actor.get_num_trainable_vars(), int(args['action_item_num']), float(args['gamma']),
                        float(args['tau']), float(args['critic_lr']))

        exploration_noise = OUNoise(a_dim)

        test_reward, time_consuming = train_test(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args,
                                                 replay_buffer)


    return time_consuming, test_reward


if __name__ == '__main__':
    args = {}
    args['embedding'] = 19
    args['state_item_num'] = 5
    args['action_item_num'] = 1  # currently only generate 1 item
    args['actor_lr'] = 0.0001
    args['critic_lr'] = 0.001
    args['gamma'] = 0.9
    args['tau'] = 0.001
    args['buffer_size'] = 1000000
    args['batch_size'] = 1
    args['max_episodes'] = 10
    args['max_episodes_len'] = 50
    args['test_episodes_len'] = 10
    args['summary_dir'] = 'out/logs/results'
    args['summary_dir_test'] = 'out/logs/test_results'

    # 训练的用户数
    user_train = user = np.load('data/user_train.npy', allow_pickle=True)

    clusters_accuracy = {}

    # num_of_clusters = [2, 4, 16, 64, 128, 256, 512, len(user_train)]
    num_of_clusters = [len(user_train)]

    for num in num_of_clusters:
        # cluster_data(num)

        path1 = 'data/dumped/user_movies.json'  # 与各个中心点最近的用户对电影的评分
        path2 = 'data/user_train.npy'  # u.user 文件中的用户,然后用train_and_test切分成2部分,训练部分用户
        path3 = 'data/dumped/movie_items.json'  # u.item文件中去除4个字段后的数据,相当与把id当成了index的df
        path4 = 'data/user_test.npy'  # u.user 文件中的用户,然后用train_and_test切分成2部分,测试部分用户
        path5 = 'data/dumped/test_user_movies.json'  # 和user_movies.json文件格式一样,只是这是经过train_and_test切分之后的测试用户的数据
        path6 = 'data/dumped/whole_user_movies.json'  # 和user_movies.json文件格式一样,这是u.user文件中的所有用户
        history_len = 5 #采取前5个预测
        test_data, item_embed_test, user_embed_test = process_data(path5, path4, path3, history_len)
        test_data = test_data.T
        data, item_embed, user_embed = process_data(path1, path2, path3, history_len)
        data = data.T
        whole_data, item_embed, user_embed = process_data(path6, path2, path3, history_len)
        whole_data = whole_data.T

        time_consuming, test_reward = main(args)
        print('total training time {0}, total test reward {1}'.format(time_consuming, test_reward))

        clusters_accuracy[num] = [time_consuming, test_reward]