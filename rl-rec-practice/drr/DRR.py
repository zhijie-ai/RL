#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/22 上午11:20                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

# https://github.com/BoogieCloud/Deep-RL-Recommendation-System/blob/master/DRR.py

import tensorflow as tf
import numpy as np
from scipy.special import comb
import time

from actor import Actor
from critic import Critic
from noise import OUNoise
from replay_buffer import ReplayBuffer
from simulator import Simulator
from preprocesing import process_data

#state representation module
#require user has same dimension with items (both 1*k dimension)


path1 = 'data/dumped/user_movies.json'# 与各个中心点最近的用户对电影的评分
path2 = 'data/user_train.npy'# u.user 文件中的用户,然后用train_and_test切分成2部分,训练部分用户
path3 = 'data/dumped/movie_items.json'#u.item文件中去除4个字段后的数据,相当与把id当成了index的df
path4 = 'data/user_test.npy'# u.user 文件中的用户,然后用train_and_test切分成2部分,测试部分用户
path5 = 'data/dumped/test_user_movies.json'#和user_movies.json文件格式一样,只是这是经过train_and_test切分之后的测试用户的数据
path6 = 'data/dumped/whole_user_movies.json'# 和user_movies.json文件格式一样,这是u.user文件中的所有用户
history_len = 5
data, item_embed, user_embed = process_data(path1, path2, path3, history_len)
data=data.T
test_data, item_embed_test, user_embed_test = process_data(path5, path4, path3, history_len)
test_data= test_data.T

def gene_actions(item_space, weight_batch):
    """use output of actor network to calculate action list
    Args:
        item_space: recall items, dict: id: embedding
        weight_batch: actor network outputs
    Returns:
        recommendation list
    """
    item_ids = list(item_space.keys())
    item_weights = list(item_space.values())
    max_ids = list()
    for weight in weight_batch:
        score = np.dot(item_weights, weight)
        idx = np.argmax(score)
        max_ids.append(item_ids[idx])
    return max_ids


def gene_action(item_space, weight):
    item_ids = list(item_space.keys())
    item_weights = list(item_space.values())
    score = np.dot(item_weights, weight)
    idx = np.argmax(score)
    return item_ids[idx]


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("reward", episode_reward)
    episode_max_q = tf.Variable(0.)
    tf.summary.scalar("max_q_value", episode_max_q)
    critic_loss = tf.Variable(0.)
    tf.summary.scalar("critic_loss", critic_loss)

    summary_vars = [episode_reward, episode_max_q, critic_loss]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars

def state_mod(state):
    sample = []
    for row in state:
      #print(row)
      x,y = row.shape
      for i in range(x):
        for j in range(i+1, x):
          row = np.vstack([row, np.multiply(row[i], row[j])])
      sample.append(row)

    return np.array(sample)


def learn_from_batch(replay_buffer, batch_size, actor, critic, item_space, action_len, s_dim, a_dim):
    """
    :param replay_buffer:
    :param batch_size:
    :param actor:
    :param critic:
    :param item_space:
    :param action_len:action_item_num,即一次性推荐的item数量
    :param s_dim:
    :param a_dim:
    :return:
    """
    seq_len = np.array([a_dim/action_len], dtype=np.int32)
    if replay_buffer.size() < batch_size:
        pass
    samples = replay_buffer.sample_batch(batch_size)
    #print(samples)
    state_batch = []
    action_batch = []
    reward_batch = []
    n_state_batch = []
    for row in samples:
      state_batch.append(row[0])
      action_batch.append(row[1])
      reward_batch.append(row[2])
      n_state_batch.append(row[3])
    state_batch = np.array(state_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    n_state_batch = np.array(n_state_batch)

    # calculate predicted q value
    new_state = np.concatenate(state_mod(state_batch), axis=0 )
    new_state = np.reshape(new_state, [-1, s_dim])

    action_weights = actor.predict_target(new_state, seq_len)
    n_action_batch = gene_actions(item_space, action_weights)

    new_action_batch = []
    for idx in n_action_batch:
      new_action_batch.append(item_space[idx])
    new_action_batch = np.array(new_action_batch)

    n_new_state = np.concatenate(state_mod(n_state_batch), axis=0 )
    n_new_state = np.reshape(n_new_state, (-1, s_dim))
    target_q_batch = critic.predict_target(n_new_state, new_action_batch.reshape((-1, a_dim)), len_seq=seq_len)

    y_batch = []
    for i in range(batch_size):
        y_batch.append(reward_batch[i] + critic.gamma * target_q_batch[i])
    y_batch = np.array(y_batch)
    y_batch = np.concatenate(y_batch, axis=0)

    # train critic
    q_value, critic_loss, _ = critic.train(new_state, action_batch, np.reshape(y_batch, (batch_size, 1)), seq_len)

    # train actor
    action_weight_batch_for_gradients = actor.predict(new_state, seq_len)
    action_batch_for_gradients = gene_actions(item_space, action_weight_batch_for_gradients)

    action_batch_gra = []
    for idx in action_batch_for_gradients:
      action_batch_gra.append(item_space[idx])
    action_batch_gra = np.array(action_batch_gra)

    a_gradient_batch = critic.action_gradients(new_state, action_batch_gra.reshape((-1, a_dim)), seq_len)

    actor.train(new_state, a_gradient_batch[0], seq_len)

    # update target networks
    actor.update_target_network()
    critic.update_target_network()

    return np.amax(q_value), critic_loss


def train_test(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args, replay_buffer):
    start = time.time()
    # set up summary operators
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # initialize target network weights
    actor.hard_update_target_network()# 硬更新
    critic.hard_update_target_network()# 硬更新

    for i in range(int(args['max_episodes'])):
        ep_reward = 0.
        ep_q_value = 0.
        loss = 0.
        item_space = item_embed
        epoch_num = 0
        # 训练用户.首先把由训练用户日志生成的(s,a,r,s_)放到replay_buffer中,然后再用训练用户与环境交互,得到
        #   数据,再放入replay_buffer中
        for idx in list(data.index):
            epoch_num += 1
            user_space = user_embed[idx]
            if epoch_num == 1:
                # 每个用户都初始化自己的state
                state = env.reset(user_idx=idx, user_embed=user_space)# 当前用户的第四个state的5个item所组成的矩阵
                # print('AAAAAAAAAAAAA',state.shape)#(15, 19)
            # update average parameters every 10 episodes
            for j in range(args['max_episodes_len']):
                weight = actor.predict(np.reshape(state, [-1, s_dim]),
                                       [int(args['embedding'])]) + exploration_noise.noise()
                action = gene_actions(item_space, weight)# 将当前得到的action与item矩阵算score
                reward, n_state = env.step(action[0], idx)

                # print(state,action,reward,n_state)

                replay_buffer.add(state[:args['state_item_num']],
                                  item_embed[str(action[0])],  # need more work
                                  np.array(reward),
                                  np.vstack((n_state[:args['state_item_num'] - 1], n_state[-2:-1])))

                ep_reward += reward[0]
                ep_q_value_, critic_loss = learn_from_batch(replay_buffer, args['batch_size'], actor, critic,
                                                            item_space,
                                                            args['action_item_num'], s_dim, a_dim)
                ep_q_value += ep_q_value_
                loss += critic_loss
                state = n_state
                '''
                if (j + 1) % 50 == 0:
                    print("=========={0} episode of {1} round of {2}-th user: reward {3} loss {4}=========".format(
                        i, j, idx, ep_reward, critic_loss))
                '''

        print('======{0}-th episode, {1} total reward======'.format(i, ep_reward))
        summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: ep_reward,
                                                       summary_vars[1]: ep_q_value,
                                                       summary_vars[2]: loss})
        writer.add_summary(summary_str, i)

    writer.close()
    end = time.time()
    time_consuming = end - start
    # print('Training time = ', time_consuming)

    ep_reward_ = 0.
    ep_q_value = 0.
    loss = 0.
    item_space = item_embed_test
    for idx in list(test_data.index):
        user_space = user_embed_test[idx]
        # update average parameters every 10 episodes
        for j in range(args['test_episodes_len']):
            weight = actor.predict(np.reshape(state, [-1, s_dim]), [int(args['embedding'])]) + exploration_noise.noise()
            action = gene_actions(item_space, weight)
            reward, n_state = env.step(action[0], idx)

            # print(state,action,reward,n_state)
            '''
            replay_buffer.add(state[:args['state_item_num']],
                              item_space[str(action[0])],  
                              np.array(reward),
                              np.vstack((n_state[:args['state_item_num']-1], n_state[-2:-1])))
  
            ep_q_value_, critic_loss = learn_from_batch(replay_buffer, args['batch_size'], actor, critic, item_space,
                                                        args['action_item_num'], s_dim, a_dim)
            ep_q_value += ep_q_value_
            loss += critic_loss
            '''
            ep_reward_ += reward[0]
            state = n_state

    saver = tf.train.Saver()
    model_name = 'drr-1st.ckpt'
    log_dir = 'out/logs/{}'.format(model_name)
    saver.save(sess, log_dir, write_meta_graph=False)
    return ep_reward_, time_consuming


def main(args):
    tf.reset_default_graph()

    with tf.Session() as sess:
        # simulated environment
        env = Simulator()

        # initialize replay memory
        replay_buffer = ReplayBuffer(int(args['buffer_size']))

<<<<<<< HEAD
=======
        # emb的维度*(state的数量+state数量的两两组合)
>>>>>>> bf5ebce... RL 相关
        s_dim = int(args['embedding']) * (
                    int(args['state_item_num']) + int(comb(int(args['state_item_num']), 2)))  ### need more work here
        a_dim = int(args['embedding']) * int(args['action_item_num'])

        actor = Actor(sess, s_dim, a_dim,
                      int(args['batch_size']), int(args['embedding']),
                      int(args['action_item_num']), float(args['tau']),
                      float(args['actor_lr']))

        critic = Critic(sess, s_dim, a_dim,
                        actor.get_num_trainable_vars(), int(args['action_item_num']), float(args['gamma']),
                        float(args['tau']), float(args['critic_lr']))

        exploration_noise = OUNoise(a_dim)# 给action增加一个噪声,相当与探索

        test_reward, time_consuming = train_test(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args,
                                                 replay_buffer)


    return time_consuming, test_reward

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

# saver = tf.train.Saver()
#
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#     # Restore variables from disk.
#     saver.restore(sess, "/tmp/model.ckpt")
#     print("Model restored.")