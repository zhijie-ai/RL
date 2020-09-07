#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/22 下午3:52                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from drr.preprocesing import *

path1 = 'data/dumped/user_movies.json'# 与各个中心点最近的用户对电影的评分
path2 = 'data/user_train.npy'# u.user 文件中的用户,然后用train_and_test切分成2部分,训练部分用户
path3 = 'data/dumped/movie_items.json'#u.item文件中去除4个字段后的数据,相当与把id当成了index的df
path4 = 'data/user_test.npy'# u.user 文件中的用户,然后用train_and_test切分成2部分,测试部分用户
path5 = 'data/dumped/test_user_movies.json'#和user_movies.json文件格式一样,只是这是经过train_and_test切分之后的测试用户的数据
path6 = 'data/dumped/whole_user_movies.json'# 和user_movies.json文件格式一样,这是u.user文件中的所有用户
history_len = 5
# test_data, item_embed_test, user_embed_test = process_data(path5, path4, path3, history_len)
# data, item_embed, user_embed = process_data(path1, path2, path3, history_len)

whole_data, item_embed, user_embed = process_data(path6, path2, path3, history_len)
whole_data = whole_data.T # 行代表用户数,列为4,其中每列又为一个向量
print(whole_data.shape,len(user_embed))
print(item_embed.keys())
print(set(whole_data.loc[1]['action_float']))

# ==================Simulator=====================
print('=-===========================================')
from drr.simulator import Simulator
simu = Simulator(item_embed,whole_data)
init_state = simu.reset(1,0)
print(init_state.shape)

action = np.array(item_embed[str(272)])
user_idx = 1

a3 = np.array(item_embed[str(273)])
a4 = np.array(item_embed[str(274)])
a5 = np.array(item_embed[str(275)])
a6 = np.array(item_embed[str(276)])
r = simu.simulate_reward((simu.current_state.reshape((1,15*19)),
                                                 action.reshape((1,1*19))),user_idx)
r3 = simu.simulate_reward((simu.current_state.reshape((1,15*19)),
                                                 a3.reshape((1,1*19))),user_idx)
r4 = simu.simulate_reward((simu.current_state.reshape((1,15*19)),
                                                 a4.reshape((1,1*19))),user_idx)
r5 = simu.simulate_reward((simu.current_state.reshape((1,15*19)),
                                                 a5.reshape((1,1*19))),user_idx)
r6 = simu.simulate_reward((simu.current_state.reshape((1,15*19)),
                                                 a6.reshape((1,1*19))),user_idx)
print(r,r3,r4,r5,r6)

# ======================Actor==============================
from drr.actor import Actor
from scipy.special import comb
import tensorflow as tf

print('--------------ACTOR----------------')
args = {}
args['embedding'] = 19
args['state_item_num'] = 5
args['action_item_num'] = 2  # currently only generate 1 item
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
s_dim = int(args['embedding']) * (
                    int(args['state_item_num']) + int(comb(int(args['state_item_num']), 2)))  ### need more work here
a_dim = int(args['embedding']) * int(args['action_item_num']) #action_item_num:代表推荐几个item

sess = tf.Session()
actor = Actor(sess, s_dim, a_dim,
                      int(args['batch_size']), int(args['embedding']),
                      int(args['action_item_num']), float(args['tau']),
                      float(args['actor_lr']))

