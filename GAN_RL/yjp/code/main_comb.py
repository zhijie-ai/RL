#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/6/9 17:28                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
# 用训练得到的env模型来训练强化学习模型
from utils.yjp_decorator import cost_time_minute
import pickle
import numpy as np
from collections import deque

from GAN_RL.yjp.code.dqn import DQN
from GAN_RL.yjp.code.env import Enviroment
from GAN_RL.yjp.code.options import get_options
import datetime,os,time
from GAN_RL.yjp.code.dataset import Dataset
import threading
from multiprocessing import Process,Pool
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

np.set_printoptions(suppress=True)

import warnings
warnings.filterwarnings('ignore',category=FutureWarning,module='tensorflow')
warnings.filterwarnings('ignore',category=UserWarning,module='tensorflow')
warnings.filterwarnings('ignore',category=DeprecationWarning,module='tensorflow')
warnings.filterwarnings('ignore')

lock = threading.Lock()

def step(dataset,env,dqn,sim_vali_user,states,sim_u_reward):
    max_q_feed_dict,states_tr,history_order_tr,history_user_tr = dataset.format_max_q_feed_dict(sim_vali_user,states)

    # 1. find best recommend action
    max_action,max_action_disp_feature = dqn.choose_action(max_q_feed_dict)
    # 2. compute reward
    disp_2d_split_user = np.kron(np.arange(len(sim_vali_user)),np.ones(dqn.k)).astype(int)
    reward_feed_dict={}
    reward_feed_dict[env.placeholder['Xs_clicked']]=states_tr
    reward_feed_dict[env.placeholder['history_order_indices']]=history_order_tr
    reward_feed_dict[env.placeholder['history_user_indices']]=history_user_tr
    reward_feed_dict[env.placeholder['disp_2d_split_user_ind']]=disp_2d_split_user
    reward_feed_dict[env.placeholder['disp_action_feature']]=max_action_disp_feature
    _, transition_p,u_disp,_ = env.conpute_reward(reward_feed_dict)
    reward_u = np.reshape(u_disp,[-1,dqn.k])
    # 5. sample reward and new states
    sim_vali_user ,states,sim_u_reward = env.sample_new_states(sim_vali_user,states,transition_p,reward_u,sim_u_reward,max_action,env.k)

    return sim_vali_user ,states,sim_u_reward

def multi_compute_validation(current_best_reward,dataset,env,dqn,user_set):
    global user_sum_reward,sum_clk_rate

    user_sum_reward = 0.0
    sum_clk_rate = 0.0
    thread_u = [[] for _ in range(dataset.args.num_thread)]
    for ii in range(len(user_set)):
        thread_u[ii%dataset.args.num_thread].append(user_set[ii])

    threads = []
    for ii in range(dataset.args.num_thread):
        if dataset.args.compu_type=='thread':
            thread = threading.Thread(target=validation,args=(dataset,env,dqn,thread_u[ii]))
        else:
            thread = Process(target=validation,args=(dataset,env,dqn,thread_u[ii]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    user_avg_reward = user_sum_reward/len(user_set)
    if user_avg_reward> current_best_reward:
        current_best_reward = user_avg_reward

    return user_avg_reward,sum_clk_rate/len(user_set),current_best_reward

# 新启线程来运行
@cost_time_minute##1459.6893651485s
def validation(dataset,env,dqn,vali_user):
    # initialize empty states
    global user_sum_reward,sum_clk_rate

    sim_vali_user = vali_user.copy()
    states = [[] for _ in range(len(sim_vali_user))]
    sim_u_reward = {}

    for t in range(dataset.args.vali_time_horizon):
        sim_vali_user ,states,sim_u_reward = step(dataset,env,dqn,sim_vali_user,states,sim_u_reward)

        if len(sim_vali_user) ==0:
            break

    user_sum_reward,clk_sum_rate = env.compute_average_reward(vali_user,sim_u_reward)

    lock.acquire()
    user_sum_reward += user_sum_reward
    sum_clk_rate += clk_sum_rate

    lock.release()

@cost_time_minute
def validation_train(current_best_reward,dataset,env,dqn,vali_user):
    sim_u_reward = {}
    sim_vali_user = vali_user.copy()
    states = [[] for _ in range(len(sim_vali_user))]

    for t in range(dataset.args.vali_time_horizon):
        sim_vali_user ,states,sim_u_reward = step(dataset,env,dqn,sim_vali_user,states,sim_u_reward)

        if len(sim_vali_user) ==0:
            break

    user_sum_reward,clk_sum_rate = env.compute_average_reward(vali_user,sim_u_reward)

    user_avg_reward = user_sum_reward/len(vali_user)
    if user_avg_reward>current_best_reward:
        current_best_reward =user_avg_reward

    return user_avg_reward, clk_sum_rate/len(vali_user) ,current_best_reward

# @cost_time_minute#0.5478950063 m
def train_on_epoch(data_collection,dataset,dqn,loss):
    # START TRAINING for this batch of users
    num_samples = len(data_collection['user'])
    arr = np.arange(num_samples)
    for n in range(0,num_samples,dataset.args.training_batch_size):
        batch_sample = arr[n:n+dataset.args.training_batch_size]
        states_batch = [data_collection['state'][c] for c in batch_sample]
        user_batch = [data_collection['user'][c] for c in batch_sample]
        action_batch = [data_collection['action'][c] for c in batch_sample]
        y_batch = [data_collection['y'][c] for c in batch_sample]

        q_feed_dict = dataset.data_prepare_for_loss_placeholder(user_batch,states_batch,action_batch,y_batch)
        loss_val,step = dqn.train_on_batch(q_feed_dict)
        loss_val = np.round(loss_val,10)

        if np.mod(step//10,10) == 0:#因为global_step是10的倍数，训练一次，实际上minimize 10次,而minimine 10次相当于是1次train
            loss_val = list(loss_val[-dqn.k:])
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print_loss = ' '
            for kkk in range(dqn.k):
                loss[kkk].append(loss_val[kkk])
                print_loss += ' %.5g,'
            print(('%s: itr(%d), training loss:'+print_loss) % tuple([log_time, step//10]+loss_val))

@cost_time_minute
def train(dataset,dqn,train_user):
    loss = [[] for _ in range(dqn.k)]

    data_collection = dataset.data_collection_all(train_user)
    file = open('data/data_collection_comb.pkl'.format(), 'wb')
    pickle.dump(data_collection, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    print('train data length:{}'.format(len(data_collection['user'])))

    for _ in tqdm(range(dataset.args.epoch)):
        train_on_epoch(data_collection,dataset,dqn,loss)

    dqn.save('best-reward-comb-{}'.format(dataset.args.epoch))
    return loss

@cost_time_minute
def main(args):
    env = Enviroment(args)
    env.initialize_environment()
    dqn = DQN(env,args)
    dataset = Dataset(args,env,dqn)

    train_user = np.random.choice(env.train_user,int(len(env.train_user)*0.2),replace=False)
    loss_greedy = train(dataset,dqn,train_user)
    file = open('loss/loss_comb_{}_{}_{}_{}_filtered.pkl'.format(args.noclick_weight,args.epoch,args.dqn_lr,args.training_batch_size), 'wb')
    pickle.dump(loss_greedy, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

    print(multi_compute_validation(0.0,dataset,env,dqn,env.vali_user))
    print(validation_train(0.0,dataset,env,dqn,env.vali_user))


if __name__ == '__main__':
    cmd_args = get_options()
    t1 = time.time()
    print('>>>>>>>>>>>',cmd_args,'<<<<<<<<<<<<<<<<<<<<<<')
    main(cmd_args)
    t2 = time.time()
    print('finished!!!!!!,time cost:{} m'.format((t2-t1)/60))