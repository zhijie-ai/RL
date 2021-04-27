#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/21 15:34                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
# 用训练得到的env模型来训练强化学习模型
from collections import deque
from utils.yjp_decorator import cost_time_def

import numpy as np

from GAN_RL.yjp.code.dqn import DQN
from GAN_RL.yjp.code.env import Enviroment
from GAN_RL.yjp.code.options import get_options
import datetime,os,time
from GAN_RL.yjp.code.dataset import Dataset
import threading
from multiprocessing import Process
from tqdm import tqdm

np.set_printoptions(suppress=True)

lock = threading.Lock()

@cost_time_def
def train_with_random_action(dataset,env,dqn):
    for ind in tqdm(range(0,len(env.train_user),dataset.args.sample_batch_size)):
        end = ind+dataset.args.sample_batch_size
        training_user = env.train_user[ind:end]
        data_collection = dataset.data_collection(training_user)
        # START TRAINING for this batch of users
        # data_collection相当于是从环境中收集到的数据
        num_samples = len(data_collection['user'])
        arr = np.arange(num_samples)
        for n in range(0,num_samples,dataset.args.training_batch_size):
            batch_sample = arr[n:n+dataset.args.training_batch_size]
            states_batch = [data_collection['state'][c] for c in batch_sample]
            user_batch = [data_collection['user'][c] for c in batch_sample]
            action_batch = [data_collection['action'][c] for c in batch_sample]
            y_batch = [data_collection['y'][c] for c in batch_sample]

            q_feed_dict = dataset.data_prepare_for_loss_placeholder(user_batch,states_batch,action_batch,y_batch)
            loss_val = dqn.train_on_batch(q_feed_dict)
            loss_val = np.round(loss_val,10)

            if np.mod(n,250) == 0:
                loss_val = list(loss_val[-dqn.k:])
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print_loss = ' '
                for kkk in range(dqn.k):
                    print_loss += ' %.5g,'
                print(('%s: init itr(%d), training loss:'+print_loss) % tuple([log_time, n]+loss_val))

        end = end if end <len(env.train_user) else len(env.train_user)
        print('finish init iteration!! completed user [%d/%d]' % (end,len(env.train_user)))
    # save model
    dqn.save('init-q')

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
    global user_avg_reward,current_avg_reward,clk_rate, \
        current_avg_clkrate,best_reward

    user_avg_reward = 0.0
    current_avg_reward = 0.0
    clk_rate = 0.0
    current_avg_clkrate = 0.0
    best_reward = 0.0
    thread_u = [[] for _ in range(dataset.args.num_thread)]
    for ii in range(len(user_set)):
        thread_u[ii%dataset.args.num_thread].append(user_set[ii])

    threads = []
    for ii in range(dataset.args.num_thread):
        if dataset.args.compu_type=='thread':
            thread = threading.Thread(target=validation,args=(current_best_reward,dataset,env,dqn,thread_u[ii]))
        else:
            thread = Process(target=validation,args=(current_best_reward,dataset,env,dqn,thread_u[ii]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return np.mean(user_avg_reward),current_avg_reward,np.mean(clk_rate),current_avg_clkrate,best_reward

@cost_time_def
def validation(current_best_reward,dataset,env,dqn,sim_vali_user):
    # initialize empty states
    global user_avg_reward,current_avg_reward,clk_rate, \
        current_avg_clkrate,best_reward

    states = [[] for _ in range(len(sim_vali_user))]
    sim_u_reward = {}

    for t in range(dataset.args.time_horizon):
        sim_vali_user ,states,sim_u_reward = step(dataset,env,dqn,sim_vali_user,states,sim_u_reward)

        if len(sim_vali_user) ==0:
            break

    user_avg_reward,current_avg_reward,clk_rate, \
    current_avg_clkrate,best_reward = env.compute_average_reward(sim_vali_user,sim_u_reward,current_best_reward)

    lock.acquire()
    user_avg_reward += user_avg_reward
    current_avg_reward += current_avg_reward
    clk_rate += clk_rate
    current_avg_clkrate += current_avg_clkrate
    best_reward += best_reward
    lock.release()

@cost_time_def
def train_with_greedy_action(dataset,env,dqn):
    # 用全部数据训练一遍
    current_best_reward = 0.0
    for ind in tqdm(range(0,len(env.train_user),dataset.args.sample_batch_size)):
        end = ind+dataset.args.sample_batch_size
        training_user = env.train_user[ind:end]

        # initialize empty states
        data_collection = dataset.data_collection(training_user,'greedy')

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
            loss_val = dqn.train_on_batch(q_feed_dict)
            loss_val = np.round(loss_val,10)

            if np.mod(n,250) == 0:
                loss_val = list(loss_val[-dqn.k:])
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print_loss = ' '
                for kkk in range(dqn.k):
                    print_loss += ' %.5g,'
                print(('%s: greedy itr(%d), training loss:'+print_loss) % tuple([log_time, n]+loss_val))

        end = end if end <len(env.train_user) else len(env.train_user)
        print('finish iteration!! completed user [%d/%d]' % (end,len(env.train_user)))

        # TEST
        # new_reward = test_during_training(current_best_reward,dataset,env,dqn)
        _,_,_,_,new_reward = multi_compute_validation(current_best_reward,dataset,env,dqn,env.vali_user)
        if new_reward > current_best_reward:
            save_path = os.path.join(dataset.args.model_path, 'best-reward')
            dqn.save('best-reward')
            current_best_reward = new_reward

@cost_time_def
def main(args):
    env = Enviroment(args)
    env.initialize_environment()
    dqn = DQN(env,args)
    dataset = Dataset(args,env,dqn)

    # 参照强化学习的训练逻辑，EE问题。在收集数据的时候兼顾EE问题。此论文的思路将EE问题分开来解决。
    #   首先用随机策略收集数据，其次，在随机策略的训练基础上再使用贪婪策略来训练策略。
    # 首先，根据随机策略来收集并训练
    train_with_random_action(dataset,env,dqn)

    # 使用贪婪策略收集的数据来训练我们的推荐引擎
    train_with_greedy_action(dataset,env,dqn)

    # dqn.restore('init-q')
    # dqn.restore('best-reward')

    # current_best_reward = 0.0
    # print(multi_compute_validation(current_best_reward,dataset,env,dqn,env.test_user[0:10]))




if __name__ == '__main__':
    cmd_args = get_options()
    print('>>>>>>>>>>>',cmd_args,'<<<<<<<<<<<<<<<<<<<<<<')
    main(cmd_args)
    print('finished!!!!!!')