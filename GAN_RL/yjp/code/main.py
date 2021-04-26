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

np.set_printoptions(suppress=True)

@cost_time_def
def data_collection_random_action(data_collection,args, env, dqn, states,training_user):
    # (1) first step: initialize Q as the expected rwd function
    # current_best_reward = 0.0
    for t in range(args.time_horizon):
        data_collection['state'].extend(states)
        data_collection['user'].extend(training_user)
        # prepare to feed max_Q
        states_tr,history_order_tr,history_user_tr = dqn.form_init_Q_feed_dict(training_user,states)

        # 1. sample random action，模仿推荐系统推荐的action
        # feature_space 保存的是和用户相关的sku的特征:其实就是sku的embedding向量
        random_action = [[] for _ in range(len(training_user))]
        random_action_feature = []
        # 每个用户曝光10个item
        for u_i in range(len(training_user)):
            user_i = training_user[u_i]
            # 从用户曝光过的sku中随机选出k个sku作为推荐，模拟推荐引擎的选择
            random_action[u_i] = np.random.choice(list(set(np.arange(len(env.feature_space[user_i])))-set(states[u_i])),dqn.k,replace=False).tolist()
            random_action_feature += [env.feature_space[user_i][jj] for jj in random_action[u_i]]

        # 为了过滤第一次就不满足个数的情况。比如当用户相关的sku个数不足10个时，np.random.choice会报错，所以过滤这部分数据，其实，刚开始的时候，用户相关的
        #   sku个数至少应该满足10个要求的。过滤之后，只剩90个user,如果没有下面的代码，则disp_2d_split_user还是100个，因此会报错。
        #   这种方式不行，应该在最开始的时候满足用户相关的sku个数大于10的阈值。states_tr中的training_user的数量没有减少

        best_action = [random_action,random_action_feature]
        data_collection['action'].extend(best_action[0])

        #2. compute expected immediate reward
        disp_2d_split_user = np.kron(np.arange(len(training_user)),np.ones(dqn.k))
        # reward_feed_dict = {Xs_clicked: [], history_order_indices: [], history_user_indices: [], disp_2d_split_user_ind: [], disp_action_feature:[]}
        reward_feed_dict={}
        reward_feed_dict[env.placeholder['Xs_clicked']]=states_tr
        reward_feed_dict[env.placeholder['history_order_indices']]=history_order_tr
        reward_feed_dict[env.placeholder['history_user_indices']]=history_user_tr
        reward_feed_dict[env.placeholder['disp_2d_split_user_ind']]=disp_2d_split_user
        reward_feed_dict[env.placeholder['disp_action_feature']]=best_action[1]

        # 为exp操作前的reward*对应的权重
        # Reward_r = tf.segment_sum(tf.multiply(u_disp, p_disp), disp_2d_split_user_ind)
        # 计算的是用户的总reward
        best_action_reward,transition_p,_,_ = env.conpute_reward(reward_feed_dict)

        # 4. save to memory
        y_value = best_action_reward
        data_collection['y'].extend(y_value.tolist())##y存的是用户推荐引擎推荐的10个item，也即对用户曝光的10个item总的reward

        # 5. sample new states
        remove_set = []
        for j in range(len(training_user)):
            if len(env.feature_space[training_user[j]])-len(states[j]) <= dqn.k+1:
                remove_set.append(j)

            disp_item = best_action[0][j]
            #transition_p[j, :]得到的是每个用户对k个item的权重
            # 如果np.sum(transition_p[j,:]为1，则说明当前用户一定是选了一个，注意力被平均的分配到了10个item身上。
            # 如果和不为1，则说明当前用户对此时的10个sku并不感兴趣
            no_click = [max(1.0-np.sum(transition_p[j,:]),0.0)]
            prob = np.array(transition_p[j,:].tolist()+no_click)
            prob = prob/float(prob.sum())
            # 模拟用户的选择
            rand_choice = np.random.choice(disp_item+[-100],1,p = prob)
            if rand_choice != -100:
                states[j] += rand_choice.tolist()

        previous_size = len(training_user)
        states = [states[j] for j in range(previous_size) if j not in remove_set]
        training_user = [training_user[j] for j in range(previous_size) if j not in remove_set]

    return data_collection

@cost_time_def
def format_max_q_feed_dict(user_set,states,dqn,env):
    action_mean_tr, action_std_tr, action_user_indice_tr, action_tensor_indice_tr, action_shape_tr, \
    action_space_tr, states_tr, history_order_tr, history_user_tr, action_cnt_tr, \
    action_space_cnt_tr, action_id_tr = dqn.form_max_q_feed_dict(user_set, states)

    max_q_feed_dict = {}
    max_q_feed_dict[dqn.placeholder['all_action_id']]=action_id_tr
    max_q_feed_dict[dqn.placeholder['all_action_user_indices']]=action_user_indice_tr
    max_q_feed_dict[dqn.placeholder['all_action_tensor_indices']]=action_tensor_indice_tr
    max_q_feed_dict[dqn.placeholder['all_action_tensor_shape']]=action_shape_tr
    max_q_feed_dict[dqn.placeholder['current_action_space']]=action_space_tr
    max_q_feed_dict[dqn.placeholder['action_space_mean']]=action_mean_tr
    max_q_feed_dict[dqn.placeholder['action_space_std']]=action_std_tr
    max_q_feed_dict[env.placeholder['Xs_clicked']]=states_tr
    max_q_feed_dict[env.placeholder['history_order_indices']]=history_order_tr
    max_q_feed_dict[env.placeholder['history_user_indices']]=history_user_tr
    max_q_feed_dict[dqn.placeholder['action_count']]=action_cnt_tr
    max_q_feed_dict[dqn.placeholder['action_space_count']]=action_space_cnt_tr

    return max_q_feed_dict,states_tr,history_order_tr,history_user_tr

@cost_time_def
def data_collection_greedy_action(data_collection,args, env, dqn, states,training_user):
    for t in range(args.time_horizon):
        data_collection['state'].extend(states)
        data_collection['user'].extend(training_user)
        # prepare to feed max_Q
        max_q_feed_dict,states_tr,history_order_tr,history_user_tr = format_max_q_feed_dict(training_user,states,dqn,env)


        # 1. find best recommend action
        max_action,max_action_disp_feature = dqn.choose_action(max_q_feed_dict)
        data_collection['action'].extend(max_action.tolist())
        # 2. compute reward
        disp_2d_split_user = np.kron(np.arange(len(training_user)),np.ones(dqn.k)).astype(int)
        reward_feed_dict={}
        reward_feed_dict[env.placeholder['Xs_clicked']]=states_tr
        reward_feed_dict[env.placeholder['history_order_indices']]=history_order_tr
        reward_feed_dict[env.placeholder['history_user_indices']]=history_user_tr
        reward_feed_dict[env.placeholder['disp_2d_split_user_ind']]=disp_2d_split_user
        reward_feed_dict[env.placeholder['disp_action_feature']]=max_action_disp_feature
        _, transition_p,u_disp,_ = env.conpute_reward(reward_feed_dict)
        reward_u = np.reshape(u_disp,[-1,dqn.k])
        # 3. sample new states
        states, training_user, old_training_user, next_states, sampled_reward, remove_set = \
            env.sample_new_states_for_train(training_user, states, transition_p, reward_u, max_action, dqn.k)

        # 4. compute one-step delayed reward
        max_q_feed_dict,_,_,_ = format_max_q_feed_dict(old_training_user,next_states,dqn,env)
        max_q_value = dqn.get_max_q_value(max_q_feed_dict)

        #4. save to memory
        y_value = sampled_reward+args.gamma*max_q_value
        data_collection['y'].extend(y_value.tolist())

    return data_collection

@cost_time_def
def train_with_random_action(args,env,dqn):
    training_user = np.random.choice(env.train_user, args.sample_batch_size, replace=False).tolist()
    states = [[] for _ in range(len(training_user))]
    data_size_init = len(training_user)*args.time_horizon + 100
    data_collection = {'user':deque(maxlen=data_size_init),'state':deque(maxlen=data_size_init),
                       'action':deque(maxlen=data_size_init),'y':deque(maxlen=data_size_init)}
    data_collection = data_collection_random_action(data_collection,args, env, dqn, states, training_user)
    # START TRAINING for this batch of users
    num_samples = len(data_collection['user'])
    batch_iterations = int(np.ceil(num_samples / args.training_batch_size))
    # data_collection相当于是从环境中收集到的数据
    for n in range(batch_iterations):
        batch_sample = np.random.choice(len(data_collection['user']),args.training_batch_size,replace=False)
        states_batch = [data_collection['state'][c] for c in batch_sample]
        user_batch = [data_collection['user'][c] for c in batch_sample]
        action_batch = [data_collection['action'][c] for c in batch_sample]
        y_batch = [data_collection['y'][c] for c in batch_sample]
        loss_val = train_on_batch_placeholder(user_batch,states_batch,action_batch,y_batch,dqn,env)

        if np.mod(n,250) == 0:
            loss_val = list(loss_val[-dqn.k:])
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print_loss = ' '
            for kkk in range(dqn.k):
                print_loss += ' %.5g,'
            print(('%s: init itr(%d), training loss:'+print_loss) % tuple([log_time, n]+loss_val))

    print('finish init iteration!!!!!!!')
    # save model
    dqn.save('init-q')

@cost_time_def
def train_on_batch_placeholder(user_batch,states_batch,action_batch,y_batch,dqn,env):
    action_ids_k_tr,action_space_tr,states_feature_tr,history_order_tr, \
    history_user_tr,action_mean_tr,action_std_tr = dqn.form_loss_feed_dict(user_batch,states_batch,action_batch)

    q_feed_dict={}
    q_feed_dict[dqn.placeholder['current_action_space']]=action_space_tr#(120,65)
    q_feed_dict[dqn.placeholder['action_space_mean']]=action_mean_tr#(5,65)
    q_feed_dict[dqn.placeholder['action_space_std']]=action_std_tr#(5,65)
    q_feed_dict[env.placeholder['Xs_clicked']]=states_feature_tr#(21,65)
    q_feed_dict[env.placeholder['history_order_indices']]=history_order_tr#(21,)
    q_feed_dict[env.placeholder['history_user_indices']]=history_user_tr#(21,)
    q_feed_dict[dqn.placeholder['y_label']]=y_batch#(5,)


    action_k_id = ['action_k_{}'.format(i) for i in np.arange(dqn.k)]
    for ii in range(dqn.k):
        q_feed_dict[dqn.placeholder[action_k_id[ii]]] = action_ids_k_tr[ii]#(5,)

    loss_val = dqn.train_on_batch(q_feed_dict)
    loss_val = np.round(loss_val,10)
    return loss_val

@cost_time_def
def test_during_training(current_best_reward,test_user,time_horizon,dqn,env):
    # initialize empty states
    sim_vali_user = test_user
    states = [[] for _ in range(len(sim_vali_user))]
    sim_u_reward = {}

    for t in range(time_horizon):
        max_q_feed_dict,states_tr,history_order_tr,history_user_tr = format_max_q_feed_dict(sim_vali_user,states,dqn,env)

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

        if len(sim_vali_user) ==0:
            break

    _,_,_,_,new_best_reward = env.compute_average_reward(sim_vali_user,sim_u_reward,current_best_reward)
    return new_best_reward

@cost_time_def
def train_with_greedy_action(args,env,dqn):
    data_size = args.time_horizon * args.sample_batch_size + 100
    current_best_reward = 0.0
    for itr in range(args.epochs):
        training_start_point = (itr*args.sample_batch_size)%25000
        training_user = env.train_user[training_start_point:training_start_point+ args.sample_batch_size]

        # for i in range(0,len(env.train_user),args.sample_batch_size):


        # initialize empty states
        states = [[] for _ in range(len(training_user))]
        data_collection = {'user': deque(maxlen=data_size), 'state': deque(maxlen=data_size),
                           'action': deque(maxlen=data_size), 'y': deque(maxlen=data_size)}
        data_collection = data_collection_greedy_action(data_collection,args, env, dqn, states,training_user)

        # START TRAINING for this batch of users
        num_samples = len(data_collection['user'])
        batch_iterations = int(np.ceil(num_samples / args.training_batch_size))
        for n in range(batch_iterations):
            batch_sample = np.random.choice(len(data_collection['user']),args.training_batch_size,replace=False)
            states_batch = [data_collection['state'][c] for c in batch_sample]
            user_batch = [data_collection['user'][c] for c in batch_sample]
            action_batch = [data_collection['action'][c] for c in batch_sample]
            y_batch = [data_collection['y'][c] for c in batch_sample]

            loss_val = train_on_batch_placeholder(user_batch,states_batch,action_batch,y_batch,dqn,env)

            if np.mod(n,250) == 0:
                loss_val = list(loss_val[-dqn.k:])
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print_loss = ' '
                for kkk in range(dqn.k):
                    print_loss += ' %.5g,'
                print(('%s: greedy itr(%d), training loss:'+print_loss) % tuple([log_time, n]+loss_val))

        print('finish iteration %d' % itr)

        # TEST
        new_reward = test_during_training(current_best_reward,env.test_user[0:10],args.time_horizon,dqn,env)
        if new_reward > current_best_reward:
            save_path = os.path.join(args.model_path, 'best-reward')
            dqn.save('best-reward')
            current_best_reward = new_reward

@cost_time_def
def repeated_test(_repeated_best_reward,args,dqn,env):
    sim_user_reward = [{} for _ in range(args.num_test)]
    user_avg_reward = [[] for _ in range(args.num_test)]
    click_rate = [[] for _ in range(args.num_test)]
    mean_user_avg_reward = np.zeros(args.num_test)
    mean_click_rate = np.zeros(args.num_test)

    for i_th in range(args.num_test):
        sim_vali_user = env.test_user
        states = [[] for _ in range(len(sim_vali_user))]

        for t in range(args.time_horizon):
            max_q_feed_dict,states_tr,history_order_tr,history_user_tr = format_max_q_feed_dict(sim_vali_user,states,dqn,env)

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
            sim_vali_user, states, sim_user_reward[i_th] = env.sample_new_states(sim_vali_user, states, transition_p,
                                                                                 reward_u, sim_user_reward[i_th],  max_action, dqn.k)
            if len(sim_vali_user) == 0:
                break
        user_avg_reward[i_th], mean_user_avg_reward[i_th], \
        click_rate[i_th], mean_click_rate[i_th], _ = env.compute_average_reward(env.test_user, sim_user_reward[i_th], 100.0)

    if np.mean(mean_user_avg_reward) > _repeated_best_reward:
        print('new repeated best reward!!!!!!!!!!')
        _repeated_best_reward = np.mean(mean_user_avg_reward)
        dqn.save('repeated_best-reward')
        filename = 'RL_recommend'+str(args.k)+'_'+str(args.noclick_weight)+'.pickle'
        env.save_results(args.time_horizion, env.test_user, sim_user_reward, user_avg_reward, mean_user_avg_reward, click_rate, mean_click_rate, filename)
    else:
        print(['mean, reward of all experiments:', np.mean(mean_user_avg_reward)])
        print(['std, reward of all experiments:', np.std(mean_user_avg_reward)])
        print(['mean, click rate of all experiments:', np.mean(mean_click_rate)])
        print(['std, click rate of all experiments:', np.std(mean_click_rate)])

    return _repeated_best_reward

@cost_time_def
def main(args):
    env = Enviroment(args)
    env.initialize_environment()
    dqn = DQN(env,args)

    # 参照强化学习的训练逻辑，EE问题。在收集数据的时候兼顾EE问题。此论文的思路将EE问题分开来解决。
    #   首先用随机策略收集数据，其次，在随机策略的训练基础上再使用贪婪策略来训练策略。
    # 首先，根据随机策略来收集并训练
    # train_with_random_action(args,env,dqn)

    # 使用贪婪策略收集的数据来训练我们的推荐引擎
    train_with_greedy_action(args,env,dqn)

    # dqn.restore('init-q')
    # dqn.restore('best-reward')
    # _ = repeated_test(0.0, args,dqn,env)



if __name__ == '__main__':
    cmd_args = get_options()
    print('>>>>>>>>>>>',cmd_args,'<<<<<<<<<<<<<<<<<<<<<<')
    main(cmd_args)
    print('finished!!!!!!')