#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/25 15:45                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
from utils.yjp_decorator import cost_time_def
from GAN_RL.yjp.code.options import get_options
import datetime,os
import numpy as np
from collections import deque
from itertools import chain
import threading
from multiprocessing import Process

class Dataset():
    def __init__(self,args,env,dqn):
        self.args = args
        self.env = env
        self.dqn = dqn
        self.lock = threading.Lock()

    @cost_time_def
    def data_collection(self, training_user,type='random'):
        states = [[] for _ in range(len(training_user))]
        data_size_init = len(training_user)*self.args.train_time_horizon+100
        data_collection = {'user':deque(maxlen=data_size_init),'state':deque(maxlen=data_size_init),
                           'action':deque(maxlen=data_size_init),'y':deque(maxlen=data_size_init)}
        if type=='random':
            # (1) first step: initialize Q as the expected rwd function
            # current_best_reward = 0.0
            for t in range(self.args.train_time_horizon):
                data_collection['state'].extend(states)
                data_collection['user'].extend(training_user)
                # prepare to feed max_Q
                states_tr,history_order_tr,history_user_tr = self.form_init_Q_feed_dict(training_user,states)

                # 1. sample random action，模仿推荐系统推荐的action
                # feature_space 保存的是和用户相关的sku的特征:其实就是sku的embedding向量
                random_action = [[] for _ in range(len(training_user))]
                random_action_feature = []
                # 每个用户曝光10个item
                for u_i in range(len(training_user)):
                    user_i = training_user[u_i]
                    # 从用户曝光过的sku中随机选出k个sku作为推荐，模拟推荐引擎的选择
                    random_action[u_i] = np.random.choice(list(set(np.arange(len(self.env.feature_space[user_i])))-set(states[u_i])),self.dqn.k,replace=False).tolist()
                    random_action_feature += [self.env.feature_space[user_i][jj] for jj in random_action[u_i]]

                # 为了过滤第一次就不满足个数的情况。比如当用户相关的sku个数不足10个时，np.random.choice会报错，所以过滤这部分数据，其实，刚开始的时候，用户相关的
                #   sku个数至少应该满足10个要求的。过滤之后，只剩90个user,如果没有下面的代码，则disp_2d_split_user还是100个，因此会报错。
                #   这种方式不行，应该在最开始的时候满足用户相关的sku个数大于10的阈值。states_tr中的training_user的数量没有减少

                best_action = [random_action,random_action_feature]
                data_collection['action'].extend(best_action[0])

                #2. compute expected immediate reward
                disp_2d_split_user = np.kron(np.arange(len(training_user)),np.ones(self.dqn.k))
                # reward_feed_dict = {Xs_clicked: [], history_order_indices: [], history_user_indices: [], disp_2d_split_user_ind: [], disp_action_feature:[]}
                reward_feed_dict={}
                reward_feed_dict[self.env.placeholder['Xs_clicked']]=states_tr
                reward_feed_dict[self.env.placeholder['history_order_indices']]=history_order_tr
                reward_feed_dict[self.env.placeholder['history_user_indices']]=history_user_tr
                reward_feed_dict[self.env.placeholder['disp_2d_split_user_ind']]=disp_2d_split_user
                reward_feed_dict[self.env.placeholder['disp_action_feature']]=best_action[1]

                # 为exp操作前的reward*对应的权重
                # Reward_r = tf.segment_sum(tf.multiply(u_disp, p_disp), disp_2d_split_user_ind)
                # 计算的是用户的总reward
                best_action_reward,transition_p,u_disp,_ = self.env.conpute_reward(reward_feed_dict)

                # 4. save to memory
                y_value = best_action_reward
                data_collection['y'].extend(y_value.tolist())##y存的是用户推荐引擎推荐的10个item，也即对用户曝光的10个item总的加权reward

                # 5. sample new states
                remove_set = []
                for j in range(len(training_user)):
                    if len(self.env.feature_space[training_user[j]])-len(states[j]) <= self.dqn.k+1:
                        remove_set.append(j)

                    disp_item = best_action[0][j]
                    #transition_p[j, :]得到的是每个用户对k个item的权重
                    # 如果np.sum(transition_p[j,:]为1，则说明当前用户一定是选了一个，注意力被平均的分配到了10个item身上。
                    # 如果和不为1，则说明当前用户对此时的10个sku并不感兴趣
                    no_click = [max(1.0-(1.0 if np.sum(transition_p[j,:])>self.args.threshold else np.sum(transition_p[j,:])),0.0)]
                    prob = np.array(transition_p[j,:].tolist()+no_click)
                    prob = prob/float(prob.sum())
                    # 模拟用户的选择
                    rand_choice = np.random.choice(disp_item+[-100],1,p = prob)
                    if rand_choice != -100:
                        states[j] += rand_choice.tolist()

                previous_size = len(training_user)
                states = [states[j] for j in range(previous_size) if j not in remove_set]
                training_user = [training_user[j] for j in range(previous_size) if j not in remove_set]
        else:#greedy
            for t in range(self.args.train_time_horizon):
                data_collection['state'].extend(states)
                data_collection['user'].extend(training_user)
                # prepare to feed max_Q
                max_q_feed_dict,states_tr,history_order_tr,history_user_tr = self.format_max_q_feed_dict(training_user,states)


                # 1. find best recommend action
                max_action,max_action_disp_feature = self.dqn.choose_action(max_q_feed_dict)
                data_collection['action'].extend(max_action.tolist())
                # 2. compute reward
                disp_2d_split_user = np.kron(np.arange(len(training_user)),np.ones(self.dqn.k)).astype(int)
                reward_feed_dict={}
                reward_feed_dict[self.env.placeholder['Xs_clicked']]=states_tr
                reward_feed_dict[self.env.placeholder['history_order_indices']]=history_order_tr
                reward_feed_dict[self.env.placeholder['history_user_indices']]=history_user_tr
                reward_feed_dict[self.env.placeholder['disp_2d_split_user_ind']]=disp_2d_split_user
                reward_feed_dict[self.env.placeholder['disp_action_feature']]=max_action_disp_feature
                _, transition_p,u_disp,_ = self.env.conpute_reward(reward_feed_dict)
                reward_u = np.reshape(u_disp,[-1,self.dqn.k])
                # 3. sample new states
                states, training_user, old_training_user, next_states, sampled_reward, remove_set = \
                    self.env.sample_new_states_for_train(training_user, states, transition_p, reward_u, max_action, self.dqn.k)

                # 4. compute one-step delayed reward
                max_q_feed_dict,_,_,_ = self.format_max_q_feed_dict(old_training_user,next_states)
                max_q_value = self.dqn.get_max_q_value(max_q_feed_dict)

                #4. save to memory
                y_value = sampled_reward+self.args.gamma*max_q_value
                data_collection['y'].extend(y_value.tolist())


        ind = np.random.permutation(len(data_collection['user']))
        user = np.array(data_collection['user'])[ind].tolist()
        data_collection['user'].clear()
        data_collection['user'].extend(user)
        user = np.array(data_collection['state'])[ind].tolist()
        data_collection['state'].clear()
        data_collection['state'].extend(user)
        user = np.array(data_collection['action'])[ind].tolist()
        data_collection['action'].clear()
        data_collection['action'].extend(user)
        user = np.array(data_collection['y'])[ind].tolist()
        data_collection['y'].clear()
        data_collection['y'].extend(user)

        return data_collection


    def collection(self,u_set,type='random'):
        global data_collection
        data = self.data_collection(u_set,type)
        self.lock.acquire()
        data_collection['user'].extend(data['user'])
        data_collection['state'].extend(data['state'])
        data_collection['action'].extend(data['action'])
        data_collection['y'].extend(data['y'])
        self.lock.release()

    #用多线程收集数据反而变慢了
    @cost_time_def
    def multi_collect_data(self,user_set,num_sets):
        global data_collection

        thread_u = [[] for _ in range(num_sets)]
        for ii in range(len(user_set)):
            thread_u[ii%num_sets].append(user_set[ii])


        data_size_init = len(user_set)*self.args.train_time_horizon+100
        data_collection = {'user':deque(maxlen=data_size_init),'state':deque(maxlen=data_size_init),
                           'action':deque(maxlen=data_size_init),'y':deque(maxlen=data_size_init)}

        threads = []
        for ii in range(num_sets):
            print('======',len(thread_u[ii]))
            if self.args.compu_type=='thread':
                thread = threading.Thread(target=self.collection,args=(thread_u[ii],))
            else:
                thread = Process(target=self.collection,args=(thread_u[ii],))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return data_collection


    def format_max_q_feed_dict(self,user_set,states):
        action_mean_tr, action_std_tr, action_user_indice_tr, action_tensor_indice_tr, action_shape_tr, \
        action_space_tr, states_tr, history_order_tr, history_user_tr, action_cnt_tr, \
        action_space_cnt_tr, action_id_tr = self.form_max_q_feed_dict(user_set, states)

        max_q_feed_dict = {}
        max_q_feed_dict[self.dqn.placeholder['all_action_id']]=action_id_tr
        max_q_feed_dict[self.dqn.placeholder['all_action_user_indices']]=action_user_indice_tr
        max_q_feed_dict[self.dqn.placeholder['all_action_tensor_indices']]=action_tensor_indice_tr
        max_q_feed_dict[self.dqn.placeholder['all_action_tensor_shape']]=action_shape_tr
        max_q_feed_dict[self.dqn.placeholder['current_action_space']]=action_space_tr
        max_q_feed_dict[self.dqn.placeholder['action_space_mean']]=action_mean_tr
        max_q_feed_dict[self.dqn.placeholder['action_space_std']]=action_std_tr
        max_q_feed_dict[self.env.placeholder['Xs_clicked']]=states_tr
        max_q_feed_dict[self.env.placeholder['history_order_indices']]=history_order_tr
        max_q_feed_dict[self.env.placeholder['history_user_indices']]=history_user_tr
        max_q_feed_dict[self.dqn.placeholder['action_count']]=action_cnt_tr
        max_q_feed_dict[self.dqn.placeholder['action_space_count']]=action_space_cnt_tr

        return max_q_feed_dict,states_tr,history_order_tr,history_user_tr

    def prepare_validation_data(self,user_set,num_sets):
        thread_u = [[] for _ in range(num_sets)]
        for ii in range(len(user_set)):
            thread_u[ii%num_sets].append(user_set[ii])


    def form_loss_feed_dict(self,user_set,states_id,action_id):
        states_feature = [[] for _ in range(len(user_set))]
        history_order = [[] for _ in range(len(user_set))]  # np.zeros([len(user_set)], dtype=np.int64)
        history_user = [[] for _ in range(len(user_set))]  # np.arange(len(user_set), dtype=np.int64)

        action_space = []

        candidate_action_mean = [[] for _ in range(len(user_set))]
        candidate_action_std = [[] for _ in range(len(user_set))]

        action_ids_k = [[] for _ in range(self.dqn.k)]

        for uu,user in enumerate(user_set):
            candidate_action = []

            if len(states_id[uu]) == 0:
                states_feature[uu] = np.zeros([1, self.dqn.f_dim], dtype=np.float32).tolist()
                history_order[uu].append(0)
                history_user[uu].append(uu)
            else:
                states_feature[uu] = deque(maxlen=self.dqn.band_size)
                for idd in states_id[uu]:
                    states_feature[uu].append(self.env.feature_space[user][idd])

                states_feature[uu] = list(states_feature[uu])
                id_cnt = len(states_feature[uu])
                history_order[uu] = np.arange(id_cnt, dtype=np.int64).tolist()
                history_user[uu] = list(uu * np.ones(id_cnt, dtype=np.int64))

            action_candidate = np.array(list(set(np.arange(len(self.env.feature_space[user]))) - set(states_id[uu])))
            for idd in action_candidate:
                candidate_action.append(self.env.feature_space[user][idd])

            candidate_action_mean[uu] = np.mean(np.array(candidate_action), axis=0).tolist()
            candidate_action_std[uu] = np.std(np.array(candidate_action), axis=0).tolist()

            # all actions
            # action_indicate += map(lambda x: x + len(action_space), action_id[uu])
            for jj in range(self.dqn.k):
                action_ids_k[jj].append(action_id[uu][jj]+len(action_space))# 加了len(action_space)变成了绝对顺序了


            # action space
            action_space += self.env.feature_space[user]

        states_feature = list(chain.from_iterable(states_feature))
        history_order = list(chain.from_iterable(history_order))
        history_user = list(chain.from_iterable(history_user))

        return action_ids_k,action_space,states_feature,history_order,history_user,candidate_action_mean,candidate_action_std

    def form_max_q_feed_dict(self,user_set,states_id):
        # states_feature = np.zeros([len(user_set), _f_dim], dtype=np.float32)

        states_feature = [[] for _ in range(len(user_set))]
        history_order = [[] for _ in range(len(user_set))]  # np.zeros([len(user_set)], dtype=np.int64)
        history_user = [[] for _ in range(len(user_set))]  # np.arange(len(user_set), dtype=np.int64)

        action_space = []

        # action_indicate_u = [[] for _ in range(len(user_set))]

        # action_indicate = []

        action_id = []

        action_user_indice = []
        action_tensor_indice = []

        max_act_size = 0

        candidate_action_mean = [[] for _ in range(len(user_set))]
        candidate_action_std = [[] for _ in range(len(user_set))]

        action_cnt = [0 for _ in range(len(user_set))]

        action_space_cnt = [0 for _ in range(len(user_set))]

        for uu,user in enumerate(user_set):
            candidate_action = []

            if len(states_id[uu]) == 0:
                states_feature[uu]=np.zeros([1,self.dqn.f_dim],dtype=np.float32).tolist()
                history_order[uu].append(0)
                history_user[uu].append(uu)
            else:
                states_feature[uu] = deque(maxlen=self.dqn.band_size)
                for idd in states_id[uu]:
                    states_feature[uu].append(self.env.feature_space[user][idd])

                states_feature[uu] = list(states_feature[uu])
                id_cnt= len(states_feature[uu])
                history_order[uu] = np.arange(id_cnt,dtype=np.int64).tolist()
                history_user[uu] = list(uu*np.ones(id_cnt,dtype=np.int64))


            action_candidate = np.array(list(set(np.arange(len(self.env.feature_space[user])))-set(states_id[uu])))

            for idd in action_candidate:
                candidate_action.append(self.env.feature_space[user][idd])

            candidate_action_mean[uu]=np.mean(np.array(candidate_action),axis=0)
            candidate_action_std[uu] = np.std(np.array(candidate_action),axis=0)

            action_space_cnt[uu] = len(action_space)

            action_id_u = list(action_candidate+action_space_cnt[uu])
            action_id += action_id_u
            # all possible actions
            # action_indicate_u[uu] = list(chain.from_iterable(combinations(action_candidate + action_space_cnt[uu], _k)))
            # action_indicate += action_indicate_u[uu]

            action_cnt[uu] = len(action_id_u)
            action_user_indice += [uu for _ in range(action_cnt[uu])]

            if action_cnt[uu] == 0:
                print('action_cnt 0')
                print(action_candidate)
                print(action_candidate + action_space_cnt[uu])
                print(states_id[uu])

            max_act_size = max(max_act_size,action_cnt[uu])
            # action_user_indice += [uu for _ in range(action_cnt[uu])]
            action_tensor_indice += map(lambda x:[uu,x],np.arange(action_cnt[uu]))

            # action space
            action_space +=self.env.feature_space[user]

        action_cnt =np.cumsum(action_cnt)
        action_cnt = [0] + list(action_cnt[:-1])

        action_shape = [len(user_set),max_act_size]

        states_feature = list(chain.from_iterable(states_feature))
        history_order = list(chain.from_iterable(history_order))
        history_user = list(chain.from_iterable(history_user))

        return candidate_action_mean,candidate_action_std,action_user_indice,action_tensor_indice,action_shape, \
               action_space,states_feature,history_order,history_user,action_cnt,action_space_cnt,action_id

    def form_init_Q_feed_dict(self,user_set,states_id):
        # states_feature = np.zeros([len(user_set), _f_dim], dtype=np.float32)
        states_feature = [[] for _ in range(len(user_set))]
        history_order = [[] for _ in range(len(user_set))] #np.zeros([len(user_set)], dtype=np.int64)
        history_user = [[] for _ in range(len(user_set))] # np.arange(len(user_set), dtype=np.int64)

        for uu,user in enumerate(user_set):
            if len(states_id[uu]) ==0:
                states_feature[uu] = np.zeros([1,self.dqn.f_dim],dtype=np.float32).tolist()
                history_order[uu].append(0)
                history_user[uu].append(uu)
            else:
                states_feature[uu] =deque(maxlen=self.dqn.band_size)
                for idd in states_id[uu]:# 用户点击的item
                    states_feature[uu].append(self.env.feature_space[user][idd])

                states_feature[uu] = list(states_feature[uu])
                id_cnt = len(states_feature[uu])
                history_order[uu] = np.arange(id_cnt,dtype=np.int64).tolist()
                history_user[uu] = list(uu*np.ones(id_cnt,dtype=np.int64))

        states_feature = list(chain.from_iterable(states_feature))
        history_order = list(chain.from_iterable(history_order))
        history_user = list(chain.from_iterable(history_user))
        return states_feature,history_order,history_user

    def data_prepare_for_loss_placeholder(self,user_batch,states_batch,action_batch,y_batch):
        action_ids_k_tr,action_space_tr,states_feature_tr,history_order_tr, \
        history_user_tr,action_mean_tr,action_std_tr = self.form_loss_feed_dict(user_batch,states_batch,action_batch)
        q_feed_dict={}
        q_feed_dict[self.dqn.placeholder['current_action_space']]=action_space_tr#(120,65)
        q_feed_dict[self.dqn.placeholder['action_space_mean']]=action_mean_tr#(5,65)
        q_feed_dict[self.dqn.placeholder['action_space_std']]=action_std_tr#(5,65)
        q_feed_dict[self.env.placeholder['Xs_clicked']]=states_feature_tr#(21,65)
        q_feed_dict[self.env.placeholder['history_order_indices']]=history_order_tr#(21,)
        q_feed_dict[self.env.placeholder['history_user_indices']]=history_user_tr#(21,)
        q_feed_dict[self.dqn.placeholder['y_label']]=y_batch#(5,)

        action_k_id = ['action_k_{}'.format(i) for i in np.arange(self.dqn.k)]
        for ii in range(self.dqn.k):
            q_feed_dict[self.dqn.placeholder[action_k_id[ii]]] = action_ids_k_tr[ii]#(5,)

        return q_feed_dict

if __name__ == '__main__':
    from GAN_RL.yjp.code.env import Enviroment
    from GAN_RL.yjp.code.dqn import DQN

    cmd_args = get_options()
    env = Enviroment(cmd_args)
    env.initialize_environment()
    dqn = DQN(env,cmd_args)
    dataset = Dataset(cmd_args,env,dqn)

    data = dataset.multi_collect_data(env.train_user,cmd_args.num_thread)
    print(len(data['user']))