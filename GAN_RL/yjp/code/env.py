#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/19 16:38                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
# 使用GAN训练得到的强化学习环境
import pickle,os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from GAN_RL.yjp.code.options import get_options

class Enviroment():
    def __init__(self,args):
        self.data_folder = args.data_folder
        self.random_seed = args.random_seed
        self.model_path = os.path.join(args.save_dir,args.user_model)

        self.k = args.k
        self.noclick_weight = args.noclick_weight
        self.band_size = args.pw_band_size
        self.pw_dim = args.pw_dim
        self.user_model = args.user_model
        self.save_dir = args.save_dir
        self.hidden_dims = args.dims
        self.std = args.env_std
        self.placeholder = {}
        self.sess=tf.compat.v1.InteractiveSession()

        np.random.seed(self.random_seed)

    def format_feature_space(self):
        with open(self.data_folder+'data_behavior.pkl','rb') as f:
            data_behavior = pickle.load(f)

        filename = self.data_folder+'user-split.pkl'
        file = open(filename, 'rb')
        self.train_user = pickle.load(file)
        self.vali_user = pickle.load(file)
        self.test_user = pickle.load(file)
        self.size_user = pickle.load(file)
        self.size_item = pickle.load(file)
        file.close()

        filename =self.data_folder+'embedding.pkl'
        file = open(filename, 'rb')
        self.sku_embedding = pickle.load(file)
        self.user_embedding = pickle.load(file)
        id2key_user = pickle.load(file)
        id2key_sku = pickle.load(file)

        self.f_dim = self.sku_embedding.shape[1]
        random_emb = np.random.randn(self.f_dim).tolist()

        self.sku_emb_dict = {id2key_sku.get(ind,'UNK'):emb.tolist() for ind,emb in enumerate(self.sku_embedding)}
        self.user_emb_dict = {id2key_user.get(ind,'UNK'):emb.tolist() for ind,emb in enumerate(self.user_embedding)}
        file.close()

        self.feature_space=defaultdict(list)
        # [self.feature_space[line[0]].append(self.sku_emb_dict.get(sku,random_emb)) for line in data_behavior for sku in line[1]]
        for ind in range(self.size_user):
            u = data_behavior[ind][0]

            for event in range(len(data_behavior[ind][1])):
                disp_id = data_behavior[ind][1][event]
                for id in disp_id:
                    emb = self.sku_emb_dict.get(id,random_emb)
                    self.feature_space[u].append(emb)


    def construct_placeholder(self):

        self.placeholder['disp_action_feature'] = tf.placeholder(dtype=tf.float32, shape=[None, self.f_dim])
        self.placeholder['Xs_clicked'] = tf.placeholder(dtype=tf.float32, shape=[None, self.f_dim])

        self.placeholder['news_size'] = tf.placeholder(dtype=tf.int64, shape=[])
        self.placeholder['user_size'] = tf.placeholder(dtype=tf.int64, shape=[])
        self.placeholder['disp_indices'] = tf.placeholder(dtype=tf.int64, shape=[None, 2])

        self.placeholder['disp_2d_split_user_ind'] = tf.placeholder(dtype=tf.int64, shape=[None])

        self.placeholder['history_order_indices'] = tf.placeholder(dtype=tf.int64, shape=[None])
        self.placeholder['history_user_indices'] = tf.placeholder(dtype=tf.int64, shape=[None])

    def mlp(self,x,hidden_dims,output_dim,activation,sd,act_last=False):
        hidden_dims = tuple(map(int,hidden_dims.split('-')))
        for h in hidden_dims:
            x = tf.layers.dense(x,h,activation=activation,trainable=True,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=sd))

        if act_last:
            return tf.layers.dense(x,output_dim,activation=activation,trainable=True,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=sd))
        else:
            return tf.layers.dense(x, output_dim, trainable=True,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=sd))

    def _init_graph(self):

        # (1) history feature --- net ---> clicked_feature
        # (1) construct cumulative history
        click_history = [[] for _ in range(self.pw_dim)]
        position_weight = [[] for _ in range(self.pw_dim)]
        for ii in range(self.pw_dim):
            position_weight[ii] = tf.get_variable('p_w'+str(ii),[self.band_size],initializer=tf.constant_initializer(0.0001))
            # np.arange(id_cnt) 当前用户上一时刻的点击的item的数量
            position_weight_values = tf.gather(position_weight[ii],self.placeholder['history_order_indices'])
            weighted_feature = tf.multiply(self.placeholder['Xs_clicked'],tf.reshape(position_weight_values,[-1,1]))
            click_history[ii] = tf.segment_sum(weighted_feature,self.placeholder['history_user_indices'])

        self.user_states = tf.concat(click_history,axis=1)

        #disp_2d_split_user = np.kron(np.arange(len(training_user)), np.ones(_k))
        disp_history_feature = tf.gather(self.user_states,self.placeholder['disp_2d_split_user_ind'])

        # (4) combine features
        # disp_action_feature(40*20) 当前批次用户的所有展示的item，即曝光的item
        concat_disp_features = tf.reshape(tf.concat([disp_history_feature,self.placeholder['disp_action_feature']],axis=1),
                                          [-1,self.f_dim*self.pw_dim+self.f_dim])

        # (5) compute utility
        self.u_disp = self.mlp(concat_disp_features,self.hidden_dims,1,tf.nn.elu,sd=self.std,act_last=False)
        # (5)
        self.u_disp = tf.reshape(self.u_disp, [-1])
        exp_u_disp = tf.exp(self.u_disp)
        #当_noclick_weight的结果不足以影响每个用户的sum时，此时，sum会为1.即noclick_weight和env计算出来的reward是同量级时。和就不会为1
        sum_exp_disp = tf.segment_sum(exp_u_disp,self.placeholder['disp_2d_split_user_ind'])+float(np.exp(self.noclick_weight))
        scatter_sum_exp_disp = tf.gather(sum_exp_disp,self.placeholder['disp_2d_split_user_ind'])
        self.p_disp = tf.div(exp_u_disp,scatter_sum_exp_disp)

        self.exp_u_disp = exp_u_disp
        self.scatter_sum_exp_disp = scatter_sum_exp_disp
        self.disp_2d_split_user_ind = self.placeholder['disp_2d_split_user_ind']



    def initialize_environment(self,reuse=False):
        self.format_feature_space()
        self.construct_placeholder()
        print(['_k', self.k, '_noclick_weight', self.noclick_weight])

        with tf.variable_scope('model',reuse=reuse):
            self._init_graph()


        self.agg_variables = tf.compat.v1.trainable_variables()
        self.saver = tf.compat.v1.train.Saver(var_list=self.agg_variables,max_to_keep=None)
        # self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.agg_variables))
        self.restore('best-loss')

    def restore(self,model_name):
        best_save_path = os.path.join(self.model_path, model_name)
        self.saver.restore(self.sess, best_save_path)
        print('model:{} loaded success!!!!'.format(model_name))


    def conpute_reward(self,reward_feed_dict):
        # 1. reward
        Reward_r = tf.segment_sum(tf.multiply(self.u_disp, self.p_disp), self.placeholder['disp_2d_split_user_ind'])
        Reward_1 = tf.segment_sum(self.p_disp, self.placeholder['disp_2d_split_user_ind'])
        trans_p = tf.reshape(self.p_disp, [-1, self.k])

        Reward_r,trans_p,u_disp = self.sess.run([Reward_r,trans_p,self.u_disp],feed_dict=reward_feed_dict)
        # u_disp,p_disp,exp_u_disp,u1,u2 = self.sess.run([self.u_disp,self.p_disp,self.exp_u_disp,
        #                                           self.scatter_sum_exp_disp,self.disp_2d_split_user_ind],feed_dict=reward_feed_dict)
        # # 错误的代码中reward_r的shape为(10, 100)，因为tf.exp(u_disp)用的是未变形的变量
        # print('=================',Reward_r.shape,u_disp.shape,p_disp.shape,exp_u_disp.shape,u1.shape,u2.shape)


        reward_feed_dict = {self.placeholder['Xs_clicked']: [],
                            self.placeholder['history_order_indices']: [],
                            self.placeholder['history_user_indices']: [],
                            self.placeholder['disp_2d_split_user_ind']: [],
                            self.placeholder['disp_action_feature']:[]}



        return Reward_r,trans_p,u_disp,reward_feed_dict

    def sample_new_states(self,sim_vali_user,states,trasition_p,reward_u,sim_user_reward,best_action_id,_k):
        remove_set=[]
        for j in range(len(sim_vali_user)):
            if len(self.feature_space[sim_vali_user[j]])-len(states[j])<=self.k+1:
                remove_set.append(j)

            disp_item = best_action_id[j].tolist()
            no_click = [max(1.0-np.sum(trasition_p[j,:]),0.0)]
            prob = np.array(trasition_p[j,:].tolist()+no_click)
            #transition_p[j, :]得到的是每个用户对k个item的权重
            # 如果np.sum(transition_p[j,:]为1，则说明当前用户一定是选了一个，注意力被平均的分配到了10个item身上。
            # 如果和不为1，则说明当前用户对此时的10个sku并不感兴趣
            prob = prob/float(prob.sum())
            rand_choice = np.random.choice(disp_item+[-100],1,p = prob)

            if sim_vali_user[j] not in sim_user_reward:
                sim_user_reward[sim_vali_user[j]] =[]

            if rand_choice[0] != -100:
                states[j] += rand_choice.tolist()
                idx = disp_item.index(rand_choice[0])
                sim_user_reward[sim_vali_user[j]].append(reward_u[j][idx])
            else:
                sim_user_reward[sim_vali_user[j]].append(0)

        previous_size = len(sim_vali_user)
        states = [states[j] for j in range(previous_size) if j not in remove_set]
        sim_vali_user = [sim_vali_user[j] for j in range(previous_size) if j not in remove_set]
        return sim_vali_user,states,sim_user_reward

    def compute_average_reward(self,sim_vali_user,sim_user_rewrd,current_best_reward):
        user_avg_reward = []
        clk_rate = []
        for j in range(len(sim_vali_user)):
            user_j_reward = sim_user_rewrd[sim_vali_user[j]]
            num_clk = np.sum(np.array(user_j_reward)==0)
            clk_rate.append(1.0-float(num_clk)/len(user_j_reward))

            cusum_reward = np.cumsum(user_j_reward)
            # user_cusum_reward.append(cusum_reward[-1])
            avg_cumsum_reward = cusum_reward/np.arange(1,len(cusum_reward)+1)
            user_avg_reward.append(avg_cumsum_reward[-1])

        current_sum_reward = np.sum(user_avg_reward)
        current_sum_clkrate = np.sum(clk_rate)

        # best_or_not =''
        # if current_avg_reward>current_best_reward:
        #     current_best_reward = current_avg_reward
        #     best_or_not = 'new best!!!'
        # print(['mean avg reward', current_avg_reward, 'clk_rate:', current_avg_clkrate,  best_or_not])

        return current_sum_reward,current_sum_clkrate


    def sample_new_states_for_train(self,training_user, states, transition_p, reward_u,  best_action_id, _k):
        remove_set = []
        sampled_reward = []
        for j in range(len(training_user)):
            # 如果某个用户可选的action数量<某个阈值，则该用户不用再处理了
            if len(self.feature_space[training_user[j]]) - len(states[j]) <= _k+1:
                remove_set.append(j)

            disp_item = best_action_id[j].tolist()
            no_click = [max(1.0 - np.sum(transition_p[j, :]), 0.0)]
            p_ = np.sum(transition_p[j,:])
            if p_ == 1:
                print('GGGGGGGGGGGG!!!!!!',p_)
            prob = np.array(transition_p[j, :].tolist()+no_click)
            # print('===========',prob,no_click,transition_p[j, :].tolist())#[0.42553002 0.57446998] [0.5744699835777283] [0.42553001642227173]
            prob = prob / float(prob.sum())
            rand_choice = np.random.choice(disp_item + [-100], 1, p=prob)

            if rand_choice[0] != -100:
                states[j] += rand_choice.tolist()
                idx = disp_item.index(rand_choice[0])
                sampled_reward.append(reward_u[j][idx])
            else:# 如果什么也没选，则reward置零
                sampled_reward.append(0)

        previous_size = len(training_user)
        states_removed = [states[j] for j in range(previous_size) if j not in remove_set]
        training_user_removed = [training_user[j] for j in range(previous_size) if j not in remove_set]

        return states_removed, training_user_removed, training_user, states, np.array(sampled_reward), remove_set


    def save_results(self,time_horizon, sim_vali_user, sim_user_reward, user_avg_reward, mean_user_avg_reward, clk_rate, mean_clk_rate, filename):

        print(['mean, reward of all experiments:', np.mean(mean_user_avg_reward)])
        print(['std, reward of all experiments:', np.std(mean_user_avg_reward)])
        print(['mean, click rate of all experiments:', np.mean(mean_clk_rate)])
        print(['std, click rate of all experiments:', np.std(mean_clk_rate)])

        with open(filename, 'wb') as handle:
            pickle.dump(sim_vali_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(sim_user_reward, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(user_avg_reward, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(mean_user_avg_reward, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(clk_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(mean_clk_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(time_horizon, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    cmd_args = get_options()
    print('current args:{}'.format(cmd_args))
    env = Enviroment(cmd_args)
    env.initialize_environment()
    print(len(env.train_user))
    np.save('vali_user',env.vali_user)
