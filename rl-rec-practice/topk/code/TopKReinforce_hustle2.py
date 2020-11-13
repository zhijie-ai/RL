#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/6/15 下午6:44                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import math
import pickle
from tensorflow.contrib import rnn
import os

'''
Top-K Off-Policy Correction for a REINFORCE Recommender System论文的实现思路,该论文实现思路和原论文的有点不一样，
原论文在实现off-policy correction时，action应该是从beta策略中抽样出来的，然后是计算action在PI和beta策略的概率取比值。
而本论文的实现思路是，action均从各自的分布中采样，然后取比值。代码作者认为该实现和原论文差不多，甚至更好。参考 https://github.com/awarebayes/RecNN/issues/7
'''
# 与TopKReinforce_hustle.py不同的地方在于pi loss的计算，交叉熵的计算,前者用的是pi_log_prob,而本代码使用和TopKReinforce.py的实现。
# TopKReinforce_hustle.py中pi loss一直波动，不下降的原因应该是loss的计算，交叉熵的计算虽然是最大化action对应的概率。可是它的实现似乎都没有用到y值。
# 无效的原因是不是在随机选择action，然后将该action的概率最大化？
# 主网络和beta网络的实现
# topk修正后的概率
# beta的loss降不下去
def cascade_model(p,k):
    return 1-(1-p)**k

# lambda比重
def gradient_cascade(p, k):
    return k*(1-p)**(k-1)

def load_data(path='../data/session.pickle',time_step=15,gamma=0.95):
    historys=[]
    actions=[]
    rewards=[]

    def _discount_and_norm_rewards(rewards):
        discounted_episode_rewards = np.zeros_like(rewards,dtype='float64')
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[t]
            discounted_episode_rewards[t] = cumulative
        # Normalize the rewards
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    with open(path,'rb') as f:
        trajectory,rewards_= pickle.load(f)
        print('FFFFFFFFFF',len(set([i for n in trajectory for i in n])))
        for t,r in zip(trajectory,rewards_):
            r = _discount_and_norm_rewards(r)
            for i in range(len(t)-time_step):
                historys.append(list(t[i:i+time_step]))
                actions.append(t[i+time_step])
                rewards.append(r[i+time_step])


    return np.array(historys),np.array(actions),np.array(rewards)


def load_data_movie_length(path='../data/ratings.dat',time_step=15,gamma=.9):
    historys=[]
    actions=[]
    rewards=[]

    def _discount_and_norm_rewards(rewards):
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
    print(ratings.head())

    items = list(sorted(ratings.itemid.unique()))
    key_to_id_item = dict(zip(items,range(len(items))))
    id_to_key_item = dict(zip(range(len(items)),items))
    users = list(set(sorted(ratings.userid.unique())))
    key_to_id_user = dict(zip(users,range(len(users))))
    id_to_key_user = dict(zip(range(len(users)),users))

    ratings.userid = ratings.userid.map(key_to_id_user)
    ratings.itemid = ratings.itemid.map(key_to_id_item)
    ratings = ratings.sort_values(by=['timestamp']).drop('timestamp',axis=1).groupby('userid')
    for _,df in ratings:
        r = _discount_and_norm_rewards(df.rating.values)
        items = df.itemid.values
        for i in range(len(items)-time_step):
            historys.append(list(items[i:i+time_step]))
            actions.append(items[i+time_step])
            rewards.append(r[i+time_step])


    return np.array(historys),np.array(actions),np.array(rewards)

class TopKReinforce():
    def __init__(self,sess,item_count,embedding_size=64,is_train=True,topK=1,
                 weight_capping_c=math.e**3,batch_size=128,epochs = 1000,gamma=0.95,model_name='reinforce',time_step=15):
        self.sess = sess
        self.item_count=item_count
        self.embedding_size=embedding_size
        self.rnn_size = 128
        self.log_out = 'out/logs_hustle'
        self.topK = topK
        self.weight_capping_c = weight_capping_c# 方差减少技术中的一种 weight capping中的常数c
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.model_name=model_name
        self.checkout = 'checkout/model_hustle'
        self.kl_targ = 0.02
        self.time_step = time_step

        self.historys,self.actions,self.rewards = load_data_movie_length(time_step=time_step,gamma=gamma)
        self.num_batches = len(self.rewards) // self.batch_size
        self.action_source = {"pi": "pi", "beta": "beta"}

        self._init_graph()

        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.log_writer = tf.summary.FileWriter(self.log_out, self.sess.graph)


        if not is_train:
            self.restore_model()

    def __str__(self):
        dit = self.__dict__
        show = ['item_count','embedding_size','is_train','topK','weight_capping_c','batch_size','epochs','gamma','model_name','time_step']
        dict = {key:val for key,val in dit.items() if key in show}
        return str(dict)


    def weight_capping(self,cof):
        return tf.minimum(cof,self.weight_capping_c)

    # 注意，本论文的off-policy的实现和真正的off-policy有点不太一样。真正的off-policy的beta策略是需要和环境
    # 交互收集数据的。需要探索的过程。而此论文的off-policy中的beta策略不需要去收集收集。只需要提供计算概率的功能就ok了
    def choose_action(self, history):
        # Reshape observation to (num_features, 1)
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.PI, feed_dict = {self.input: history})
        action = list(map(lambda x:np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel()),prob_weights))

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        # action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())

        # # exploration to allow rare data appeared
        # if random.randint(0,1000) < 1000:
        #     pass
        # else:
        #     action = random.randint(0,self.n_y-1)
        return action

    # 推理
    def predict(self, history):
        # Reshape observation to (num_features, 1)
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.alpha, feed_dict = {self.input: history})
        # action = tf.arg_max(prob_weights[0])
        actions = tf.nn.top_k(prob_weights[0],self.topK)
        return actions

    def save_model(self,step):
        if not os.path.exists(self.checkout):
            os.makedirs(self.checkout)

        self.saver.save(self.sess,os.path.join(self.checkout,self.model_name),global_step=step,write_meta_graph=False)

    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.checkout)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print('HHHHHHHHHHHHHH',ckpt_name)
            self.saver.restore(self.sess,os.path.join(self.checkout,ckpt_name))


    def pi_beta_sample(self):
        # 1. obtain probabilities
        # note: detach is to block gradient
        beta_probs =self.beta
        pi_probs = self.PI

        # 2. probabilities -> categorical distribution.
        beta_categorical = tf.distributions.Categorical(beta_probs)
        pi_categorical = tf.distributions.Categorical(pi_probs)

        # 3. sample the actions
        # See this issue: https://github.com/awarebayes/RecNN/issues/7
        # usually it works like:
        # pi_action = pi_categorical.sample(); beta_action = beta_categorical.sample();
        # but changing the action_source to {pi: beta, beta: beta} can be configured to be:
        # pi_action = beta_categorical.sample(); beta_action = beta_categorical.sample();
        available_actions = {
            "pi": pi_categorical.sample(),
            "beta": beta_categorical.sample(),
        }
        pi_action = available_actions[self.action_source["pi"]]
        beta_action = available_actions[self.action_source["beta"]]

        # 4. calculate stuff we need
        pi_log_prob = pi_categorical.log_prob(pi_action)
        beta_log_prob = beta_categorical.log_prob(beta_action)

        return pi_log_prob, beta_log_prob, pi_probs

    def _init_graph(self):
        with tf.variable_scope('input'):
            self.input = tf.placeholder(shape=[None,self.time_step],name='X',dtype=tf.int32)
            self.label = tf.placeholder(shape=[None],name='label',dtype=tf.int32)
            self.discounted_episode_rewards_norm = tf.placeholder(shape=[None],name='discounted_rewards',dtype=tf.float32)

        cell = rnn.BasicLSTMCell(self.rnn_size)
        with tf.variable_scope('emb'):
            embedding = tf.get_variable('item_emb',[self.item_count,self.embedding_size])
            inputs = tf.nn.embedding_lookup(embedding,self.input)

        outputs,_ = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32)# outputs为最后一层每一时刻的输出

        # state = tf.reshape(outputs,[-1,self.rnn_size])#bs*step,rnn_size,state
        state = tf.nn.relu(outputs[:,-1,:])

        with tf.variable_scope('main_policy'):
            weights=tf.get_variable('item_emb_pi',[self.item_count,self.rnn_size])
            bias = tf.get_variable('bias',[self.item_count])
            self.PI =tf.add(tf.matmul(state,tf.transpose(weights)),bias)
            self.PI =  tf.nn.softmax(self.PI)# PI策略
            self.alpha = cascade_model(self.PI,self.topK)

        with tf.variable_scope('beta_policy'):
            weights_beta=tf.get_variable('item_emb_beta',[self.item_count,self.rnn_size])
            bias_beta = tf.get_variable('bias_beta',[self.item_count])
            self.beta =tf.add(tf.matmul(state,tf.transpose(weights_beta)),bias_beta)
            self.beta =  tf.nn.softmax(self.beta)# β策略


        label = tf.reshape(self.label,[-1,1])
        with tf.variable_scope('loss'):
            pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample()
            ce_loss_main =tf.nn.sampled_softmax_loss(
                weights,bias,label,state,5,num_classes=self.item_count)

            topk_correction =gradient_cascade(tf.exp(pi_log_prob),self.topK)# lambda 比值
            off_policy_correction = tf.exp(pi_log_prob)/tf.exp(beta_log_prob)
            off_policy_correction = self.weight_capping(off_policy_correction)
            # print('CCCCCCCC',self.PI.shape,self.beta.shape)
            # print('DDDDDDD',off_policy_correction.shape,topk_correction.shape,ce_loss_main.shape)# (?,) (?,) (?,)
            # 最开始此处有个负号，不应该有负号。
            self.pi_loss = tf.reduce_mean(off_policy_correction*topk_correction*self.discounted_episode_rewards_norm*ce_loss_main)
            tf.summary.scalar('pi_loss',self.pi_loss)

            self.beta_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights_beta,bias_beta,label,state,5,num_classes=self.item_count))
            tf.summary.scalar('beta_loss',self.beta_loss)

        with tf.variable_scope('optimizer'):
            # beta_vars = [var for var in tf.trainable_variables() if 'item_emb_beta' in var.name or 'bias_beta' in var.name]
            beta_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='beta_policy')
            self.train_op_pi = tf.train.AdamOptimizer(0.01).minimize(self.pi_loss)
            self.train_op_beta = tf.train.AdamOptimizer(0.01).minimize(self.beta_loss,var_list=beta_vars)

    def train(self):
        pi = []
        beta=[]
        merged = tf.summary.merge_all()
        counter = 1
        for epoch in range(self.epochs):
            for idx in range(self.num_batches):
                hist = self.historys[idx*self.batch_size:(idx+1)*self.batch_size]
                actions = self.actions[idx*self.batch_size:(idx+1)*self.batch_size]
                rewards = self.rewards[idx*self.batch_size:(idx+1)*self.batch_size]

                pi_old,beta_old = self.sess.run([self.PI,self.beta],feed_dict={self.input:hist})
                pi_new,beta_new,pi_loss,beta_loss,_,_,summary= self.sess.run([self.PI,self.beta,self.pi_loss,self.beta_loss,
                                                                              self.train_op_pi,self.train_op_beta,merged],
                                                                             feed_dict={self.input:hist,
                                                                                        self.label:actions,
                                                                                        self.discounted_episode_rewards_norm:rewards})

                print('ite:{},epoch:{},pi loss:{:.2f},beta loss:{:.2f}'.format(counter,epoch,pi_loss,beta_loss))
                pi.append(pi_loss)
                beta.append(beta_loss)
                self.log_writer.add_summary(summary,counter)
                counter+=1
                kl_pi = np.mean(np.sum(pi_old * (
                        np.log(pi_old + 1e-10) - np.log(pi_new + 1e-10)),
                                       axis=1)
                                )
                kl_beta = np.mean(np.sum(beta_old * (
                        np.log(beta_old + 1e-10) - np.log(beta_new + 1e-10)),
                                         axis=1)
                                  )
                if (kl_pi > self.kl_targ * 4) and (kl_beta>self.kl_targ*4) :  # early stopping if D_KL diverges badly
                    self.save_model(step=counter)
                    break

        # 保存模型
        self.save_model(step=counter)
        return pi,beta

    # 取前10个后10个的均值
    def plot_pi(self,pi_loss,num=10):
        pi_loss_ = [np.mean(pi_loss[ind-num:ind+num]) for ind ,val in enumerate(pi_loss) if ind%5000==num]
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        plt.subplot(211)
        plt.plot(range(len(pi_loss)),pi_loss,label='pi-loss',color='g')

        plt.subplot(212)
        plt.plot(range(len(pi_loss_)),pi_loss_,label='pi-loss',color='g')
        plt.xlabel('Training Steps')
        plt.ylabel('pi-loss')
        plt.legend()
        # plt.show()
        plt.savefig('jpg/reinforce_top_k_pi_hustle.jpg')

    def plot_beta(self,beta_loss,num=10):
        # pi_loss_ = [val for ind ,val in enumerate(pi_loss) if ind%5000==0]
        # beta_loss_ = [val for ind ,val in enumerate(beta_loss) if ind%5000==0]
        beta_loss_ = [np.mean(beta_loss[ind-num:ind+num]) for ind ,val in enumerate(beta_loss) if ind%5000==num]
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        plt.subplot(211)
        plt.plot(range(len(beta_loss)),beta_loss,label='beta-loss',color='r')

        plt.subplot(212)
        plt.plot(range(len(beta_loss_)),beta_loss_,label='beta-loss',color='r')
        plt.xlabel('Training Steps')
        plt.ylabel('beta-loss')
        plt.legend()
        # plt.show()
        plt.savefig('jpg/reinforce_top_k_beta_hustle.jpg')


if __name__ == '__main__':
    t1 = time.time()
    print('start model training.......{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))))
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    with tf.Session(config=config) as sess:
        reinforce = TopKReinforce(sess,item_count=6040,epochs=500,time_step=15,batch_size=256)
        print('model config :{}'.format(reinforce))
        pi_loss,beta_lss = reinforce.train()
        reinforce.plot_pi(pi_loss)
        reinforce.plot_beta(beta_lss)
    t2 = time.time()
    print('model training end~~~~~~{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t2))))
    print('time cost :{} m'.format((t2-t1)/60))






