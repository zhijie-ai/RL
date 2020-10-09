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
import time
import math
import pickle
from tensorflow.contrib import rnn

# 主网络和beta网络的实现

# topk修正后的概率
def cascade_model(p,k):
    return 1-(1-p)**k

# lambda比重
def gradient_cascade(p, k):
    return k*(1-p)**(k-1)

def get_index(actions):
    idx = []
    for i,v in enumerate(actions):
        idx.append([i,v])
    return idx


def load_data(path='../data/session.pickle',time_step=7,gamma=0.95):
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
        #discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        #discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    with open(path,'rb') as f:
        trajectory,rewards_= pickle.load(f)
        for t,r in zip(trajectory,rewards_):
            r = _discount_and_norm_rewards(r)
            for i in range(len(t)-time_step):
                historys.append(list(t[i:i+time_step]))
                actions.append(t[i+time_step])
                rewards.append(r[i+time_step])


    return np.array(historys),np.array(actions),np.array(rewards)


class TopKReinforce():
    def __init__(self,sess,item_count,embedding_size=64,is_train=True,topK=1,
                 weight_capping_c=math.e**3,batch_size=128,epochs = 1000,gamma=0.95):
        self.sess = sess
        self.item_count=item_count
        self.embedding_size=embedding_size
        self.rnn_size = 128
        self.log_out = 'out/logs'
        self.topK = topK
        self.weight_capping_c = weight_capping_c# 方差减少技术中的一种 weight capping中的常数c
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma

        self.historys,self.actions,self.rewards = load_data()
        print('AAAAAAAA',self.historys.shape,self.actions.shape,self.rewards.shape)
        self.num_batches = len(self.rewards) // self.batch_size

        self._init_graph()

        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.log_writer = tf.summary.FileWriter(self.log_out, self.sess.graph)


        if not is_train:
            self.restore_model()


    def weight_capping(self,cof):
        # return min(cof,self.weight_capping_c)
        return tf.where(cof)

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

    def _init_graph(self):
        with tf.variable_scope('input'):
            self.input = tf.placeholder(shape=[None,7],name='X',dtype=tf.int32)
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

        with tf.variable_scope('beta_policy'):
            weights_beta=tf.get_variable('item_emb_beta',[self.item_count,self.rnn_size])
            bias_beta = tf.get_variable('bias_beta',[self.item_count])
            self.beta =tf.add(tf.matmul(state,tf.transpose(weights_beta)),bias_beta)
            self.beta =  tf.nn.softmax(self.beta)# β策略


        label = tf.reshape(self.label,[-1,1])
        with tf.variable_scope('loss'):
            prob_weights = self.PI
            action = list(map(lambda x:np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel()),prob_weights))
            p_at = list(self.beta[9])

            ce_loss_main =tf.nn.sampled_softmax_loss(
                weights,bias,label,state,5,num_classes=self.item_count)
            topk_correction =gradient_cascade(self.PI,self.topK)# lambda 比值
            off_policy_correction = self.weight_capping(ratio)
            print('DDDDDDD',off_policy_correction.shape,topk_correction.shape,ce_loss_main.shape)#(?, 10000) (?, 10000) (?,)
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
        merged = tf.summary.merge_all()
        counter = 1
        for epoch in range(self.epochs):
            for idx in range(self.num_batches):
                hist = self.historys[idx*self.batch_size:(idx+1)*self.batch_size]
                actions = self.actions[idx*self.batch_size:(idx+1)*self.batch_size]
                self.ind = get_index(actions)
                rewards = self.rewards[idx*self.batch_size:(idx+1)*self.batch_size]

                print('CCCCCCCCCC',hist.shape,actions.shape,rewards.shape)

                pi_loss,beta_loss,summary= self.sess.run([self.train_op_pi,self.train_op_beta,merged],
                                                  feed_dict={self.input:hist,
                                                             self.label:actions,
                                                             self.discounted_episode_rewards_norm:rewards})

                print('ite:{},pi loss:{:.2f},beta loss:{:.2f}'.format(counter,pi_loss,beta_loss))
                self.log_writer.add_summary(summary,counter)
                counter+=1


if __name__ == '__main__':
    with tf.Session() as sess:
        tkr = TopKReinforce(sess,item_count=10000)
        tkr.train()






