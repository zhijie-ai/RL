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
from tensorflow.contrib import rnn

# 主网络和beta网络的实现

class Reinforce():
    def __init__(self,item_count,embedding_size=64,is_train=True):
        self.item_count=item_count
        self.embedding_size=embedding_size
        self.rnn_size = 128
        self.log_out = 'out/logs'

        self._init_graph()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.log_writer = tf.summary.FileWriter(self.log_out, self.session.graph)

        if not is_train:
            self.restore_model()


    def _init_graph(self):
        with tf.variable_scope('input'):
            self.history = tf.placeholder([None,None],name='X')
            self.label = tf.placeholder([None],name='action')

        cell = rnn.BasicLSTMCell(self.rnn_size)
        with tf.variable_scope('emb'):
            embedding = tf.get_variable('emb_w',[self.item_count,self.embedding_size])
            inputs = tf.nn.embedding_lookup(embedding,self.history)

        outputs,_ = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32)

        #state = tf.reshape(outputs,[-1,self.rnn_size*])
        state = tf.nn.relu(outputs[:,-1,:])

        with tf.variable_scope('main_policy'):
            weights=tf.get_variable('item_emb',[self.item_count,self.rnn_size])
            bias = tf.get_variable('bias',[self.item_count])

        with tf.variable_scope('beta_policy'):
            weights_beta=tf.get_variable('item_emb_beta',[self.item_count,self.rnn_size])
            bias_beta = tf.get_variable('bias_beta',[self.item_count])

        with tf.variable_scope('loss'):
            self.loss_main = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights,bias,self.label,state,5,num_classes=self.item_count))

            self.loss_beta = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights_beta,bias_beta,self.label,state,5,num_classes=self.item_count))


        # beta_vars = [var for var in tf.trainable_variables() if 'item_emb_beta' in var.name or 'bias_beta' in var.name]
        beta_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='beta_policy')
        self.train_op_main = tf.train.AdamOptimizer(0.01).minimize(self.loss_main)
        self.train_op_main = tf.train.AdamOptimizer(0.01).minimize(self.loss_beta,var_list=beta_vars)


    def train(self,X,y):
        _,loss = self.sess.run([self.train_op,self.loss],feed_dict={self.history:X,self.label:y})








