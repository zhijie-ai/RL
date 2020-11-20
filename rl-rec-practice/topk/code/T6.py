#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/11/9 15:58                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import tensorflow as tf
import numpy as np

num_unit = 100
cell = tf.nn.rnn_cell.LSTMCell(num_unit)

input = np.random.rand(32, 100)
inputs = tf.constant(value=input, shape=(32, 100), dtype=tf.float32)
states = tf.placeholder(tf.float32,[32,100],name='rnn_state')
inputs = tf.placeholder(tf.float32,[32,100])

h0 = cell.zero_state(32, np.float32)
print(type(h0))#LSTMStateTuple
print(h0)
output, h1 = cell.__call__(inputs, (states,states))# 因为LSTM的state为(c,h)的元组
output, h1 = cell(inputs, h0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(output))
    print(sess.run(tf.shape(output)))
    print(sess.run(tf.shape(h1.c)))