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

num_units = [50, 200, 300]
cells = [tf.nn.rnn_cell.LSTMCell(num_unit) for num_unit in num_units]
mul_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
print(mul_cells.state_size.shape)

input = np.random.rand(32, 100)
inputs = tf.constant(value=input, shape=(32, 100), dtype=tf.float32)

h0 = mul_cells.zero_state(32, np.float32)
output, h1 = mul_cells.__call__(inputs, h0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))
    print(sess.run(tf.shape(output)))
    print(sess.run(tf.shape(h1[0].c)))
    print(sess.run(tf.shape(h1[1].c)))
    print(sess.run(tf.shape(h1[2].c)))