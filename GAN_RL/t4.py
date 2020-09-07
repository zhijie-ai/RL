#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/9 17:36                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np

data = np.random.randn(6,20)
indices = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
data = tf.convert_to_tensor(data)
indices = tf.convert_to_tensor(indices)

clk_tensor_indice=[[0, 0], [1, 2], [2, 3], [3, 0], [4, 2], [5, 4]]
clk_tensor_val = np.ones(len(clk_tensor_indice))
denseshape=[6,6]
click_tensor = tf.SparseTensor(clk_tensor_indice, clk_tensor_val, denseshape)
densetensor = tf.sparse_tensor_to_dense(click_tensor)
res = tf.reduce_sum(densetensor,axis=1)
click_cnt = tf.sparse_reduce_sum(click_tensor, axis=1)

disp_tensor_indice =[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [3, 0], [3, 2], [3, 4], [3, 5], [4, 0], [4, 2], [4, 4], [4, 5], [5, 0], [5, 2], [5, 4], [5, 5]]
exp_u_disp = np.random.randn(27,1)
exp_disp_ubar_ut = tf.SparseTensor(disp_tensor_indice, tf.reshape(exp_u_disp, [-1]), denseshape)
dense_exp_disp_util = tf.sparse_tensor_to_dense(exp_disp_ubar_ut, default_value=0.0)

argmax_click = tf.argmax(tf.sparse_tensor_to_dense(click_tensor, default_value=0.0), axis=1)
argmax_disp = tf.argmax(dense_exp_disp_util, axis=1)

top_2_disp = tf.nn.top_k(dense_exp_disp_util, k=2, sorted=False)[1]
# d = [0, 2 ,3 ,0 ,2, 4]
# d2 = tf.nn.top_k(d, k=2, sorted=False)[1]

precision_1_sum = tf.reduce_sum(tf.cast(tf.equal(argmax_click, argmax_disp), tf.float32))
precision_1 = precision_1_sum / 6
precision_2_sum = tf.reduce_sum(
    tf.cast(tf.equal(tf.reshape(argmax_click, [-1, 1]), tf.cast(top_2_disp, tf.int64)), tf.float32))
precision_2 = precision_2_sum / 6

d1 = [[0],[2],[3],[4],[6]]
d2 = [[0,0],[2,1],[0,2],[1,3],[2,5]]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.gather(data,indices)).shape)
    print(sess.run(densetensor))
    print(sess.run(res))
    print(sess.run(click_cnt))
    print(sess.run(tf.reduce_sum(click_cnt)))
    print(sess.run(tf.reshape(exp_u_disp, [-1])))
    print('==============')
    print(sess.run(argmax_click))
    print(sess.run(argmax_disp))
    print(sess.run(top_2_disp))
    print(sess.run(tf.cast(tf.equal(tf.reshape(argmax_click, [-1, 1]), tf.cast(top_2_disp, tf.int64)), tf.float32)))
    print(sess.run(precision_2))
    print('=================')
    print(sess.run(tf.reshape(argmax_click, [-1, 1])))
    print(sess.run(tf.cast(top_2_disp, tf.int64)))
    print(sess.run(tf.equal(d1,d2)))
