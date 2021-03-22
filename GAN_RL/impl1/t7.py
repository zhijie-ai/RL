#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/3/22 11:43                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import tensorflow as tf
import numpy as np

data_click = [[],[]]
click_tensor_indice=[]
data_click[0] = [[0,0],[1,2],[2,5]]
data_click[1] = [[0,1],[1,4],[2,3]]

sec_cnt_x = 0
click_tensor_indice_tmp = list(map(lambda x: [x[0] + sec_cnt_x, x[1]], data_click[0]))
print(click_tensor_indice_tmp)
click_tensor_indice += click_tensor_indice_tmp
print(click_tensor_indice)

sec_cnt_x= 3
click_tensor_indice_tmp = list(map(lambda x: [x[0] + sec_cnt_x, x[1]], data_click[0]))
print(click_tensor_indice_tmp)
click_tensor_indice += click_tensor_indice_tmp
print(click_tensor_indice)
clk_tensor_val = np.ones(len(click_tensor_indice))

denseshape = [6, 10]
click_tensor = tf.SparseTensor(click_tensor_indice, clk_tensor_val, denseshape)

sess  = tf.Session()
res = sess.run(click_tensor)
print(sess.run(tf.sparse_tensor_to_dense(click_tensor)))


click_cnt = tf.sparse_reduce_sum(click_tensor, axis=1)
print(sess.run(click_cnt))

argmax_click = tf.argmax(tf.sparse_tensor_to_dense(click_tensor, default_value=0.0), axis=1)#点击


