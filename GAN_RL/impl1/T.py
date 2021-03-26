#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/14 14:24                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import tensorflow as tf

#三角矩阵应该是这样的
t_indice=[]
sec_cnt_x=0
data_time=[6,4]
b_size = 20
for u in range(len(data_time)):
    for kk in range(min(b_size, data_time[u])):
        t_indice += map(lambda x: [x + kk  + sec_cnt_x, x + sec_cnt_x], np.arange(data_time[u] - (kk)))


    sec_cnt_x += data_time[u]
    print('AAA',t_indice,len(t_indice),sec_cnt_x)
    tril_value_indice = np.array(list(map(lambda x: (x[0] - x[1]), t_indice)))
    print('BBB',tril_value_indice)
# tril_value_indice = [-0.38675551, -0.38675551, -0.38675551, -0.38675551, -0.38675551,
#                      -0.38675551, -1.09132216, -1.09132216, -1.09132216, -1.09132216,
#                      -1.09132216,  0.290232  ,  0.290232  ,  0.290232  ,  0.290232  ,
#                      -0.70196606, -0.70196606, -0.70196606, -0.70893215, -0.70893215,
#                      -0.61415437]

shape = np.array([sec_cnt_x,sec_cnt_x], dtype=np.int64)
t_indice.sort(key= lambda x:(x[0],x[1]))
position_weight = tf.get_variable('p_w', [20],initializer=tf.constant_initializer(2))
cumsum_tril_value = tf.gather(position_weight, tril_value_indice)
print(t_indice)

# t_indice=[[0, 0], [1, 1], [2, 2], [3, 4]]
# t_indice=[ [1, 1], [2, 2], [2, 1],[3, 4]]#Invalid argument: indices[2] = [2,1] is out of order
# tril_value_indice = [1, 2, 3, 4]
# shape = [5,6]

mat = tf.SparseTensor(t_indice, cumsum_tril_value,dense_shape=shape)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
res = sess.run(mat)
# print(res)
print(sess.run(tf.sparse_tensor_to_dense(res)))
# print(sess.run(tf.sparse_tensor_to_dense(mat)))


