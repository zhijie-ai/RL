#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/3/19 12:04                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import tensorflow as tf
import numpy as np

indices = np.array([[0, 0], [1, 1], [2, 2], [3, 4]], dtype=np.int32)
values = np.array([1, 2, 3, 4], dtype=np.int32)
shape = np.array([5, 5], dtype=np.int32)
x = tf.SparseTensor(values=values,indices=indices,dense_shape=shape)
print(x)

with tf.Session() as sess:
    result = sess.run(x)
    print(result)

    result_value = tf.sparse_tensor_to_dense(result)
    print('value:\n', sess.run(result_value))