#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/12 22:24                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import tensorflow as tf

data = np.arange(24).reshape(2,3,4)
print(data)
indices = [[0,1],[1,2]]
indices2 = [[0,1,1],[1,1,2]]
res = tf.gather_nd(data,indices)
res2 = tf.gather_nd(data,indices2)
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    print(sess.run(res))
    print(sess.run(res2))