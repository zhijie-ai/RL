#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/5 12:56                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np

data = np.random.randn(6)
da=[[],[]]
data2 = np.random.randn(3,4)
data3 = np.random.randn(3,4)
da[0]=data2
da[1]=data2
print(data)
indices = [[0,1],[2,3]]
data = tf.convert_to_tensor(data)
indices = tf.convert_to_tensor(indices)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.gather(data,indices)))
    print(sess.run(tf.concat(da,0)).shape)
    print(sess.run(tf.concat(da,axis=1)).shape)
    print(sess.run(tf.concat(data2  ,axis=1)).shape)