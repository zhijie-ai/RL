#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/10/23 12:03                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class MYM(keras.Model):
    def __init__(self):
        ipt1 = keras.Input(shape=(13,13,7),name="view")
        ipt2 = keras.Input(shape=(34),name="feature")
        x = layers.Conv2D(7,kernel_size=3)(ipt1)
        x = layers.Conv2D(1,kernel_size=3)(x)
        x = layers.Flatten()(x)
        #print(x,ipt2)
        x =tf.concat([x,ipt2],axis=-1)
        x = layers.Dense(128)(x)
        #print(x)
        x = layers.Dense(21)(x)
        out = layers.Softmax(axis=-1)(x)
        super(MYM,self).__init__(inputs=[ipt1,ipt2],outputs=out)

net = MYM()
net.summary()


# x = np.random.rand(13,13,7)
# ix=np.array([x,x])
# x2 = np.random.rand(34)
# ix2=np.array([x2,x2])
# print(ix.shape,ix2.shape)
#
# y=net(inputs=[ix,ix2])
# print(y.shape)
# tf.argmax(y,axis=-1)