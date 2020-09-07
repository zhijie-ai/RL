#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/27 下午3:22                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell

state = np.random.randn(1,4,19*15)
state_= tf.convert_to_tensor(state,dtype=tf.float32)


def cli_value(x, v):
    x = tf.cast(x, tf.int64)
    y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
    return tf.where(tf.greater(x, y), x, y)


def _gather_last_output(data, seq_lens):
    this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype=tf.int64)
    tmp_end = tf.map_fn(lambda x: cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
    indices = tf.stack([this_range, tmp_end], axis=1)
    return tf.gather_nd(data, indices)



cell = rnn_cell.GRUCell(19,activation=tf.nn.relu)
outputs, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32,sequence_length=[19])



outputs1 = _gather_last_output(outputs, [19])
layer1 = tf.layers.Dense(64, activation=tf.nn.relu)(outputs1)
layer2 = tf.layers.Dense(32, activation=tf.nn.relu)(layer1)
outputs = tf.layers.Dense(19, activation=tf.nn.tanh)(layer2)

print('Actor outputs shape',outputs.shape)

# 不使用GPU,2种方式
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

with tf.Session(config=tf.ConfigProto(device_count={'gpu':-1})) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(outputs).shape)

