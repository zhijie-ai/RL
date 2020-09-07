#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/6/15 下午6:44                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np
import time

# 主网络和beta网络的实现

class MainPolicy():
    def __init__(self,item_count,embedding_size=64,is_train=True):
        self.item_count=item_count
        self.embedding_size=embedding_size
        self.log_out = 'out/logs'

        self._init_graph()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.log_writer = tf.summary.FileWriter(self.log_out, self.session.graph)

        if not is_train:
            self.restore_model()