#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/11/24 17:07                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import tensorflow as tf

ckpt = tf.train.get_checkpoint_state('checkout/model_prior_rnn')
print(ckpt)
print(ckpt.model_checkpoint_path)
print('AAAAAAAAAAAAAAAA')
print(tf.train.latest_checkpoint('checkout/model_prior_rnn'))