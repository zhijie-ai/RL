#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/2 16:49                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np
import tensorflow as tf

sess = tf.Session()

def func(_noclick_weight=0,user=1,_k = 2):
    num = np.random.randint(-100,100,1)
    u_disp = np.random.randn(user*_k)
    u_disp =u_disp +num
    disp_2d_split_user_ind = np.kron(np.arange(user),np.ones(_k)).astype(int)
    exp_u_disp = tf.exp(u_disp)
    sum_exp_disp = tf.segment_sum(exp_u_disp, disp_2d_split_user_ind) + float(np.exp(_noclick_weight))
    scatter_sum_exp_disp = tf.gather(sum_exp_disp, disp_2d_split_user_ind)
    p_disp = tf.div(exp_u_disp, scatter_sum_exp_disp)#权重
    trans_p = tf.reshape(p_disp, [-1, _k])
    trans_p = sess.run(trans_p)
    if 1.0 in np.sum(trans_p,axis=1):
        print('AAAAA',np.sum(trans_p,axis=1),
              sess.run(exp_u_disp),sess.run(scatter_sum_exp_disp),np.exp(_noclick_weight))
    # print(np.sum([True for i in np.sum(trans_p,axis=1) if i>1]))

if __name__ == '__main__':
    for i in range(10):
        func(0)

    # print('=============_noclick_weight=0.1')
    # for i in range(10):
    #     func(0.1)
    #
    # print('=============_noclick_weight=0.4')
    # for i in range(10):
    #     func(0.4)
    #
    # print('=============_noclick_weight=0.8')
    # for i in range(10):
    #     func(0.8)
    #
    # print('=============_noclick_weight=1')
    # for i in range(10):
    #     func(1)
