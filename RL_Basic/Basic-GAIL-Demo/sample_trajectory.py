#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/27 21:02                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
#得到高手的状态-动作对，保存在trajectory文件夹

import sys
import os
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import gym
import numpy as np
from network_models.policy_net import Policy_net
import tensorflow as tf

# noisespection PyTypeChecker
def open_file_and_save(file_path,data):
    '''

    :param file_path:
    :param data:
    :return:
    '''
    try:
        with open(file_path,'ab') as f_handle:
            np.savetxt(f_handle,data,fmt='%s')
    except FileNotFoundError:
        with open(file_path,'wb') as f_handle:
            np.savetxt(f_handle,data,fmt='%s')

def argparser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',help='filename of model to test',default='trained_models/ppo/model.csv')
    parser.add_argument('--iteration',default=10,type=int)

    return parser.parse_args()

def main(args):
    env=gym.make('CartPole-v0')
    env.seed(0)
    ob_space=env.observation_space
    Policy=Policy_net('policy',env)
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(args.model):
            saver.restore(sess,args.model)
        obs=env.reset()
        print(obs)

        for iteration in range(args.iteration):# episode
            observations=[]
            actions=[]
            run_steps=0
            while True:
                run_steps+=1
                # prepare to feed placeholder Policy.obs
                obs=np.stack([obs]).astype(dtype=np.float32)

                act,_=Policy.act(obs=obs,stochastic=True)
                act=np.asscalar(act)

                observations.append(obs)
                actions.append(act)

                next_obs,reward,done,info=env.step(act)

                if done:
                    print(run_steps)
                    obs=env.reset()
                    break
                else:
                    obs=next_obs

            observations=np.reshape(observations,newshape=[-1]+list(ob_space.shape))
            actions=np.array(actions).astype(dtype=np.int32)

            open_file_and_save('trajectory/observations.csv',observations)
            open_file_and_save('trajectory/actions.csv',actions)

if __name__ == '__main__':
    args=argparser()
    main(args)