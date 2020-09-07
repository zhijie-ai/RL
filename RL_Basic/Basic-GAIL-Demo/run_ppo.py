#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/28 14:20                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import sys
import os
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from algo.ppo import PPOTrain

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir',help='log directory',default='log/train/ppo')
    parser.add_argument('--savedir',help='save directory',default='trained_models/ppo')
    parser.add_argument('--gamma',default=0.95,type=float)
    parser.add_argument('--iteration',default=int(1e6),type=int)

    return parser.parse_args()

def main(args):
    env=gym.make('CartPole-v0')
    env.seed(0)
    ob_space=env.observation_space
    Policy = Policy_net('policy',env)
    Old_Policy=Policy_net('old_policy',env)
    PPO=PPOTrain(Policy,Old_Policy,gamma=args.gamma)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir,sess.graph)
        sess.run(tf.global_variables_initializer())
        obs=env.reset()
        success_num=0

        for iteration in range(args.iteration):
            observations=[]
            actions=[]
            rewards=[]
            v_preds=[]
            episode_length=0

            while True:
                episode_length+=1
                obs=np.stack([obs]).astype(dtype=np.float32)#prepare to feed placeholder Policy.obs
                act,v_pred=Policy.act(obs=obs,stochastic=True)
                act=np.asscalar(act)
                v_pred=np.asscalar(v_pred)

                next_obs,reward,done,info=env.step(act)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    next_obs=np.stack([next_obs]).astype(dtype=np.float32)#prepare to feed placeholder Policy.obs
                    _,v_pred=Policy.act(next_obs,stochastic=True)
                    v_preds_next=v_preds[1:]+[np.asscalar(v_pred)]#如果此episode结束了，下一个状态的V(s)是自己定义的。
                    obs=env.reset()
                    break
                else:
                    obs=next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length',
                                                                  simple_value=episode_length)]),
                               iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward',
                                                                  simple_value=sum(rewards))]),
                               iteration)

            if sum(rewards)>195:
                success_num+=1
                if success_num>=100:
                    if not os.path.exists(args.savedir):
                        os.makedirs(args.savedir)
                    saver.save(sess,args.savedir+'/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num=0

            gaes=PPO.get_gaes(rewards=rewards,v_preds=v_preds,v_preds_next=v_preds_next)

            #convert list to numpy array for feeding tf.placeholder
            observations=np.reshape(observations,newshape=(-1,)+ob_space.shape)
            actions=np.array(actions).astype(dtype=np.int32)
            gaes=np.array(gaes).astype(dtype=np.float32)
            gaes=(gaes-gaes.mean())/gaes.std()
            rewards=np.array(rewards).astype(dtype=np.float32)
            v_preds_next=np.array(v_preds_next).astype(dtype=np.float32)

            PPO.assign_policy_parameters()

            inp=[observations,actions,gaes,rewards,v_preds_next]

            #train
            for epoch in range(6):
                # sample indics from [low,high]
                sample_indices=np.random.randint(low=0,high=observations.shape[0],size=32)
                sample_inp=[np.take(a=a,indices=sample_indices,axis=0) for a in inp] # sampe training data
                PPO.train(obs=sample_inp[0],
                          actions=sample_inp[1],
                          gaes=sample_inp[2],
                          rewards=sample_inp[3],
                          v_preds_next=sample_inp[4])
            summary=PPO.get_summary(obs=inp[0],
                                    actions=inp[1],
                                    gaes=inp[2],
                                    rewards=inp[3],
                                    v_preds_next=inp[4])
            writer.add_summary(summary,iteration)
            print(success_num)
        print(success_num,'------------')
        writer.close()

if __name__ == '__main__':
    args=argparser()
    main(args)