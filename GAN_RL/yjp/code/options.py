#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/8 14:31                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse



def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Args for recommendation system reinforce_gan model")

    #-----------------------------------------env training---------------------------------------------
    parser.add_argument('--click_path',type =str,default ='../data/raw/click.csv',help = 'dataset_folder')
    parser.add_argument('--exposure_path',type =str,default ='../data/raw/exposure.csv',help = 'dataset_folder')
    parser.add_argument('--data_folder',type =str,default ='../data/handled/',help = 'dataset_folder')
    parser.add_argument('--save_dir',type = str,default = '../model/save_dir/',help='save folder')
    parser.add_argument('--embedding_path',type = str,default = '/data1/ai-recall/bpr/model/',help='save folder')
    parser.add_argument('--random_seed',type = int,default = '1126',help='random seed')
    parser.add_argument('--clip_min_value',type = float,default = -40.0,help='tf clip min value')
    parser.add_argument('--clip_max_value',type = float,default = 40.0,help='tf clip max value')

    parser.add_argument('--resplit', type=eval, default=False)
    parser.add_argument('--num_thread', type=int, default=1, help='number of threadings')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for env training')
    parser.add_argument('--init_learning_rate', type=float, default=0.01, help='learning rate for env training')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_iters', type=int, default=50, help='num of iterations for env training')
    # might change later to policy_grad method with attetion rather than lstm
    parser.add_argument('--rnn_hidden_dim', type=int, default=20, help='LSTM hidden sizes')
    parser.add_argument('--pw_dim', type=int, default=4, help='position weight dim')
    parser.add_argument('--pw_band_size', type=int, default=20, help='position weight banded size (i.e. length of history)')
    parser.add_argument('--compu_type', type=str, default='thread', help='computation unit. only process or thread')


    parser.add_argument('--dims', type=str, default='256-32')
    parser.add_argument('--user_model', type=str, default='PW', help='architecture choice: LSTM or PW')
    # dont think that the PW model could be used atm

    #-----------------------------------------env---------------------------------------------
    parser.add_argument('--k', type=int, default=10, help='num of recommendation for each time')
    parser.add_argument('--noclick_weight', type=float, default=0.3, help='threshold for click. if >noclick_weight means user will having a positive callback ')
    parser.add_argument('--env_std', type=float, default=1e-3, help='std for normal initialization ')

    #-----------------------------------------dqn---------------------------------------------
    parser.add_argument('--train_time_horizon', type=int, default=15, help='time step for collecting data with env while training')
    parser.add_argument('--vali_time_horizon', type=int, default=15, help='time step for collecting data with env while validation or test')
    parser.add_argument('--dqn_type', type=str, default='offline', help='collection data with env online or offline.')
    parser.add_argument('--q_std', type=float, default=1e-2, help='std for dqn normal initialization .')
    parser.add_argument('--dqn_dims', type=str, default='256-32')
    parser.add_argument('--dqn_lr', type=float, default=1e-3, help='learning rate for q-learning')
    parser.add_argument('--min_value', type=float, default=-1000000000.0, help='min value')
    parser.add_argument('--sample_batch_size', type=int, default=1024,help='num of choosing user')
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--training_batch_size', type=int, default=1024,help='batch size of training')
    parser.add_argument('--vali_batch_size', type=int, default=2000,help='batch size of validation')
    parser.add_argument('--model_path', type =str,default ='../model/dqn/',help = 'dqn model path')
    parser.add_argument('--epoch', type =int,default =5,help = 'iteration for train')

    opts = parser.parse_args(args)

    return opts