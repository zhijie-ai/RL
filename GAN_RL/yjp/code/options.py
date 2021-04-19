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

    parser.add_argument('--click_path',type =str,default ='../data/raw/click.csv',help = 'dataset_folder')
    parser.add_argument('--exposure_path',type =str,default ='../data/raw/exposure.csv',help = 'dataset_folder')
    parser.add_argument('--data_folder',type =str,default ='../data/handled/',help = 'dataset_folder')
    parser.add_argument('--save_dir',type = str,default = '../model/save_dir/',help='save folder')
    parser.add_argument('--embedding_path',type = str,default = '/data1/ai-recall/bpr/model/',help='save folder')
    parser.add_argument('--random_seed',type = int,default = '1126',help='random seed')

    parser.add_argument('--resplit', type=eval, default=False)
    parser.add_argument('--num_thread', type=int, default=10, help='number of threadings')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_iters', type=int, default=50, help='num of iterations for q learning')
    # might change later to policy_grad method with attetion rather than lstm
    parser.add_argument('--rnn_hidden_dim', type=int, default=20, help='LSTM hidden sizes')
    parser.add_argument('--pw_dim', type=int, default=4, help='position weight dim')
    parser.add_argument('--pw_band_size', type=int, default=20, help='position weight banded size (i.e. length of history)')


    parser.add_argument('--dims', type=str, default='64-64')
    parser.add_argument('--user_model', type=str, default='PW', help='architecture choice: LSTM or PW')
    # dont think that the PW model could be used atm

    opts = parser.parse_args(args)

    return opts