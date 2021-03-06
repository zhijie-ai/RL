#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/28 14:16                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 1.训练一个高手
#   首先运行run_ppo.py文件，该文件会通过PPO算法来训练一个agent，
#   如果训练的agent能够满足我们的高手定义，则终止学习，此时学习到的agent我们就认为是一个高手。
# 2.得到高手的交互序列
#   接下来，我们运行sample_trajectory.py文件，得到高手的状态-动作对，保存在trajectory文件夹下。
# 3.训练GAIL模型
#   接下来运行run_gail.py文件，基于上述介绍的GAIL的基本思路，来进行学习。
# 4.检验学习成果
#   最后，运行test_policy.py文件，我们可以检验一下我们通过GAIL学习到的agent的学习成果。