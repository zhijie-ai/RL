#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/1 22:33                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

class Categorical_DQN():
    v_min = 0
    v_max = 1000
    atoms = 51

    # ENV_NAME = "CartPole-v1"
    ENV_NAME = 'Breakout-v0'  # 0: hold   1: throw the ball   2: move right   3: move left
    # ENV_NAME = "Freeway-v0"
    GAMMA = 0.99  # discount factor for target Q
    START_TRAINING = 1000  # experience replay buffer size
    BATCH_SIZE = 64  # size of minibatch
    UPDATE_TARGET_NET = 400  # update eval_network params every 200 steps
    LEARNING_RATE = 0.01
    MODEL_PATH = './model/C51DQN_model'

    INITIAL_EPSILON = 0.9  # starting value of epsilon
    FINAL_EPSILON = 0.05  # final value of epsilon
    EPSILIN_DECAY = 0.9999

    replay_buffer_size = 2000
    iteration = 5
    episode = 300  # 300 games per iteration
