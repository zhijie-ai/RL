#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/17 12:29                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
class NoisyNetDQNConfig():
    # ENV_NAME='CartPole-v1'
    ENV_NAME='Breakout-v0'#0:hold 1:throw the ball 2:move right 3:move left
    # ENV_NAME='Freeway-v0'
    GAMMA = 0.99 #Discount factor for target Q
    START_TRAINING=1000 # experiment replay buffer size
    BATCH_SIZE=64 # size for minibatch
    UPDATE_TARGET_NET = 400 # update eval_network params every 200 steps
    LEARNING_RATE = 0.01
    MODEL_PATH = './modelNosiyNetDQN_model'

    INITAIL_EPSILON=1.0 # starting value of epsilon
    FINAL_EPSILON=0.01 # final value of epsilon
    EPSILON_DECAY=0.999

    replay_buffer_size=2000
    ieteration=5
    episode = 300 # 300 games per iteration

    nosiy_distribution='factorised' # independent or factorised