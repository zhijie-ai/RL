#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/6/15 下午3:12                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import pickle

np.random.seed(1234)

def gen_data(num_user=10000,items=100000):
    clicked_item=[]
    rewards=[]
    for i in range(num_user):
        step = np.random.randint(8,16)# 每个用户的session最多16步，最少8步
        clicked_item.append(np.random.choice(range(items),step,replace=False))
        rewards.append(np.random.randint(-10,10,step))# 0-20分选step个

    return np.array(clicked_item),np.array(rewards)






if __name__ == '__main__':
    # data = gen_data()
    # with open('../data/session.pickle', 'wb') as f:
    #     pickle.dump(data,f)
    
    with open('../data/session.pickle','rb') as f:
        data = pickle.load(f)
        print(len(data[0]))