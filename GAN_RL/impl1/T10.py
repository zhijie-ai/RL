#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/2 17:46                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np

def func(no_click=0.8):
    p = np.random.rand(10)
    disp_item = np.random.randint(0,100,10).tolist()
    no_click = [no_click]
    prob = np.array(p.tolist()+no_click)
    prob = prob / float(prob.sum())
    print(disp_item,prob)
    rand_choice = np.random.choice(disp_item + [-100], 1, p=prob)
    print(rand_choice)

if __name__ == '__main__':
    for _ in range(10):
        func()