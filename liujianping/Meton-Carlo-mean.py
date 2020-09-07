#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/12/11 15:07
 =================知行合一=============
'''

import numpy as np

#蒙特卡洛方法求积分
# 1、投点法
# 2、均值法

x=np.random.uniform(0,1,10**7)
def f(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)

print(np.mean(f(x)))

