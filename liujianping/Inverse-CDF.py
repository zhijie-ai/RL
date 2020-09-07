#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/12/13 11:29
 =================知行合一=============
'''
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#逆采样
m = 10000
u = [random.uniform(0,1) for x in range(m)]

def inverse(u):
    if u>=0 and u<0.25:
        return np.sqrt(u)/2
    elif u>=0.25 and u<=1:
        return 1-np.sqrt(3*(1-u))/2

x = np.array([inverse(u1) for u1 in u])
x_ser = pd.Series(x)
plt.plot(x,8*x,'bo')
plt.plot(x,((8/3)-(8/3)*x),'r+')
x_ser.plot(kind='kde')
plt.xlim((0,1))
plt.ylim((0,2))
plt.show()


