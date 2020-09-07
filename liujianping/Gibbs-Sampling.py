#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/12/17 10:44
 =================知行合一=============
'''

import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
# samplesource = multivariate_normal(mean=[5,-1],cov=[[1,0,5],[0.5,2]])
samplesource = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])

rho=2
def p_ygivenx(x,m1,m2,s1,s2):
    return (random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt(1 - rho ** 2) * s2))

def p_xgiveny(y, m1, m2, s1, s2):
    return (random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt(1 - rho ** 2) * s1))

N=5000
K=20
x_res=[]
y_res=[]
z_res=[]
m1=5
m2=-1
s1=1
s2=2

rho=0.5
y=m2

for i in range(N):
    for j in range(K):
        x= p_xgiveny(y,m1,m2,s1,s2)
        y = p_ygivenx(x,m1,m2,s1,s2)
        z = samplesource.pdf([x,y])
        x_res.append(x)
        y_res.append(y)
        z_res.append(z)

num_bins = 50
plt.hist(x_res,num_bins,normed=1,facecolor='green',alpha=0.5)
plt.hist(y_res,num_bins,normed=1,facecolor='red',alpha=0.5)
plt.title('Histogram')



fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(x_res, y_res, z_res,marker='o')
plt.show()

