#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/11/24 11:21                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1,10,20)
y = np.sin(x)*3
std = np.random.rand(20)
plt.errorbar(x,y,yerr=std,fmt='o-',ecolor='r',color='b',elinewidth=2,capsize=4)
plt.show()