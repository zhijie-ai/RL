#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/9/30 15:46                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
data = np.random.randn(100)

plt.plot(range(len(data)),data)
# plt.show()
plt.savefig('test.jpg')