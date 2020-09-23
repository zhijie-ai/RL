#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/22 11:58                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
t_indice=[]
sec_cnt_x=0
for kk in range(min(5, 6)):
    print('kk',kk)
    t_indice += map(lambda x: [x + kk + 1 + sec_cnt_x, x + sec_cnt_x], np.arange(6 - (kk + 1)))
    print('t_indice',t_indice)

tril_value_indice = map(lambda x: (x[0] - x[1] - 1), t_indice)
print(t_indice)
print(list(tril_value_indice))

data_click = [[0, 0], [1, 2], [2, 3], [3, 0], [4, 2], [5, 4]]
data_disp = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [3, 0], [3, 2], [3, 4], [3, 5], [4, 0], [4, 2], [4, 4], [4, 5], [5, 0], [5, 2], [5, 4], [5, 5]]
print(list(map(lambda x:data_disp.index(x),data_click)))#[0, 7, 13, 15, 20, 25]