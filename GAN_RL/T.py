#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/14 14:24                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
t_indice=[]
sec_cnt_x=0
for kk in range(min(20, 10)):
    print('kk',kk)
    print('np.arange(20 - (kk + 1))',np.arange(10 - (kk + 1)))
    t_indice += map(lambda x: [x + kk + 1 + sec_cnt_x, x + sec_cnt_x], np.arange(10 - (kk + 1)))
    print(len(list(t_indice)))

print(len(list(t_indice)))
print('-------------')

data=[[10,1],[20,2]]
print(list(map(lambda x:[x[0]+0,x[1]],data)))

news_dict={}
d = [[1,[1,2,3],[1,2,3,4,5]],[2,[6,7,8],[6,7,8,9,10,16]]]
for event in range(2):
    disp_list = d[event][2]
    pick_list = d[event][1]
    for id in disp_list:  # 展示
        if id not in news_dict:
            news_dict[id] = len(news_dict)  # for each user ,news id start from 0
print(''.center(10,'A'))
print(news_dict)
print(len(news_dict))
print(''.center(10,'A'))


