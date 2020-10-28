#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/10/28 11:38                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np
import pandas as pd
from collections import Counter


"""Remove all users who have read less than K Articles"""

data = pd.read_csv(r'User History Data.csv')

l1 = []
"""Users who have read less than K articles will be removed"""
K = 4
for i in range(len(data['userId'])):
    l1.append(data['userId'][i])

freq = Counter(l1)
res = [ele for ele in l1 if freq[ele] < K]

for i  in range(0,len(res)):
    data = data[data.userId != res[i]]

data.to_csv(r'Train Data.csv',index = False)

"""Create UserData"""
data1 = pd.read_csv(r'Train Data.csv')

"""Make a csv file for a dummy user with column of userId blank and having Articles Index and 0 as Action"""
userdata = pd.read_csv(r'Dummy User.csv')

"""Find all the unique userIDs"""
l2 = []
for i in range(0,len(data1['userId'])):
    l2.append(data1['userId'][i])

l2 = np.unique(l2)# 去重后的用户ID。

"""Fill userdata and create CSV for every user"""
for i in range(len(l2)):
    x = (l2[i]).__str__()
    user_i = userdata.fillna(l2[i])
    l3 = []# 存的是article_id
    for j in range(len(data1['userId'])):
        if data1['userId'][j] == l2[i]:
            l3.append(data1['article_id'][j])# l3存的是当前用户有过操作纪律的ariticle


    for k in range(len(user_i['userId'])):
        for l in range(len(l3)):
            if user_i['article_id'][k] == l3[l]:
                user_i['action'][k] = 1
    user_i.to_csv(r'user_' + x + '.csv',index = False)