#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/7/31 15:46                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np
import  pandas as pd

day = [0,0,1,1,2]
uid = [1,2,1,1,3]
aid = [1,2,1,1,2]

df = pd.DataFrame({'day':day,'uid':uid,'aid':aid})
print(df)

dic={}
day=0
sentence = []
df = df.sort_values(by='day')
for item in df.values:
    if day != item[0]:
        for key in dic:
            sentence.append(dic[key])
        dic={}
        day=item[0]
    try:
        dic[item[1]].append(str(int(item[2])))
    except :
        dic[item[1]] = [str(int(item[2]))]

for key in dic:
    sentence.append(dic[key])


print(sentence)