#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/1/12 9:41                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import pandas as pd

df = pd.read_csv('ratings_1m.csv')
print(df.head())

sentence=[]
dic={}
day=0
df =df.sort_values(by='timestamp')[['timestamp','userid','itemid']]

for item in df.values:
    if day != item[0]:
        for key in dic:
            sentence.append(dic[key])
        dic={}
        day=item[0]
    try:
        dic[item[1]].append(str(int(item[2])))
    except:
        dic[item[1]]=[str(int(item[2]))]

for key in dic:
    sentence.append(dic[key])

print(sentence[0:6])