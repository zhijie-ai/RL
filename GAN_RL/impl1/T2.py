#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/14 17:30                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np

behavior_filename = "data/e3_behavior_v1.txt"
category_filename = "data/e3_news_category_v1.txt"
feature_filename = "data/e3_user_news_feature_v1.txt"
splitter = '/t'

# 1. data_behavior
max_disp_size = 0

data_behavior = [[] for x in range(5)]

fd = open(behavior_filename)
for row in fd:
    row = row.split()[0]
    row = row.split(splitter)
    u_id = int(row[0])
    time_stamp = int(row[1])
    disp_list = list(map(int, row[2].split(',')))
    max_disp_size = max(max_disp_size, len(disp_list))
    pick_list = list(map(int, row[3].split(',')))
    data_behavior[u_id].append([time_stamp, disp_list, pick_list])
fd.close()

for i in range(len(data_behavior)):
    data_behavior[i] = sorted(data_behavior[i], key=lambda x: x[0])

import json
# print(json.dumps(data_behavior, indent=4))
print(data_behavior)

max_category=0
min_category=0
movie_category={}#e.g movie_category[1]=[1,2,3,4]
fd = open(category_filename)
for row in fd:
    row=row.split()[0]
    row = row.split(splitter)
    news_id=int(row[0])
    news_cat = list(map(int,row[1].split(',')))
    news_cat=list(np.array(news_cat)+1) # let category start from 1. leave category 0 for non-clicking
    movie_category[news_id]=news_cat
    max_category=max(max_category,max(news_cat))
    min_category=min(min_category,min(news_cat))

fd.close()
print(movie_category,min_category,max_category)
print(','.join(list(map(str,map(lambda x:round(x,2),np.random.uniform(0,1,20))))))

user_news_feature={}
fd = open(feature_filename)
for row in fd:
    row=row.split()[0]
    row = row.split(splitter)
    key = 'u'+row[0]+'n'+row[1]
    user_news_feature[key] = list(map(float,row[2].split(',')))
fd.close()
print(user_news_feature)