#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/22 下午5:32                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd
import json
import numpy as np
from collections import deque
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


u_cols = ['user_id','age','sex','occupation','zip_code']
users = pd.read_csv('data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

# reading rating file
r_cols = ['user_id','movie_id','rating','unix_timestamp']
ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

# reading items file
i_cols = ['movie_id','movie_title','release date','video release date','IMDb URL','unknown','Action',
          'Adventure','Animation','Children\s','Comedy','Crime','Documentary','Drama','Fantasy','File-Noir',
          'Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
items = pd.read_csv('data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('data/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
# print(ratings_train.shape, ratings_test.shape)

# encode string into intergers in new_users
new_users = users.values
new_items = items.values
new_items = np.delete(new_items,[1,2,3,4],1)

# convert ratings to 1/2(ratings-3)
new_ratings = ratings.values
new_ratings = new_ratings.astype(float)
for row in new_ratings:
    a = row[2]
    a=(a-3)/2
    row[2]=a

a = LabelEncoder()
new_users[:,2] = a.fit_transform(new_users[:,2])
new_users[:,3] = a.fit_transform(new_users[:,3])
new_users[:,4] = a.fit_transform(new_users[:,4])

user_train, user_test = train_test_split(new_users, test_size = 0.2)
# print(user_train.shape,user_test.shape)

# kmeans = KMeans(5,random_state=0).fit(user_train)
# print(kmeans.cluster_centers_.shape)

# begin the drr

def cluster_data(num_clusters):
    # n clusters for users: young/old, male/female, 16 jobs, zip code
    train_data = np.delete(user_train,0,1) # training user data without user_id
    kmeans = KMeans(num_clusters,random_state=0).fit_predict(train_data)
    b={}
    unique = set(kmeans)
    for i in unique:
        num = 0
        for j in kmeans:
            if j==i:
                num+=1
        b[i]=num

    # find the n nearest data points of each cluster center and get their user_id
    kmeans = KMeans(n_clusters=num_clusters,random_state=0).fit(train_data)
    center = kmeans.cluster_centers_
    # 与中心点距离最近的点
    closest, _ = pairwise_distances_argmin_min(center, train_data)
    train_userid = []# 与中心点距离最近的点的用户id
    for i in closest:
        train_userid.append(user_train[i][0])

    # find the clusters for test users
    test_clusters = {}#存储的是id->cluster的映射关系
    # 测试集中每条样本所在的簇
    index = kmeans.predict(np.delete(user_test, 0, 1))
    for i in range(len(user_test)):
        test_clusters[int(user_test[i][0])] = int(index[i])

    # find all relevant movies_id and ratings of these picked train users
    train_movieid = {}
    for i in  train_userid:
        item=[]
        for row in new_ratings:
            movie = {}
            if i == row[0]:
                movie['movie_id']=int(row[1])
                movie['ratings']=float(row[2])
                item.append(movie)
        train_movieid[int(i)] = item

    #convert items into dictionary
    movie_items={}
    for row in new_items:
        movie_items[int(row[0])] = list(row[1:])

    # find all relevant movies_id and ratings of these picked test users
    test_movieid = {}
    for i in user_test[:,0]:
        item = []
        for row in new_ratings:
            movie = {}
            if i == row[0]:
                movie['movie_id']=int(row[1])
                movie['ratings']=float(row[2])
                item.append(movie)
        test_movieid[int(i)] = item

    whole_movieid={}
    for i in new_users[:,0]:
        item=[]
        for row in new_ratings:
            movie={}
            if i == row[0]:
                movie['movie_id'] = int(row[1])
                movie['ratings'] = float(row[2])
                item.append(movie)
        whole_movieid[int(i)]=item


    with open('data/dumped/user_movies.json', 'w') as fp:
        json.dump(train_movieid,fp)

    with open('data/dumped/test_user_movies.json', 'w') as fp:
        json.dump(test_movieid,fp)

    with open('data/dumped/whole_user_movies.json', 'w') as fp:
        json.dump(whole_movieid, fp)

    with open('data/dumped/movie_items.json', 'w') as fp:
        json.dump(movie_items, fp)

    with open('data/dumped/test_users.json', 'w') as fp:
        json.dump(test_clusters, fp)

    np.save('drr/data/user_train', user_train)
    np.save('drr/data/user_test', user_test)

#process the data to given form(state, action, reward, next_state, recall)
def process_data(data_path,users_path,items_path,history_len):
    with open(data_path,'r') as fp:
        # 用户对电影的评分'737': [{'movie_id': 428, 'ratings': 0.5}, {'movie_id': 186, 'ratings': 1.0},...
        ori_data = json.load(fp)

    whole_data={}
    for k in ori_data.keys():# 遍历的是数据中的每一个userid
        data = {}
        state = deque()
        n_state=deque()
        items = ori_data[k]#当前用户所有评过分的电影以及对应的评分,should be chronological,but not
        # ex as :[{'item_id':...,'rating':...}.....]

        state_list = []
        action_list = []
        n_state_list = []
        reward_list=[]

        # 根据历史评分的电影,生成(s,a,s_,r)
        n_state.append(items[0]['movie_id'])#取历史评分电影中的第一部电影
        for i in range(len(items)-1):
            if len(state) < history_len:
                state.append(items[i]['movie_id'])# 也就是说,state是历史中类似点击过的5条id
            else:
                state.popleft()
                state.append(items[i]['movie_id'])

            state_list.append(list(state))
            action_list.append(items[i+1]['movie_id'])# action为下一条
            reward_list.append(items[i+1]['ratings']) # the reward of action
            if len(n_state)<history_len:
                n_state.append(items[i+1]['movie_id'])
            else:
                n_state.popleft()
                n_state.append(items[i+1]['movie_id'])
            n_state_list.append(list(n_state))

        # print (action_list)
        # print( state_list)
        # sample.append((list(state), action, reward, list(n_state), recall))

        data['state_float'] = state_list[history_len-1:]# We only need state in full load,which means 5 elements
        data['action_float'] = action_list[history_len-1:]
        data['reward_float'] = reward_list[history_len-1:]
        data['n_state_float'] = n_state_list[history_len-1:]
        whole_data[int(k)]=data

    with open(items_path,'r') as fp:
        item_embed = json.load(fp)

    # 将用户信息归一化
    user = np.load(users_path,allow_pickle=True)
    new_mat = user[:,1:]
    user[:,1:] = minmax_scale(new_mat,axis=0)

    paddle = 15*[1]

    # 将原有用户的信息字段归一化后,在添加15个字段
    user_embed = {}
    for row in user:
        new_row = list(row[1:])
        new_row.extend(paddle)
        user_embed[row[0]]=new_row

    return pd.DataFrame.from_dict(whole_data),item_embed,user_embed
'''
                         1                       2
s1  [[1, 2, 3], [4, 5, 6]]  [[1, 2, 3], [4, 5, 6]]
s2  [[1, 2, 3], [4, 5, 6]]  [[1, 2, 3], [4, 5, 6]]
s3  [[1, 2, 3], [4, 5, 6]]  [[1, 2, 3], [4, 5, 6]]
'''
