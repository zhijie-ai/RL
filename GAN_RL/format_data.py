#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/14 20:21                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
def format_data():
    size_user=6
    behavior_filename='data/e3_behavior_v1.txt'
    category_filename='data/e3_news_category_v1.txt'
    feature_filename='data/e3_user_news_feature_v1.txt'
    splitter='/t'

    # 1. data_behavior
    max_disp_size=0

    #按时间序列排序的数据
    data_behavior=[[] for _ in range(size_user)]
    fd = open(behavior_filename)
    for row in fd:
        row = row.split()[0]
        row = row.split(splitter)
        u_id=int(row[0])
        time_stamp = int(row[1])
        disp_list=list(map(int,row[2].split(',')))
        max_disp_size=max(max_disp_size,len(disp_list))
        pick_list=list(map(int,row[3].split(',')))
        data_behavior[u_id].append([time_stamp,disp_list,pick_list])
    fd.close()
    k=10

    # data_behavior[1]=[1584090631,[1,2,3],[5,6,7,8]],有可能是N维3列
    for i in range(len(data_behavior)):
        data_behavior[i]=sorted(data_behavior[i],key=lambda x:x[0])# 按时间戳排序

    # 1.1 click and disp behavior

    # 2. category,处理每个news_id的类别。并记录一个最大的类别和最小的类别
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

    KK= max_category+1
    if max(movie_category.keys())!=len(movie_category.keys()):
        print('warning:news category wrong')
        exit()
    else:
        size_movie=len(movie_category.keys())+1


    # movie_category[0]=[0]

    # 3. feature
    user_news_feature={}#记录了用户对某个item的特征
    fd = open(feature_filename)
    for row in fd:
        row=row.split()[0]
        row = row.split(splitter)
        key = 'u'+row[0]+'n'+row[1]
        user_news_feature[key] = list(map(float,row[2].split(',')))
    fd.close()
    #4. save synthetic data
    data_parameter = [KK,k,size_user,size_movie]

    # another set of data,<--- this is what we finally use
    # 这部分主要是为了把news_id按照每个user 从0开始排序
    data_click = [[] for _ in range(size_user)]
    data_disp=[[] for _ in range(size_user)]
    data_time = np.zeros(size_user,dtype=np.int)#次数
    data_news_cnt=np.zeros(size_user,dtype=np.int)
    feature=[[] for _ in range(size_user)]#每个用户所有展示的item的特征
    feature_click=[[] for _ in range(size_user)]#每个用户所有click的item特征

    #data_behavior[u_id].append([time_stamp,disp_list,pick_list])
    print(len(data_behavior))
    for user in range(len(data_behavior)):
        # (1) count number of click
        click_t=0 # 每个用户的点击item的个数
        for event in range(len(data_behavior[user])):
            pick_list=data_behavior[user][event][2]
            click_t+=len(pick_list)#splitter    a event with 2 clicking to 2 events
        data_time[user]=click_t #假设为10
        # (2)
        # news_dict[1]=0,为disp id,记录的是每个用户所有展示的item id,value 代表第几个展示的item
        # eg news_dict[1]=5,代表id为1的item在当前用户中所有展示列表中处于第五个位置。data_behavior是按时间排序的
        news_dict={}
        feature_click[user]=np.zeros([click_t,20])# 每个用户总的点击次数
        click_t=0
        # [time_stamp, disp_list, pick_list]
        for event in range(len(data_behavior[user])):
            disp_list=data_behavior[user][event][1]
            pick_list=data_behavior[user][event][2]
            for id in disp_list:#展示
                if id not in news_dict:
                    news_dict[id]=len(news_dict)#for each user ,news id start from 0
            for id in pick_list:#点击，如果用户没有点击，不进入for循环
                data_click[user].append([click_t,news_dict[id]])
                key = 'u'+str(user)+'n'+str(id)
                #user_news_feature[key] 是一个vector
                feature_click[user][click_t]=user_news_feature[key]
                for idd in disp_list:
                    data_disp[user].append([click_t,news_dict[idd]])
                click_t+=1# splitter a event with 2 clickings to 2 events


        print('news_dict',news_dict)
        data_news_cnt[user]=len(news_dict)#每个用户的历史展示个数

        feature[user]=np.zeros([data_news_cnt[user],20])# 每个用户所有展示的item，应该就是论文中W

        for id in news_dict:
            key = 'u'+str(user)+'n'+str(id)
            feature[user][news_dict[id]]=user_news_feature[key]
        feature[user]=feature[user].tolist()
        feature_click[user]=feature_click[user].tolist()
    return data_click,data_disp,feature,data_time,data_news_cnt,data_parameter,feature_click,news_dict

data = format_data()
print('data_click',data[0][0])
print('data_disp',data[1][0])
# print('feature',format_data()[2])
# print('data_time',format_data()[3])# 每个用户总的点击item次数，len(pick_list)
# print('data_news_cnt',format_data()[4])# 每个用户总的display次数
# print('data_parameter',format_data()[5])
# print('feature_click',format_data()[6])
# print(list(map(lambda x:x[0],format_data()[1][0])))