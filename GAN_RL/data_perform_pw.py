#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/7 18:27                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
def data_perform(user_set,b_size):
    #(1) [session,news_id(对于每个user，从0开始)] SparseTensor的indices
    # 没一个session代表依次曝光。从0开始，按照user,然后时间排序。eg:session 0:user 0 at time 0;session 1:user 0 at time 1
    display_tensor_indice=[]

    # (2) [session],长度和顺序和disp_tensor_indices一样，只是不要和第二列[news_id]
    # 这个数据如果不好准备，可以直接在TensorFlow里面用tf.split操作
    display_tensor_indice_split=[]

    # (3) 和display_tensor_indice是一样的，但是第二列只包含被click了的news_id
    click_tensor_indice=[]

    #(4) 下面这2个稍微比较难理解一点。这里是构造一个triangular matrix,用来aggregate history
    # 比较难用comment解释
    # 三角矩阵
    tril_indice=[]
    tril_value_indice=[]

    # (5) 点击过的news特征。要按某个顺序排序。具体可以看下面的逻辑
    feature_clicked_x=[]
    #(6) 所有news特征(包括点击/未点击).也是要注意顺序
    disp_current_feature_x=[]
    #(7) 等价于display_tensor_indice.index(click_tensor_indice)
    click_sub_index_2d=[]

    # 数总共有多少session，所有用户所有的点击次数,比如所有用户共点击了100个item，则sec_cnt_x=100
    sec_cnt_x=0
    # max number of news(per user)，所有用户里面最大的展示次数(只是这个set里面的用户，共8个set)
    max_news_per_user=0

    for u in user_set:
        # 构造一个triangular matrix的indices
        t_indice=[]
        #data_time[user]=click_t
        # data_time是一个数组，数组索引可看成用户id，每个位置放置的是该用户对应的点击次数
        for kk in range(min(b_size-1,data_time[u]-1)):#在特征维度和点击次数之间选最小值
            t_indice += map(lambda x: [x + kk + 1 + sec_cnt_x, x + sec_cnt_x], np.arange(data_time[u] - (kk + 1)))

        # t_indice=[[14, 0], [15, 1], [16, 2], [17, 3], [18, 4], [19, 5]]
        tril_indice += t_indice# 三角矩阵的索引
        tril_value_indice += map(lambda x: (x[0] - x[1] - 1), t_indice)#索引对应的值，是由索引得到的。

        # 。。。
        # data_click[user].append([click_t,news_dict[id]])
        click_tensor_indice_tmp = map(lambda x: [x[0] + sec_cnt_x, x[1]], data_click[u])
        click_tensor_indice += click_tensor_indice_tmp#可以看做是data_click的数据格式

        #data_disp[user].append([click_t,news_dict[idd]])，在展示的index中根据点击的数据查找
        display_tensor_indice_tmp = map(lambda x: [x[0] + sec_cnt_x, x[1]], data_disp[u])
        # index 求元素所在的索引
        click_sub_index_tmp = map(lambda x: display_tensor_indice_tmp.index(x), click_tensor_indice_tmp)

        #click_sub_index_2d:[0,1,2,3,4]
        click_sub_index_2d += map(lambda x: x + len(display_tensor_indice), click_sub_index_tmp)
        display_tensor_indice += display_tensor_indice_tmp#data_disp[u]
        display_tensor_indice_split += map(lambda x: x[0] + sec_cnt_x, data_disp[u])

        sec_cnt_x += data_time[u]
        #data_news_cnt，每个用户的历史展示次数,max_news_per_user,所有用户里面，最大的展示次数
        max_news_per_user = max(max_news_per_user, data_news_cnt[u])
        disp_current_feature_x += map(lambda x: feature[u][x], [idd[1] for idd in data_disp[u]])
        feature_clicked_x += feature_click[u]

        # click_tensor_indice和display_tensor_indice类似，只是在原始的基础上增加了一个基数
        return click_tensor_indice, display_tensor_indice, \
               disp_current_feature_x, sec_cnt_x, tril_indice, tril_value_indice, \
               display_tensor_indice_split, max_news_per_user, click_sub_index_2d, feature_clicked_x