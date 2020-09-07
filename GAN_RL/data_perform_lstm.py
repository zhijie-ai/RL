#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/10 20:58                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
def data_perform(user_set):
    max_news_per_user = 0  # max number of news(per user)
    max_time = 0

    size_user = len(user_set)  # 这个batch的用户数目

    disp_tensor_indices = []  # [user_id(对于这个batch，从0开始数), 时间（对于每个user，从0开始数，整数），news_id（对于每个user，从0开始数）] SparseTensor的indices
                              # 本来只是sparsetensor的话，除了时间以外，别的不需要从0开始数，但是因为算precision的时候需要把它变成densetensor，所以要从0开始，让tensor size更小

    disp_tensor_indices_split = []  # [user_id, 时间], 长度和顺序都和disp_tensor_indices一样，只是不要第三列[news_id]
                                    # 这个数据如果不好准备，可以直接在tensorflow里面用tf.split操作

    click_tensor_indices = []  # 和disp_tensor_indices一样, 但是这里的第三列只包含被click了的news_id

    u_feature = []

    click_sub_index = []  # 因为click_tensor_indices是disp_tensor_indices的一个子集，找一下对应的sublist index.. 在tf.gather有用到

    # （1） 找出最长的时间。data_time[u]表示用户u对应的时间长度
    for u in user_set:
        max_time = max(max_time, data_time[u])

    user_time_dense = np.zeros([size_user, max_time], dtype=np.float32)
    click_feature = np.zeros([max_time, size_user, _f_dim])  # 这个作为LSTM的input
    for u_idx in range(size_user):
        u = user_set[u_idx]

        click_tensor_indices_tmp = []
        disp_tensor_indices_tmp = []

        for x in data_click[u]:
            t, click_id = x  # 这个表示：用户u在时间t点击了新闻click_id
            click_feature[t][u_idx] = feature[u][click_id]  # 把对应的特征collect起来，作为LSTM的input
            click_tensor_indices_tmp.append([u_idx, t, click_id])
            user_time_dense[u_idx, t] = 1.0

        click_tensor_indices = click_tensor_indices + click_tensor_indices_tmp

        for x in data_disp[u]:
            t, disp_id = x
            disp_tensor_indices_tmp.append([u_idx, t, disp_id])
            disp_tensor_indices_split.append([u_idx, t])
            u_feature.append(feature[u][disp_id])  # feature

        click_sub_index_tmp = map(lambda x: disp_tensor_indices_tmp.index(x), click_tensor_indices_tmp)  #找sublist index
        click_sub_index += map(lambda x: x+len(disp_tensor_indices), click_sub_index_tmp)

        disp_tensor_indices = disp_tensor_indices + disp_tensor_indices_tmp
        max_news_per_user = max(max_news_per_user, data_news_cnt[u])

    return size_user, max_time, max_news_per_user, \
           disp_tensor_indices, disp_tensor_indices_split, np.array(u_feature), click_feature, click_sub_index, \
           click_tensor_indices, user_time_dense