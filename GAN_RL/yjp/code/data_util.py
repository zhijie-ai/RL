#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/8 15:11                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import pandas as pd
import datetime
import numpy as np
from GAN_RL.yjp.code.options import get_options
from utils.yjp_decorator import cost_time_def
from sklearn.model_selection import train_test_split
import pickle


class Dataset():
    def __init__(self,args):
        self.data_click = pd.read_csv(args.click_path)
        self.data_exposure = pd.read_csv(args.exposure_path)
        self.model_type = args.user_model
        self.band_size = args.pw_band_size
        self.data_folder = args.data_folder


    @cost_time_def
    def drop_dup_row(self,data,min_count=7):

        def _parse(ser):
            '''
                     user_id          sku_id                 time
            1938238  1543646  40300004041145  2021-04-07 19:39:27
            1664582  1543646  40300004041145  2021-04-07 19:39:45
            1196214  1543646  40300004041145  2021-04-07 19:40:06
            :param ser:
            :return:
            '''
            if 'time' in ser:# 利用modin.pandas 分组功能似乎有这个项
                return 2
            # 定义2min内如果用户对某个sku有多条点击即认定为重复数据

            delta = datetime.timedelta(minutes=2)
            end = ser+delta
            if len(ser) == 1:
                return 1

            # 如果在2min之内，则flag会小于等于0
            flag = [-1]+(end.values[:-1]-ser.values[1:]).tolist()
            flag = -np.sign(flag)#大于0保留

            # tmp = ser.tolist()[1:]+[0]
            # for ind,(t,t_next) in enumerate(zip(ser,tmp)):
            #     if ind==len(tmp)-1:
            #         break
            #     t<=t_next
            #     flag[ind+1]=0

            return flag


        #1. 过滤<min_count的数据
        tmp = data.groupby('user_id').size()
        data = data[np.in1d(data.user_id,tmp[tmp>min_count].index)]


        data['time']=data.time.apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        data = data.sort_values(by=['user_id','sku_id','time'])
        data['flag'] = data.groupby(['user_id','sku_id'])['time'].transform(_parse)
        data.flag = data.flag.astype(int)

        # 过滤重复数据
        data = data[data.flag==1]
        data.drop('flag',axis=1,inplace=True)
        return data

    def filter_(self,data,min_count=7,max_count=10000):
        # 去掉重复数据,把当前7天看成一个session,因为代码里，data_behavior每个用户然有多条记录，但最终也是合并成一条
        data = data.drop_duplicates(['user_id','sku_id'])
        # data = data.sort_values(by=['user_id','time'])
        tmp = data.groupby('user_id').size()
        data = data[np.in1d(data.user_id,tmp[(tmp>=min_count) & (tmp<=max_count)].index)]
        return data

    def split_dataset(self,data):
        data = data.sample(frac=1).reset_index(drop=True)
        user_ids = data.user_id.unique()
        train_user,test_user = train_test_split(user_ids,test_size=0.2)
        train_user,val_user = train_test_split(train_user,test_size=0.1)
        data['split_tag']=3

        train_ind = data[data.user_id.isin(train_user)].index
        data.loc[train_ind,'split_tag']=0
        val_ind = data[data.user_id.isin(val_user)].index
        data.loc[val_ind,'split_tag']=1
        test_ind = data[data.user_id.isin(test_user)].index
        data.loc[test_ind,'split_tag']=2

        data = data.sort_values(by=['user_id','time'])
        return data



    def preprocess_data(self):
        click = self.filter_(self.data_click,min_count=7,max_count=20)
        exposure = self.filter_(self.data_exposure,min_count=7,max_count=200)
        click['is_click'] = 1
        exposure['is_click']=0

        behavior_data = pd.concat([click,exposure])

        # 切分数据
        behavior_data = self.split_dataset(behavior_data)

        sizes = behavior_data.nunique()
        size_user = sizes['user_id']

        print(sizes,behavior_data.head())
        data_behavior = [[] for _ in range(size_user)]

        train_user = []
        vali_user = []
        test_user = []

        for ind,user in enumerate(behavior_data.user_id.unique()):
            data_behavior[ind] = [[], [], []]
            data_behavior[ind][0] = user
            data_u = behavior_data[behavior_data.user_id==user]
            split_tag = list(data_u['split_tag'])[0]
            if split_tag == 0:
                train_user.append(user)
            elif split_tag == 1:
                vali_user.append(user)
            else:
                test_user.append(user)

            data_behavior[ind][1].extend(data_u['sku_id'].tolist())
            data_behavior[ind][2].extend(data_u[data_u.is_click==1]['sku_id'].tolist())#同时间段点击的item


        # new_features = np.eye(size_item)
        filename = self.data_folder+'data_behavior.pkl'
        file = open(filename, 'wb')
        # print (data_behavior)
        pickle.dump(data_behavior, file, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(new_features, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

        filename = self.data_folder+'user-split.pkl'
        file = open(filename, 'wb')
        pickle.dump(train_user, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(vali_user, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_user, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()


if __name__ == '__main__':
    args = get_options()
    dataset = Dataset(args)

    dataset.preprocess_data()


