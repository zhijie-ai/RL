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
import modin.pandas as pd
import datetime
import numpy as np


class Dataset():
    def __init__(self,args):
        self.data_click = pd.read_csv(args.click_path)
        self.data_exposure = pd.read_csv(args.exposure_path)
        self.model_type = args.user_model
        self.band_size = args.pw_band_size



    def concat(self,x,y):
        y['is_click']=1
        return pd.merge(x,y,on=['user_id','sku_id'],how='left')


    def drop_dup_row(self,data):

        def _parse(ser):
            if 'time' in ser:# 利用modin.pandas 分组功能似乎有这个项
                return 2
            # 定义2min内如果用户对某个sku有多条点击即认定为重复数据
            ser = ser.sort_values()
            flag = np.ones(len(ser))#flag=0丢弃
            tmp = ser.tolist()[1:]+[0]
            delta = datetime.timedelta(minutes=2)
            if len(ser) == 1:
                return 1

            for ind,(t,t_next) in enumerate(zip(ser,tmp)):
                if ind==len(tmp)-1:
                    break
                t+delta<=t_next
                flag[ind+1]=0

            return flag



        data['time']=data.time.apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        data['flag'] = data.groupby(['user_id','sku_id'])['time'].transform(_parse)
        data.flag = data.flag.astype(int)

        # 过滤重复数据
        data = data[data.flag==1]
        data.drop('flag',axis=1,inplace=True)
        return data

    def preprocess_data(self,click,exposure):
        # 去掉重复数据
        data_click = self.drop_dup_row(self.data_click)
        data_exposure = self.drop_dup_row(self.data_exposure)
        data = self.concat(data_exposure,data_click)

        data['time_x'] = data.time_x.apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        data.time_y = data.time_y.fillna('9999-07-09 09:00:00')
        data['time_y'] = data.time_y.apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))



