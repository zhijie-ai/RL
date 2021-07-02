#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/7 16:11                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import time,datetime
from utils.txt_process import read_sql
from utils.config import project_root_path, root_path
import utils.load_data as ld
from utils.yjp_decorator import cost_time_def
from multiprocessing.pool import ThreadPool
import pandas as pd


city_recall_path = project_root_path+'/GAN_RL/yjp/data/sqls2/exposure.sql'
city_recall_sql = read_sql(city_recall_path)

@cost_time_def
def train(date_s,city_recall_sql,hostname=None,port=None):
    t1 = time.time()
    print('starting time :{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))))

    city_recall_sql = city_recall_sql.replace('{}',date_s)
    # hostname = 'haproxy.release.yjp.com'
    # port = 21050
    if hostname and port and city_recall_sql:
        import utils.impala_process as imp
        data = imp.get_data_sql_with_columns(hostname,port,city_recall_sql)
    else:# 本地测试
        data = ld.get_data_from_csv(root_path + '/GAN_RL/yjp/data/raw/exposure.csv')

    t2 = time.time()
    print('fetching data cost:{} m,sql:{},{}'.format((t2 - t1) / 60,city_recall_sql,data.shape))
    return data

def get_data(flag=False):
    date_str = [(datetime.datetime.now()+datetime.timedelta(-i)).strftime('%Y%m%d') for i in range(1,8)]
    pool = ThreadPool(7)
    res = []
    for i in date_str:
        res_ = pool.apply_async(train,args=(i,city_recall_sql,'haproxy.cdh.yjp.com',21050))
        res.append(res_)

    pool.close()
    pool.join()
    df = pd.DataFrame()
    for i in res:
        df_ = i.get()
        df = pd.concat([df,df_])
    print(df.shape)

    if flag:
        df.to_csv(root_path + '/GAN_RL/yjp/data/raw/exposure.csv',index=False)
    return df

if __name__ == '__main__':
    get_data(flag=True)