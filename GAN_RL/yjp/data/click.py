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
import time
from utils.txt_process import read_sql
from utils.yjp_ml_log import log
from utils.config import project_root_path, root_path
import utils.load_data as ld
from utils.yjp_decorator import cost_time_minute


city_recall_path = project_root_path+'/GAN_RL/yjp/data/sqls2/click.sql'
city_recall_sql = read_sql(city_recall_path)


@cost_time_minute
def train(hostname='haproxy.cdh.yjp.com',port=21050,flag=True):
    t1 = time.time()
    log.logger.info('starting time :{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))))

    # hostname = 'haproxy.release.yjp.com'
    # port = 21050
    if hostname and port and city_recall_sql:
        import utils.impala_process as imp
        data = imp.get_data_sql_with_columns(hostname,port,city_recall_sql)
        print(data.shape)

        if flag:
            data.to_csv(root_path + '/GAN_RL/yjp/data/raw/click.csv',index=False)
    else:# 本地测试
        data = ld.get_data_from_csv(root_path + '/GAN_RL/yjp/data/raw/click.csv')


    t2 = time.time()
    log.logger.info('fetching data cost:{} m'.format((t2 - t1) / 60))
    data.user_id=data.user_id.astype(int)
    data.sku_id=data.sku_id.astype(int)
    return data

if __name__ == '__main__':
    train(flag=True)