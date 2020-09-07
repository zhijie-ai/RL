#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/10/16 16:33                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import numpy as np

# with open('data.txt','w') as f:
#     for i in range(20):
#         id = np.random.randint(1,21)
#         line = '{},xiaojie{},{}'.format(id,id,np.random.randint(20,30))
#         f.write(line+'\n')

import multiprocessing as mp
import pymysql

db = pymysql.connect(host="sh-cdb-ncocuxkf.sql.tencentcdb.com", user="MStest",
                     password="ookun55Ee", db="ald_MS_test", port=63196)

cursor = db.cursor()
usersvalues=[]

def writeToMysql():
    with open('data.txt', mode='r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line_split = line.split(",");
            sql = 'select * from test where id = {}'.format(line_split[0])
            cursor.execute(sql)
            data = cursor.fetchall()
            if len(data) == 0:
                usersvalues.append((line_split[0], line_split[1],line_split[2]))
        cursor.executemany('insert into test(id,name,age) value(%s,%s,%s)',
                           usersvalues)
        # 修改数据（查询和删除数据同）
        db.commit()


if __name__ == '__main__':
    # for i in range(1):
    #     p = mp.Process(target=writeToMysql)
    #     p.start()
    #     p.join()
    writeToMysql()