#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/7 17:44                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import threading
import time
from impala.dbapi import connect
import pandas as pd
import os


class MyThread(threading.Thread):
    def __init__(self, threadid, name, day, min):
        threading.Thread.__init__(self)
        self.threadID = threadid
        self.name = name
        self.day = day
        self.min = min

    def run(self):
        print("开始线程：" + self.name)
        run_task(self.name, self.day, self.min)
        print("退出线程：" + self.name)


def run_task(thread_name, day, min):
    hour = 24
    while hour:
        if hour < 11:
            _hour = '0' + str(hour - 1)
        else:
            _hour = hour - 1
        try:
            query_and_write(day, _hour, min)
        except Exception as e:
            print(str(day) + str(_hour) + '-----need--rey--try-' + str(e))
            time.sleep(10)
            continue
        time.sleep(2)
        print("%s: %s" % (thread_name, time.ctime(time.time())))
        hour -= 1


def impala_conn_exec(sql):
    conn = connect(host='*.*.*.*', port=21050)
    cur = conn.cursor()
    cur.execute(sql)
    cur.close
    result = cur.fetchall()
    return result


def query_and_write(_date, _hour, _min):
    """
    初始化sql，调用查询并写入CSV
    :param _date:
    :param _time:
    :return:
    """
    sql = "select * " \
          "from 表名 m where date= '{0}' " \
          "and substr(cast(time as string),1,15) = '{0} {1}:{2}'/*SA(production)*/".format(_date, _hour, _min)
    print(sql)
    result = impala_conn_exec(sql)
    # result = []
    df = pd.DataFrame(data=result)
    path = _date
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    file_name = "data_{0}-{1}-{2}.csv".format(_date, _hour, _min)
    df.to_csv(path + '/' + file_name, encoding="utf-8", index=False)


# 按日期范围循环 遍历区间开闭示意: [起始日期，截止日期)
for j in range(5, 7):
    daterange = "2020-02"
    if j < 10:
        daterange = daterange + "-0" + str(j)
    else:
        daterange = daterange + "-" + str(j)
    print(daterange)

    # 创建新线程
    threads = []
    for i in range(0, 6):
        if i < 10:
            id = "thread0" + str(i)
        else:
            id = "thread" + str(i)
        threadName = MyThread(1, id, daterange, i)
        threads.append(threadName)
    try:
        # 启动线程
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
            time.sleep(1)

        # 等待所有线程结束
        for thread in threads:
            thread.join()

    except Exception as e:
        print(e)

    j += 1

print("退出主线程")