#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2014-2020/5/22.year. 易久批电子商务有限公司. All rights reserved.
"""
@Time    : 2020/5/22 9:37
@Author  : tyang
@Email   : tuyang@yijiupi.com
@File    : yjp_scheduler.py
@Description: 
创建定时任务
"""
import time
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from utils.yjp_ml_log import log


def job_func(name):
    print("name is {}, now time is {} ".format(name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))


def set_blockingScheduler(job_func, hour, minute, start_date, end_date, args):
    """ BlockingScheduler 调用start函数后会阻塞当前线程 仅可用在当前你的进程之内，与当前的进行共享计算资源"""
    """ start_date 和 end_date 格式为 2020-05-21 23:59:59 """
    """scheduler.add_job(job, 'cron', hour='0-23', minute='51', start_date='2020-05-22 00:00:00', 
    end_date='2020-05-21 23:59:59') """
    scheduler = BlockingScheduler()
    # scheduler.add_job(job, 'cron', hour='0-23', minute='51', start_date='2020-05-22 00:00:00', end_date='2020-05-21
    #  23:59:59')
    scheduler.add_job(job_func, 'cron', hour=hour, minute=minute, start_date=start_date, end_date=end_date, args=args)
    scheduler.start()


def set_backgroundScheduler(job_func, hour, minute, start_date, end_date, job_id, args):
    """BackgroundScheduler 调用start后主线程不会阻塞 在后台运行调度，不影响当前的系统计算运行"""
    scheduler = BackgroundScheduler()
    # 采用非阻塞的方式
    # 采用corn的方式
    log.logger.info(
        '创建 {} 的 backgroundScheduler 任务, start_date {}, end_date {}'.format(job_func.__name__, start_date, end_date))
    scheduler.add_job(job_func, 'cron', hour=hour, minute=minute, id=job_id, start_date=start_date, end_date=end_date, args=args)
    scheduler.start()


if __name__ == '__main__':
    # set_blockingScheduler(job_func, '0-23', '40', '2020-05-22 00:00:00', end_date='2020-05-22 23:59:59',  args=['blockingScheduler'])
    set_backgroundScheduler(job_func, '0-23', '45', '2020-05-22 00:00:00', end_date='2020-05-22 23:59:59',
                            args=['backgroundScheduler'])
    while True:
        print('main-start:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        time.sleep(2)
        print('main-end:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
