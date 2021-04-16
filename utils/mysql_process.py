#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pymysql
import utils.config as config
from utils.yjp_ml_log import log
section = 'AI_MYSQL_RELEASE'


def set_section(section_en):
    global section
    section = section_en

def get_mysql_db():
    # 打开数据库连接
    conf = config.get_value('conf')
    # section = 'AI_MYSQL_RELEASE'  # 切换环境只要修改对应配置文件的名称即可
    db = pymysql.connect(
        host=conf.get(section, 'host'),
        port=conf.getint(section, 'port'),
        user=conf.get(section, 'user'),
        passwd=conf.get(section, 'pass'),
        db=conf.get(section, "database"),
        charset='utf8')
    return db


def get_data_mysql(sql):
    db = get_mysql_db()
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        return results
    except Exception as e:
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
    # 关闭数据库连接
    db.close()


def multi_execute_mysql(list):
    db = get_mysql_db()
    cursor = db.cursor()
    try:
        for sql in list:
            log.logger.info(sql)
            cursor.execute(sql)
            db.commit()
        cursor.close()
        db.close()
        log.logger.info("插入数据成功...")
    except Exception as e:
        log.logger.error("插入数据失败...")
        log.logger.error(e)


def many_execute_mysql(temp, data):

    db = get_mysql_db()
    cursor = db.cursor()

    try:
        cursor.executemany(temp, data)
        db.commit()
        log.logger.info("插入数据成功...")
    except Exception as e:
        db.rollback()
        log.logger.error("插入数据失败...")
        log.logger.error(e)
    finally:
        db.close()


def single_execute_mysql(sql):
    db = get_mysql_db()
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()

