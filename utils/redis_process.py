#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/4/30 10:43
@Author  : tyang
@Email   : tuyang@yijiupi.com
@File    : redis_process.py
@Description: 

redis 操作类
"""
import redis
from utils.yjp_ml_log import log
import ast


def init_redis_pool(host='bi-ml-redis.yjp.com', port=6379, db=2, password='Yijiupi_2019'):
    # redis_ = redis.Redis(host='bi-ml-redis.yjp.com', port=6379, db=2, password='Yijiupi_2019')
    # pool管理对一个redis server的所有连接避免每次建立、释放连接的开销
    # host = '197.168.12.58'
    pool = redis.ConnectionPool(host=host, port=port, db=db, password=password, decode_responses=True,
                                socket_connect_timeout=200)
    redis_ = redis.StrictRedis(connection_pool=pool)
    return redis_


def save_to_redis(key_, value):
    # redis_ = redis.Redis(host='bi-ml-redis.yjp.com', port=6379, db=2, password='Yijiupi_2019')
    redis_ = init_redis_pool()
    redis_.set(key_, value)
    log.logger.info("save ' {} ' to redis successfully, and key is ".format(key_))


def load_from_redis(key_):
    # redis_ = redis.Redis(host='bi-ml-redis.yjp.com', port=6379, db=2, password='Yijiupi_2019')
    redis_ = init_redis_pool()
    data_bytes = redis_.get(key_)
    strs_ = bytes.decode(data_bytes)
    value = eval(strs_)
    return value
    log.logger.info("load ' {} ' from redis successfully".format(key_))


class Redis:
    def __init__(self, host='bi-ml-redis.yjp.com', port=6379, db=2, password='Yijiupi_2019'):
        # host = '197.168.12.58'
        self.redis_con = init_redis_pool(host, port, db, password)

    def get(self, key):
        return self.redis_con.get(key)

    def set(self, key, value):
        self.redis_con.set(key, value)

    def hget(self, name, key):
        return self.redis_con.hget(name, key)

    def hset(self, name, key, value):
        self.redis_con.hset(name, key, value)

    def hget_dict(self, name, key):
        value = self.hget(name, key)
        if value:
            value_dict = ast.literal_eval(value)
            return value_dict
        else:
            return dict()

    def get_dict(self, key):
        value = self.get(key)
        if value:
            value_dict = ast.literal_eval(value)
            return value_dict
        else:
            return dict()

    def pip_hget(self, name, key):
        return self.redis_con.pipeline().hget(name, key)

    def pip_hset(self, name, key, value):
        self.redis_con.pipeline().hset(name, key, value)
    #
    # def pip_hget_dict(self, name, key):
    #     value = self.hget(name, key)
    #     if value:
    #         value_dict = ast.literal_eval(value)
    #         return value_dict
    #     else:
    #         return dict()

    def pip_excute(self):
        self.redis_con.pipeline().execute()

    def cache(self, key, result, ttl_time):
        """缓存机制..."""
        self.redis_con.set(key, result)
        self.redis_con.expire(key, ttl_time)

    def get_cache(self, key):
        """缓存机制..."""
        return self.redis_con.get(key)


if __name__ == '__main__':

    redis_ = Redis(db=3)
    redis_.set('1', "{'2020-04-10': {'promotion_price': 45.54886896862558, 'sell_price': 47.0, 'cost_price': 42.0, "
                    "'num': 1.0, 'hour_num_distribution': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., "
                    "0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}}")
    print(redis_.get('1'))

    print(redis_.redis_con.exists('promotion-1-9'))




