#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author：tyang, Leslie Dang
# File:   cost_time.py
# Initial Data : 2021/2/9 15:04
# Description: function decorator

import time
from functools import wraps
import warnings

from utils.yjp_ml_log import log


def deprecated(func):
    """
    废弃函数装饰器：该函数后期不再维护，建议新项目不要使用。
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("The function of '{}' is deprecated.".format(func.__name__), DeprecationWarning)
        log.logger.warn("The function of '{0}' is deprecated, please use 'help({0})' to know more details."
                        .format(func.__name__))
        return func(*args, **kwargs)

    return wrapper


def time_decorator(unit='s'):
    """
    时间装饰器
    :param unit: 时间单位
    :return:
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time_start = time.time()
            result = func(*args, **kwargs)
            if unit == "min":
                log.logger.info('function of "{}" execute success, cost time {} min'
                                .format(func.__name__, round((time.time() - time_start) / 60, 5)))
            else:
                log.logger.info('function of "{}" execute success, cost time {} s'
                                .format(func.__name__, round(time.time() - time_start, 5)))
            return result

        return wrapper

    return decorator


def cost_time(func):
    """时间装饰器2"""

    @wraps(func)
    def clocked(*args, **kwargs):
        time_0 = time.time()
        result = func(*args, **kwargs)
        log.logger.info('The function of "{}" cost time: {}s'.format(func.__name__, round(time.time() - time_0, 5)))
        return result

    return clocked


def cost_time_minute(func):
    """时间装饰器3(时间单位为：min)"""

    @wraps(func)
    def clocked(*args, **kwargs):
        time_0 = time.time()
        result = func(*args, **kwargs)
        log.logger.info('The function of "{}" cost time: {} minutes.'
                        .format(func.__name__, round((time.time() - time_0) / 60, 5)))
        return result

    return clocked


if __name__ == '__main__':
    @time_decorator('s')
    def add(s1, s2):
        time.sleep(6)
        return s1 + s2


    print(add(1, 2))
