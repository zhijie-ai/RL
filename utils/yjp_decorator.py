#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/10/23 17:57
@Author  : tyang
@Email   : tuyang@yijiupi.com
@File    : yjp_decorator.py
@Description: 

"""
import functools
import time
from utils.yjp_ml_log import log


def cost_time(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg)  for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            pairs = ['%s=%r' % (k, w) if not len(w) > 10 else '%s=%r' % (k, 'list') for k, w in sorted(kwargs.items())]

            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        log.logger.info('[%0.10fs] %s -> %r ' % (elapsed, name,  result))
        return result

    return clocked

def cost_time_def(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed =(time.time() - t0)/60
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg)  for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            pairs = ['%s=%r' % (k, w) if not len(w) > 10 else '%s=%r' % (k, 'list') for k, w in sorted(kwargs.items())]

            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        log.logger.info('[%0.10f m] %s ' % (elapsed, name))
        return result

    return clocked


@cost_time
def test_func(a, b):
    return a + b


if __name__ == '__main__':
    test_func(1, 2)
