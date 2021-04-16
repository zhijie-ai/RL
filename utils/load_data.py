#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/5/9 16:06
@Author  : tyang
@Email   : tuyang@yijiupi.com
@File    : load_data.py
@Description: 

"""

import pandas as pd


def get_data_from_csv(file_path, rows=None):
    try:
        df = pd.read_csv(file_path, usecols=rows, encoding='gbk')
    except UnicodeDecodeError as e:
        print(" need utf-8 ...")
        df = pd.read_csv(file_path, usecols=rows, encoding='utf-8')
    return df

