# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/20 14:04

"""
正则表达式，仅获取字符串中的中文、英文
"""

import re
only_cn_en = "[^\u4e00-\u9fa5^a-z^A-Z]"
only_cn = "[^\u4e00-\u9fa5]"


def filter_chinese_and_english(str_in):
    cop = re.compile(only_cn_en)  # 匹配不是中文、大小写的其他字符
    return cop.sub('', str_in)  # 返回中英文字符（将其他字符用空替换)


def filter_chinese(str_in):
    cop = re.compile(only_cn)  # 匹配不是中文的其他字符
    return cop.sub('', str_in)  # 返回中文字符（将其他字符用空替换)


def str_filter(str_in):
    cop = re.compile(only_cn_en)    # 匹配不是中文、大小写的其他字符
    return cop.sub('', str_in)    # 返回中英文字符（将其他字符用空替换


def regex_filter(str_in, regex_str):
    cop = re.compile(regex_str)
    return cop.sub('', str_in)  # 返回中英文字符（将其他字符用空替换


if __name__ == '__main__':
    word = '冼发水ai'
    import time

    time0 = time.time()
    filtered = filter_chinese(word)
    print(time.time() - time0)
    print(filtered)
