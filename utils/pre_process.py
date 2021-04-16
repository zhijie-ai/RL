#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/7/27 15:18
@Author  : tyang
@Email   : tuyang@yijiupi.com
@File    : pre_process.py
@Description: 

"""
import re


def preprocess_text(text):
    """处理 特殊统一没有意义的字符"""
    extra_list = ["【.*】", "[（）()_\[\]]"]
    for y in extra_list:
        text = re.sub(y, " ", text)
    """处理 统一没有意义的字符"""
    delete_list = ['含经销商赠品等政策', '（含经销商赠品等政策）', '作废',  '复制', '-复制', '活动', '（活动）']
    for x in delete_list:
        text = text.replace(x, "")
    return text


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def insert_brand_dict(jibea_object, brand_list):
    for word in brand_list:
        if isinstance(word, str):
            jibea_object.suggest_freq(word, tune=True)

    return jibea_object


