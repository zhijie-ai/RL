# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/27 17:14


# 判断字符串是否是纯中文，有空格也不算
def is_only_chinese(word):
    word = str(word)
    for ch in word:
        if not '\u4e00' <= ch <= '\u9fff':
            return False
    return True


# 判断字符串是否含有中文
def is_contain_chinese(word):
    word = str(word)
    for ch in word:
        if not '\u4e00' <= ch <= '\u9fff':
            return True
    return False


# 判断字符串是否是纯英文
def is_only_english(word):
    word = str(word.replace(' ', '')).lower()
    for ch in word:
        if not '\u0061' <= ch <= '\u007a':
            return False
    return True