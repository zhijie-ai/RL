#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/29 上午11:56                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import re
RE_WORD = re.compile(r'\w+')
class Sentence:
    def __init__(self, text):
        self.text = text
        self.words = RE_WORD.findall(text)  # re.findall函数返回一个字符串列表，里面的元素是正则表达式的全部非重叠匹配

    def __getitem__(self, index):
        print(index,'AAAAAAAAAA')
        return self.words[index]

s = Sentence('The time has come')
for i in s:
    print(i)

print(s[1])