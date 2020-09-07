#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/9/28 11:43                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import requests
from bs4 import BeautifulSoup
import re

soup = BeautifulSoup(open('data/env.html',encoding='utf8'),'lxml')
with open('data/gym_env.txt','w',encoding='utf8') as f:
    p_tag = soup.p
    for game in p_tag.stripped_strings :
        start = game.index('(')+1
        end = game.index(')')
        name = game[start:end]
        f.write(name+'\n')
