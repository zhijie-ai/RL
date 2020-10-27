#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/10/27 13:40                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import requests
from bs4 import BeautifulSoup
import re

# url = 'http://www.chinapoesy.com/ShiJingList_5.html'
url = 'https://www.shicimingju.com/chaxun/zuozhe/13046.html'
# url = 'https://www.shicimingju.com/chaxun/zuozhe/13046_3.html'

def get_poetry(url,f):
    res = requests.get(url).text
    soup = BeautifulSoup(res,'lxml')
    title = [t.text for t in soup.find_all('h3')]
    poetry = [''.join(p.text.replace('展开全文\n\n','').replace('\n收起\n','').split()) for p in soup.find_all('div',class_='shici_content')]
    print(poetry)
    poetry = zip(title,poetry)
    for t,p in poetry:
        line = '{}:{}'.format(t,p.strip())
        f.write(line+'\n')

urls =['https://www.shicimingju.com/chaxun/zuozhe/13046_{}.html'.format(i) for i in range(2,11)]
urls.insert(0,url)

with open('poetry.text','w',encoding='utf8') as f:
    for url in urls:
        get_poetry(url,f)

