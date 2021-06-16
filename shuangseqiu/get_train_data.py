# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2021/6/16 11:58                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
#!/usr/bin/python
# -*- coding:UTF-8 -*-
#coding:utf-8
#author:levycui
#date:20160513
#Description:双色球信息收集

import requests
from bs4 import BeautifulSoup	#采用BeautifulSoup
import os
import re

#伪装成浏览器登陆,获取网页源代码
def getPage(href):
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
    }
    resp = requests.get(href,headers=headers)
    return resp.text

#初始化url 双色球首页
url = 'http://kaijiang.zhcw.com/zhcw/html/ssq/list_1.html'


#===============================================================================
#获取url总页数
def getPageNum(url):
    num =0
    page = getPage(url)
    soup = BeautifulSoup(page)
    strong = soup.find('td',colspan='7')
    if strong:
        result = strong.get_text().split(' ')
        list_num = re.findall("[0-9]{1}",result[1])
        for i in range(len(list_num)):
            num = num*10 + int(list_num[i])
        return num
    else:
        return 0

#===============================================================================
#获取每页双色球的信息
def getText(url):

    for list_num in range(1,getPageNum(url)):	#从第一页到第getPageNum(url)页
        href = 'http://kaijiang.zhcw.com/zhcw/html/ssq/list_'+str(list_num)+'.html' #调用新url链接
        # for listnum in len(list_num):
        page = BeautifulSoup(getPage(href))
        em_list = page.find_all('em')	#匹配em内容
        div_list = page.find_all('td',{'align':'center'})	#匹配 <td align=center>这样的内容

        #初始化n
        n = 0
        #将双色球数字信息写入num.txt文件
        fp = open("num.txt" ,"w")
        for div in em_list:
            text = div.get_text()
            n=n+1
            if n==7:
                text = text + "\n"
                n=0
            else:
                text = text + ","
            fp.write(str(text))
        fp.close()

        #将日期信息写入date.txt文件
        fp = open("date.txt" ,"w")
        for div in div_list:
            text = div.get_text().strip('')
            list_num = re.findall('\d{4}-\d{2}-\d{2}',text)
            if len(list_num) == 0:
                continue
            elif len(list_num) > 0:
                list_num = str(list_num[::1][0])
                fp.write(str(list_num)+'\n')
        fp.close()

        #将num.txt和date.txt文件进行整合写入hun.txt文件中
        #格式如下：
        #('2016-05-03', '09,12,24,28,29,30,02')
        #('2016-05-01', '06,08,13,14,22,27,10')
        #('2016-04-28', '03,08,13,14,15,30,04')
        #
        fp01 = open("date.txt","r")
        a=[]
        for line01 in fp01:
            a.append(line01.strip('\n'))
        fp01.close()

        fp02 = open("num.txt","r")
        b=[]
        for line02 in fp02:
            b.append(line02.strip('\n'))

        fp02.close()

        fp = open("hun.txt" ,"a")
        for cc in zip(a,b):	#使用zip方法合并
            fp.write(str(cc) + '\n')
        fp.close()


#===============================================================================

if __name__=="__main__":

    # pageNum = getPageNum(url)
    getpagetext = getText(url)