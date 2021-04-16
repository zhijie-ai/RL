#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/8/21 17:41
@Author  : tyang
@Email   : tuyang@yijiupi.com
@File    : download_file.py
@Description: 

"""

import requests
import time
from tqdm import tqdm
import sys


def download_file(url, name):
    time0 = time.time()

    response = requests.get(url, stream=True)
    content_size = int(response.headers['Content-Length'])/(1024*1024)
    print("{} total size is: {}MB, start...".format(name, content_size))
    with open(name, "wb") as f:
        #for data in tqdm(response.iter_content()):
        for data in tqdm(response.iter_content(1024*1024), total=content_size, unit='MB', desc=name):
            f.write(data)
    print("\n" + name + " download finished! cost time " + str(time.time() - time0))


def download(url, file_path):
    # verify=False 这一句是为了有的网站证书问题，为True会报错
    r = requests.get(url, stream=True, verify=False)

    # 既然要实现下载进度，那就要知道你文件大小啊，下面这句就是得到总大小
    content_size = int(r.headers['Content-Length'])
    temp_size = 0
    time0 = time.time()

    with open(file_path, "wb+") as f:
        # iter_content()函数就是得到文件的内容，
        # 有些人下载文件很大怎么办，内存都装不下怎么办？
        # 那就要指定chunk_size=1024，大小自己设置，
        # 意思是下载一点写一点到磁盘。
        print("{} total size is: {}K, start...".format(file_path, content_size))
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                temp_size += len(chunk)
                f.write(chunk)
                f.flush()
                #  ############花哨的下载进度部分##############  #
                done = int(50 * temp_size / content_size)
                # 调用标准输出刷新命令行，看到\r回车符了吧
                # 相当于把每一行重新刷新一遍
                if (temp_size / content_size) % 10 == 0:
                    sys.stdout.write("\r[%s%s] %d%%" % ('*' * done, ' ' * (50 - done), 100 * temp_size / content_size))
                    sys.stdout.flush()
        print("\n" + file_path + " download finished! cost time " + str(time.time() - time0))  # 避免上面\r 回车符，执行完后需要换行了，不然都在一行显示
        print("---------------------------------------")



if __name__ == '__main__':
    url = "http://localhost:8000/download?filepath=F:/python_workspace_2020/ai-search/data/info_all.csv"
    name = url.split('/')[-1]
    download(url, name)
