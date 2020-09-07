#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/9/16 13:49                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import random

data = dict()
for i in range(10):
    ind = random.choice('abcdefghijklmn')
    print(ind,end= ' ')
    data[ind] =random.randint(1,10)

print(data)
print(data)

import time
import datetime
import pprint

print(time.ctime(1570602294))
print('开始执行run_task方法的时间{}'.format(time.ctime(1570602294)))
print('上次开始执行run_task方法的时间{}'.format(time.ctime(1570601694)))
print('save2es处理的数据量{}'.format(669747))
print('执行run_task方法花费的时间{} s'.format(205/60))
print('本次run_task方法执行完的时间{}'.format(time.ctime(1570602500)))
print('='*20)
print('开始执行run_task方法的时间{}'.format(time.ctime(1570603101)))
print('上次开始执行run_task方法的时间{}'.format(time.ctime(1570602294)))
print('save2es处理的数据量{}'.format(1336212))
print('执行run_task方法花费的时间{}'.format(417/60))
print('本次run_task方法执行完的时间{}'.format(time.ctime(1570603518)))
print('='*20)
print('开始执行run_task方法的时间{}'.format(time.ctime(1570604118)))
print('上次开始执行run_task方法的时间{}'.format(time.ctime(1570603101)))
print('save2es处理的数据量{}'.format(1346218))
print('执行run_task方法花费的时间{}'.format(416/60))
print('本次run_task方法执行完的时间{}'.format(time.ctime(1570604535)))
print('='*20)
print('开始执行run_task方法的时间{}'.format(time.ctime(1570605135)))
print('上次开始执行run_task方法的时间{}'.format(time.ctime(1570604118)))
print('save2es处理的数据量{}'.format(1839095))
print('执行run_task方法花费的时间{}'.format(599/60))
print('本次run_task方法执行完的时间{}'.format(time.ctime(1570605735)))
print('='*20)
print('开始执行run_task方法的时间{}'.format(time.ctime(1570606336)))
print('上次开始执行run_task方法的时间{}'.format(time.ctime(1570605135)))
print('save2es处理的数据量{}'.format(2014868))
print('执行run_task方法花费的时间{}'.format(662/60))
print('本次run_task方法执行完的时间{}'.format(time.ctime(1570606998)))
print('='*20)
print('开始执行run_task方法的时间{}'.format(time.ctime(1570607599)))
print('上次开始执行run_task方法的时间{}'.format(time.ctime(1570606336)))
print('save2es处理的数据量{}'.format(1892693))
print('执行run_task方法花费的时间{}'.format(662/60))
print('本次run_task方法执行完的时间{}'.format(time.ctime(1570608262)))


print(time.ctime(1569821526))
print(time.ctime(1569824348))
print(time.time())
print(datetime.datetime.fromtimestamp(time.time()))
print(datetime.datetime.fromtimestamp(time.time()))
print(time.ctime(time.time()))

print('A'*20)
print(datetime.datetime(2019,10,11,22,55,22,533000).timestamp())
print(datetime.datetime(2019,10,11,23,50,00,224000).timestamp())