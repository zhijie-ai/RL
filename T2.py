#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/9/30 16:26                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import schedule
import time


def job(name):
    print("her name is : ", name)

job('xiaojie')
schedule.every(5).seconds.do(job,'xiaojie2')

while True:
    schedule.run_pending()
    time.sleep(1)