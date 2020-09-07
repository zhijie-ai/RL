#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/1 23:19                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
def hello(name,*,age=20,sex=1):# *号后面的参数在传入是必须附加参数的名字
    print('hello {},{},{}'.format(name,age,sex))

hello('word',age = 25)

def func():
    x,feat1,feat2,feat3,feat4,y=0
    return x,feat1,feat2,feat3,feat4,y

x,*feat,y = func()#除第一个和最后一个返回值外，中间所有的返回值都会保存到feat变量中