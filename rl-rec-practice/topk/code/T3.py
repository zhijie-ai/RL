#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/9/30 15:46                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
class CLS():
    def __init__(self,name='xiaojie',age=20):
        self.name=name
        self.age=age
        self.test = 'test'

    def __str__(self):
        dit = self.__dict__
        dict = {key:val for key,val in dit.items() if key not in ['test','actions','rewards','sess']}
        return str(dict)


if __name__ == '__main__':
    cls =CLS()
    print(cls)
