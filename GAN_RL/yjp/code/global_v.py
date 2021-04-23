#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/20 16:15                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------

def _init():
    global _global_dict
    _global_dict={}

def set_value(name,value):
    _global_dict[name]=value

def get_value(name,defValue='UNK'):
    try:
        return _global_dict[name]
    except KeyError:
        return defValue